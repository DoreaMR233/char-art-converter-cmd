"""
基础处理器模块

该模块定义了图像和视频处理器的基础类 `BasedProcessor`，
提供了共享的属性和方法，包括参数初始化、信号处理、
进度条管理、单帧处理等核心功能。

主要功能：
- 统一的参数初始化和验证
- 中断信号处理和优雅退出
- 多线程环境下的进度条位置管理
- 单帧图像到字符画的转换
- 支持多线程和顺序处理两种模式
- 临时文件管理和帧加载
- GPU加速支持

依赖：
- concurrent.futures: 用于多线程处理
- logging: 用于日志记录
- multiprocessing: 用于获取CPU核心数
- platform: 用于平台检测
- signal: 用于信号处理
- threading: 用于线程安全操作
- time: 用于时间计算
- pathlib: 用于路径操作
- torch: 用于GPU加速
- PIL: 用于图像处理
- colorama: 用于彩色输出
"""
import concurrent.futures
import logging
import multiprocessing
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Callable

# 将torch导入移到init_pytorch_and_gpu函数中，确保在信号处理器设置后再导入
from PIL import Image

from ..configs import CHAR_DENSITY_CONFIG, DEFAULT_FONT_SIZE, ERROR_MESSAGES, WARNING_MESSAGES, DEFAULT_COLOR_MODE, \
    DEFAULT_DENSITY
from ..enums import SaveModes
from ..enums.color_modes import ColorModes
from ..enums.file_type import FileType
from ..utils import validate_arguments, load_font, init_pytorch_and_gpu, setup_gpu_memory_limit, format_time, \
    resize_image_for_chars, create_char_image, show_project_status_progress, \
    format_speed, calculate_resized_dimensions, check_ffmpeg_available
from ..utils.file_utils import save_file

logger = logging.getLogger(__name__)

class BasedProcessor:
    """
    基础处理器类，提供图像和视频处理器的共同功能
    Args:
        self.file_type : FileType 文件类型枚举，用于区分图像和视频处理
        self.global_exit_flag : Any 全局退出标志，用于中断处理
        self.should_stop : bool 停止处理标志
        self.density : str 字符密度级别
        self.color_mode : ColorModes 颜色模式
        self.limit_size : Optional[List[int]] 尺寸限制
        self.enable_multithread : bool 是否启动多线程
        self.with_text : bool 是否保存文本
        self.max_workers : int 最大线程数
        self.is_debug : bool 是否为Debug模式
        self.position_list : List[int] 进度条位置列表
        self.used_positions : set 已使用的进度条位置集合
        self.position_lock : threading.Lock 线程锁
        self.char_set : str 字符集
        self.char_count : int 字符集个数
        self.font_size : int 字体大小
        self.font : Any 字体对象
        self.input_path : Path 输入路径
        self.with_image: bool 是否保存帧
        self.output_dir : Path 输出目录
        self.text_path : Optional[Path] 文本输出路径
        self.image_path : Optional[Path] 图像输出路径
        self.temp_frame_paths : Dict[int, Path] 临时帧路径字典
        self.gpu_memory_limit : Optional[float] GPU内存限制
        self.torch : Optional[Any] torch模块
        self.torch_cuda_available : bool CUDA是否可用
        self.device : Optional[Any] 设备对象
        self.enable_gpu : bool 是否启用GPU加速
    """
    
    def __init__(self, args: Any, file_type: FileType) -> None:
        """
        初始化基础处理器
        
        Args:
            args: Any 包含所有处理参数的对象，可能包含exit_flag属性
            file_type: FileType 文件类型枚举，用于区分图像和视频处理
        
        Returns:
            None: 无返回值
        """
        # 存储文件类型
        self.file_type: FileType = file_type
        
        # 验证参数
        validate_arguments(args)
        # 检查FFmpeg是否可用
        check_ffmpeg_available()
        
        # 定义虚拟标志类，用于在没有提供全局exit_flag时作为默认替代
        class DummyFlag:
            """
            虚拟标志类，用于在没有提供全局exit_flag时作为默认替代
            Args:
                value : bool 标志值，默认为False，表示程序运行状态
            """
            value = False
        
        # 尝试从builtins获取全局exit_flag
        try:
            import builtins
            if hasattr(builtins, 'exit_flag'):
                self.global_exit_flag = builtins.exit_flag
                logger.debug("已从builtins获取全局exit_flag")
            elif hasattr(args, 'exit_flag'):
                self.global_exit_flag = args.exit_flag
                logger.debug("已从参数获取全局exit_flag引用")
            else:
                # 如果没有提供，创建一个默认的False标志
                self.global_exit_flag = DummyFlag()
        except (ImportError, AttributeError):
            # 如果无法访问builtins.exit_flag，尝试从args获取
            if hasattr(args, 'exit_flag'):
                self.global_exit_flag = args.exit_flag
                logger.debug("已从参数获取全局exit_flag引用")
            else:
                # 使用之前定义的DummyFlag类
                self.global_exit_flag = DummyFlag()
        
        # 初始化中断标志
        self.should_stop: bool = False
        if hasattr(self.global_exit_flag, 'value'):
            self.should_stop = self.global_exit_flag.value
        elif self.global_exit_flag:
            self.should_stop = True
        
        # 首先注册信号处理器，确保在导入任何可能阻塞的库之前
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 验证参数
        validate_arguments(args)
        # 检查FFmpeg是否可用
        check_ffmpeg_available()

        # 字符密度级别
        self.density: str = args.density if args.density is not None else DEFAULT_DENSITY
        # 颜色模式
        if args.color_mode is None:
            color_mode_value = DEFAULT_COLOR_MODE
        elif isinstance(args.color_mode, str):
            color_mode_value = ColorModes.from_string(args.color_mode)
        else:
            color_mode_value = args.color_mode
        self.color_mode: ColorModes = color_mode_value
        # 尺寸限制
        self.limit_size: Optional[List[int]] = args.limit_size if args.limit_size is not None else None
        # 是否启动多线程
        self.enable_multithread: bool = not args.no_multithread if args.no_multithread is not None else True
        # 是否保存文本
        self.with_text: bool = args.with_text or False
        # 最大线程数
        if self.enable_multithread:
            cpu_count = max(1, multiprocessing.cpu_count())
            self.max_workers = min(cpu_count if cpu_count <= 2 else cpu_count - 1, 16)
            logger.info(f"根据CPU性能自动设置线程数: {self.max_workers} (可用CPU核心数: {cpu_count})")
        else:
            self.max_workers = 1

        # 是否为Debug模式
        self.is_debug: bool = args.debug or False

        # 初始化进度条位置列表和锁
        self.position_list: List[int] = list(range(1, self.max_workers + 1))
        self.used_positions: set = set()
        self.position_lock: threading.Lock = threading.Lock()
        # 字符集
        self.char_set: str = CHAR_DENSITY_CONFIG.get(self.density, DEFAULT_DENSITY)
        # 字符集个数
        self.char_count: int = len(self.char_set)
        # 字体设置
        self.font_size: int = DEFAULT_FONT_SIZE
        self.font: Any = load_font(self.font_size)

        # 输入路径
        self.input_path: Path = Path(args.input)
        # 是否保存帧
        self.with_image: bool = args.with_image or False
        # 输出路径 - 子类需要根据自己的需求调整获取方式
        self.output_dir: Path
        self.text_path: Optional[Path] = None
        self.image_path: Optional[Path] = None
        
        # 临时文件相关
        self.temp_frame_paths: Dict[int, Path] = {}
        
        # GPU加速设置 - 在信号处理器注册后初始化torch环境
        self.gpu_memory_limit: Optional[float] = args.gpu_memory_limit if hasattr(args, 'gpu_memory_limit') else None
        self.torch: Optional[Any]
        self.torch_cuda_available: bool
        self.device: Optional[Any]
        self.torch, self.torch_cuda_available, self.device = init_pytorch_and_gpu()
        
        # GPU加速判断
        if not args.enable_gpu:
            self.enable_gpu = False
            logger.info("用户已禁用GPU加速")
        else:
            if not self.torch_cuda_available:
                self.enable_gpu = False
                logger.warning(f"{WARNING_MESSAGES['gpu_acceleration_unavailable']}")
            else:
                self.enable_gpu = True
                logger.info("GPU加速已启用")
                setup_gpu_memory_limit(self.torch, self.torch_cuda_available, self.gpu_memory_limit)
    
    def signal_handler(self, signum: int, frame: Any) -> None:
        """
        信号处理器，用于捕获中断信号
        
        Args:
            signum: int 信号编号
            frame: Any 当前堆栈帧
        
        Returns:
            None: 无返回值
        """
        logger.debug(f"收到信号 {signum}，当前帧: {frame}")
        logger.info("接收到中断信号，正在停止处理...")
        
        # 设置停止标志
        self.should_stop = True
        
        # 尝试设置builtins中的exit_flag
        try:
            import builtins
            if hasattr(builtins, 'exit_flag'):
                builtins.exit_flag = True
                logger.debug("已设置builtins.exit_flag为True")
        except Exception as e:
            logger.debug(f"无法设置builtins.exit_flag: {e}")
        
        # 同时更新原有的global_exit_flag
        if hasattr(self, 'global_exit_flag') and hasattr(self.global_exit_flag, 'value'):
            self.global_exit_flag.value = True
        
        # 在子进程中不抛出异常，避免死锁
        if frame and hasattr(frame, 'f_code') and 'runpy' in frame.f_code.co_filename:
            # 这可能是在multiprocessing的spawn过程中
            # 直接返回而不抛出异常，让进程自然退出
            logger.info("在子进程中检测到中断，准备终止")
            return
        
        # 不再抛出异常，而是让进程自然退出
        logger.debug("信号处理器已设置中断标志，等待进程自然退出")
    
    def get_progress_position(self, frame_index: int, is_multithread: bool = True) -> int:
        """
        线程安全地获取进度条位置
        
        Args:
            frame_index: int 帧索引
            is_multithread: bool 是否为多线程模式
            
        Returns:
            int: 分配的进度条位置
        """
        if not is_multithread:
            # 非多线程模式下，始终返回第一个位置
            return self.position_list[0]
        
        with self.position_lock:
            position: int = self.position_list[frame_index % len(self.position_list)]
            
            if position in self.used_positions:
                for pos in self.position_list:
                    if pos not in self.used_positions:
                        position = pos
                        break
            
            self.used_positions.add(position)
            return position
    
    def release_progress_position(self, position: int) -> None:
        """
        释放已分配的进度条位置
        
        Args:
            position: int 需要释放的进度条位置索引
        
        Returns:
            None: 无返回值
        """
        with self.position_lock:
            if position in self.used_positions:
                self.used_positions.remove(position)

    # 创建should_stop回调函数
    def should_stop_callback(self) -> bool:
        """
        检查是否应该停止处理的回调函数

        Returns:
            bool: 是否停止处理
        """
        return self.should_stop

    def process_single_frame(self, is_multithread: bool, frame_index: int, image: Image.Image, save_mode: SaveModes,text_frame_path: Path,image_frame_path: Path,duration: int = 0) -> Tuple[int, int]:
        """
        处理单帧图像并转换为字符画
        
        Args:
            is_multithread: bool 是否在多线程环境下运行
            frame_index: int 帧索引
            image: Image.Image PIL图像对象
            save_mode: SaveModes 保存模式
            text_frame_path: Path 文本帧保存路径
            image_frame_path: Path 图像帧保存路径
            duration: int 帧持续时间
            
        Returns:
            Tuple[int, int]: 包含帧索引和帧持续时间的元组
        """
        # 检查是否需要停止处理
        if self.should_stop:
            raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
        
        # 获取可用的进度条位置
        position: int = self.get_progress_position(frame_index, is_multithread)
        
        try:

            # 调整图片大小
            resized_image = resize_image_for_chars(self.limit_size, image, frame_index, False, self.font_size)
            # 当帧数为第一张时，显示调整后的图片大小
            if frame_index == 0:
                calculate_resized_dimensions(self.limit_size, image, frame_index, True, self.font_size)
            # 生成字符画图片和字符画文本
            output_image, output_text = create_char_image(
                resized_image, self.enable_gpu, self.char_set, self.char_count,
                self.color_mode, self.font, self.torch, self.device, frame_index, position,
                should_stop=self.should_stop_callback
            )
            # 保存字符画文本
            if self.with_text and self.text_path is not None:
                input_content = {'output_text':output_text,'index':frame_index}
                save_file(input_content, text_frame_path, f"保存帧{frame_index}字符画文本", SaveModes.ANIMATED_TEXT,
                          position,should_stop=self.should_stop_callback)
            # 保存字符画图片
            # 记录临时帧路径
            self.temp_frame_paths[frame_index] = image_frame_path
            input_content = {'output_image': output_image, 'index': frame_index}
            save_file(input_content, image_frame_path, f"保存帧{frame_index}字符画到临时文件夹",
                      save_mode, position,should_stop=self.should_stop_callback)
        finally:
            # 确保无论处理成功还是失败，都释放进度条位置
            self.release_progress_position(position)
        return frame_index,duration

    def process_frames_multithreaded(self, total_frames: int, pbar_description: str, process_frame_func: Callable) -> List[int]:
        """
        多线程处理帧
        
        Args:
            total_frames: int 总帧数
            pbar_description: str 进度条描述
            process_frame_func: Callable 处理单帧的函数

        Returns:
            List[int]: 持续时间列表
        """
        # 检查是否需要立即停止（包括全局exit_flag）
        if self.should_stop or hasattr(self, 'global_exit_flag') and self.global_exit_flag:
            logger.info("检测到中断标志，停止处理")
            return []
            
        durations = [0] * total_frames
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with show_project_status_progress(total=total_frames, description=pbar_description, unit='帧') as pbar:
                futures: Set[Future[Any]] = set()
                frame_index = 0
                used_time = time.time()
                
                # 提交任务和收集结果的合并逻辑
                while frame_index < total_frames or futures:
                    # 检查全局exit_flag
                    if hasattr(self, 'global_exit_flag') and self.global_exit_flag:
                        self.should_stop = True
                        break
                        
                    # 提交任务直到达到最大线程数或所有任务都已提交
                    while frame_index < total_frames and len(futures) < self.max_workers:
                        # 检查是否需要停止处理
                        if self.should_stop or (hasattr(self, 'global_exit_flag') and self.global_exit_flag):
                            self.should_stop = True
                            break
                        color_mode_value = self.color_mode.value if isinstance(self.color_mode,Enum) else self.color_mode
                        # 调用提供的处理函数提交任务
                        image,duration,save_mode,text_full_path,image_frame_path = process_frame_func(frame_index,color_mode_value)
                        future = executor.submit(self.process_single_frame, is_multithread=True, frame_index=frame_index, image=image, save_mode=save_mode, text_frame_path=text_full_path, image_frame_path=image_frame_path, duration=duration)
                        futures.add(future)
                        frame_index += 1
                        # 提交任务后更新进度条信息
                        pbar.set_postfix_str(f"已提交任务 {frame_index}/{total_frames}, 已完成 {completed_count}/{total_frames}")
                    
                    # 检查已完成的任务
                    if futures:
                        done, not_done = concurrent.futures.wait(futures, timeout=0.1, 
                                                                return_when=concurrent.futures.FIRST_COMPLETED)
                        futures = not_done
                        
                        # 检查全局exit_flag
                        if hasattr(self, 'global_exit_flag') and self.global_exit_flag:
                            self.should_stop = True
                        
                        # 处理已完成的任务
                        for future in done:
                            try:
                                frame_idx, duration = future.result()
                                durations[frame_idx] = duration
                                completed_count += 1
                                # 更新进度条
                                pbar.update(1)
                                pbar.set_postfix_str(f"已提交任务 {frame_index}/{total_frames}, 已完成 {completed_count}/{total_frames}")
                            except KeyboardInterrupt:
                                # 捕获KeyboardInterrupt并设置should_stop
                                logger.info("在任务中检测到键盘中断")
                                self.should_stop = True
                                break
                            except Exception as e:
                                logger.error(ERROR_MESSAGES['frame_processing_failed'].format(e))
                                self.should_stop = True
                                break
                        
                        # 如果需要停止，取消所有未完成的任务
                        if self.should_stop:
                            for future in futures:
                                future.cancel()
                            logger.info("处理已中断，正在清理资源...")
                            break
                
                finished_time = time.time()
                pbar.set_postfix_str(f"用时：{format_time(finished_time - used_time)}")
        
        return durations
    
    def process_frames_sequentially(self, total_frames: int, pbar_description: str, process_frame_func: Callable) -> List[int]:
        """
        顺序处理帧
        
        Args:
            total_frames: int 总帧数
            pbar_description: str 进度条描述
            process_frame_func: Callable 处理单帧的函数

        Returns:
            List[int]: 持续时间列表
        """
        # 检查是否需要立即停止（包括全局exit_flag）
        if self.should_stop or hasattr(self, 'global_exit_flag') and self.global_exit_flag:
            logger.info("检测到中断标志，停止处理")
            return []
            
        durations = [0] * total_frames
        used_time = time.time()
        
        with show_project_status_progress(total=total_frames, description=pbar_description, unit='帧') as pbar:
            for frame_index in range(total_frames):
                pbar.set_postfix_str(f"处理帧 {frame_index}")
                
                # 检查是否需要停止处理（包括全局exit_flag）
                if self.should_stop or (hasattr(self, 'global_exit_flag') and self.global_exit_flag):
                    self.should_stop = True
                    logger.info("检测到中断标志，停止处理")
                    break
                
                # 处理当前帧
                color_mode_value = self.color_mode.value if isinstance(self.color_mode,
                                                                       Enum) else self.color_mode
                # 调用提供的处理函数提交任务
                image, duration, save_mode, text_full_path, image_frame_path = process_frame_func(frame_index,color_mode_value)
                frame_idx, duration = self.process_single_frame(is_multithread=False, frame_index=frame_index, image=image, save_mode=save_mode, text_frame_path=text_full_path, image_frame_path=image_frame_path, duration=duration)
                durations[frame_idx] = duration
                pbar.update(1)
            
            finished_time = time.time()
            pbar.set_postfix_str(f"用时：{format_time(finished_time - used_time)}")
        
        return durations
    
    def load_frames_from_temp(self, total_frames: int, frames_paths: Optional[Dict[int, Path]] = None) -> List[Image.Image]:
        """
        从临时文件夹加载帧图片
        
        Args:
            total_frames: int 总帧数
            frames_paths: Optional[Dict[int, Path]] 帧路径字典，键为帧索引，值为帧路径。值为None时，从self.temp_frame_paths中加载。

        Returns:
            List[Image.Image]: 帧图片列表
        """
        frames = []
        total_size = 0
        processed_size = 0
        start_time = time.time()
        load_paths = frames_paths if frames_paths else self.temp_frame_paths
        logger.debug(f"从路径：{load_paths}加载{total_frames}帧图片")

        # 检查temp_frame_paths中的帧数是否与预期一致
        if len(load_paths) != total_frames:
            raise FileNotFoundError(ERROR_MESSAGES['read_frame_from_temp_dir_failed'].format(
                f"{'frames_paths' if frames_paths else 'temp_frame_paths'}中记录的帧数量与原数量不一致"))
                
        # 计算总文件大小
        for frame_index in sorted(load_paths.keys()):
            file_path = load_paths[frame_index]
            if file_path.exists() and file_path.is_file():
                total_size += file_path.stat().st_size
        
        # 检查是否需要停止处理（包括全局exit_flag）
        if self.should_stop or (hasattr(self, 'global_exit_flag') and self.global_exit_flag):
            self.should_stop = True
            logger.info("检测到中断标志，停止处理")
            return []
            
        with show_project_status_progress(total=total_frames, description='从临时文件夹加载帧图片', unit='帧') as pbar:
            # 按照帧编号顺序读取文件
            for frame_index in sorted(load_paths.keys()):
                # 检查是否需要停止处理（包括全局exit_flag）
                if self.should_stop or (hasattr(self, 'global_exit_flag') and self.global_exit_flag):
                    self.should_stop = True
                    logger.info("检测到中断标志，停止处理")
                    break
                    
                frame_path = load_paths[frame_index]
                if frame_path.exists() and frame_path.is_file():
                    file_size = frame_path.stat().st_size
                    frame = Image.open(frame_path)
                    frames.append(frame.copy())
                    frame.close()
                    processed_size += file_size
                        
                    # 计算读取速度
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = processed_size / elapsed  # bytes/sec
                        pbar.set_postfix_str(f"速度: {format_speed(speed)}")
                    pbar.update(1)
        
        return frames
