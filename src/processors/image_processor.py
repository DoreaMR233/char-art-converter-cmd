"""
图像处理器模块

该模块定义了 ImageProcessor 类，负责将输入图像（静态或动态）转换为字符画。
它是字符画转换的核心组件，处理图像加载、调整大小、字符映射、颜色处理、
动画帧处理等功能。

主要功能：
- 支持静态图像和动态图像（如GIF）的处理
- 提供多线程处理能力
- 支持GPU加速
- 可保存字符画为图像和文本格式
- 自动处理不同图像格式，包括HEIF/HEIC/AVIF
- 支持颜色模式和字符密度的自定义

依赖：
- logging: 用于日志记录
- time: 用于时间计算
- enum: 用于枚举处理
- pathlib: 用于路径操作
- PIL: 用于图像处理
- pillow_heif: 用于支持HEIF/HEIC/AVIF格式
"""
import logging
import time
import psutil
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Any

from PIL import Image
from pillow_heif import register_heif_opener  # type: ignore

from ..configs import SUCCESS_MESSAGES
from ..configs.message_config import ERROR_MESSAGES, ASSERT_MESSAGES, WARNING_MESSAGES
from ..enums import SaveModes
from ..enums.file_type import FileType
from ..utils import (
    is_animated_image, get_output_path, create_temp_dir, cleanup_files, format_time,
    resize_image_for_chars, create_char_image, calculate_resized_dimensions
)
from ..utils.file_utils import save_file, get_file_extension
from .based_processor import BasedProcessor
from ..utils.image_utils import create_animated_image

logger = logging.getLogger(__name__)

# 注册pillow-heif以支持HEIF/HEIC/AVIF格式
register_heif_opener()

class ImageProcessor(BasedProcessor):
    """
    图像处理器类，负责将输入图像转换为字符画
    
    Args:
        self.is_animated: bool 是否为动图
        self.temp_dir: Optional[Path] 临时文件夹路径
        # 继承自BasedProcessor的属性
        self.file_type: FileType 文件类型枚举
        self.global_exit_flag: Any 全局退出标志
        self.should_stop: bool 停止处理标志
        self.density: str 字符密度级别
        self.color_mode: ColorModes 颜色模式
        self.limit_size: Optional[List[int]] 尺寸限制
        self.enable_multithread: bool 是否启动多线程
        self.with_text: bool 是否保存文本
        self.max_workers: int 最大线程数
        self.is_debug : bool 是否为Debug模式
        self.position_list: List[int] 进度条位置列表
        self.used_positions: set 已使用的进度条位置集合
        self.position_lock: threading.Lock 线程锁
        self.char_set: str 字符集
        self.char_count: int 字符集个数
        self.font_size: int 字体大小
        self.font: Any 字体对象
        self.input_path: Path 输入路径
        self.with_image: bool 是否保存帧
        self.output_dir: Path 输出目录
        self.text_path: Optional[Path] 文本输出路径
        self.image_path: Optional[Path] 图像输出路径
        self.temp_frame_paths: Dict[int, Path] 临时帧路径字典
        self.gpu_memory_limit: Optional[float] GPU内存限制
        self.torch: Optional[Any] torch模块
        self.torch_cuda_available: bool CUDA是否可用
        self.device: Optional[Any] 设备对象
        self.enable_gpu: bool 是否启用GPU加速
    """
    
    def __init__(self, args: Any) -> None:
        """
        初始化图像处理器
        
        设置图像转换所需的所有参数和资源，包括输入输出路径、图像处理参数、多线程配置等。
        同时初始化线程安全所需的锁和状态变量，并设置信号处理器以支持优雅终止。
        
        Args:
            args: 包含所有处理参数的对象，包括输入路径、输出路径、颜色模式、字符密度等
        
        Returns:
            无返回值
        """
        # 调用父类初始化
        super().__init__(args, FileType.IMAGE)
        
        # 是否为动图
        self.is_animated: bool = is_animated_image(self.input_path)

        # 输出路径
        output_tuple = get_output_path(
            self.is_animated, self.input_path, self.color_mode, args.output, FileType.IMAGE
        )
        # 类型检查，确保是4元组
        if len(output_tuple) == 4:
            self.output_dir, self.text_path, self.frame_path, self.image_path = output_tuple
        else:
            raise ValueError(ERROR_MESSAGES['get_output_path_image_invalid'])
        
        # 临时文件夹路径
        self.temp_dir: Optional[Path] =  None
        # 如果是动图且需要保存帧，则创建临时文件夹
        if self.is_animated and self.with_image and self.frame_path:
            self.temp_dir = self.frame_path
        else:
            self.temp_dir = create_temp_dir()


    def start(self) -> None:
        """启动图像转换处理流程
        
        记录处理参数，根据图像类型（静态或动态）调用相应的处理方法。
        处理完成后记录处理时间和结果信息。
        
        Returns:
            None: 无返回值
        
        Raises:
            KeyboardInterrupt: 当处理被用户中断时
            Exception: 当处理过程中发生其他错误时
        """
        start_time: float = time.time()
        logger.info(f"输入文件: {self.input_path}")
        logger.info(f"字符密度: {self.density}")
        logger.info(f"颜色模式: {self.color_mode.value if isinstance(self.color_mode, Enum) else self.color_mode}")
        logger.info(f"图片类型: {'动图' if self.is_animated else '静图'}")
        logger.info(f"输出类型: {'图像和文本' if self.with_text else '图像'}")
        logger.info(f"图片输出路径: {self.image_path}")
        if self.with_text:
            logger.info(f"文本输出路径: {self.text_path}")
        if self.temp_dir:
            if self.is_animated and self.with_image:
                logger.info(f"动图帧文件夹路径: {self.temp_dir}")
            else:
                logger.info(f"临时文件夹路径: {self.temp_dir}")
        logger.info(f"GPU加速: {'启用' if self.enable_gpu else '禁用'}")
        if self.enable_gpu and self.gpu_memory_limit:
            logger.info(f"GPU内存限制: {self.gpu_memory_limit}MB")
        if self.is_animated:
            self.process_animated_image()
        else:
            self.process_static_image()

        end_time: float = time.time()
        processing_time: float = end_time - start_time
        logger.info(f"{SUCCESS_MESSAGES['processing_complete']}")
        logger.info(f"处理时间: {format_time(processing_time)}")

    def process_static_image(self) -> None:
        """处理静态图像转换为字符画
        
        加载静态图像，调整其大小，将其转换为字符画，并保存为图像文件和可选的文本文件。
        支持中断检查以允许用户在处理过程中停止，并在出错时清理输出文件。
        
        Returns:
            None: 无返回值
            
        Raises:
            KeyboardInterrupt: 当处理被用户中断时
            Exception: 当处理过程中发生其他错误时
        """
        try:
            # 检查是否需要停止处理
            if self.should_stop:
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])

                
            with Image.open(self.input_path) as image:
                # 调整图片大小
                resized_image = resize_image_for_chars(self.limit_size, image, None, True, self.font_size)
                # 生成字符画图片和字符画文本
                output_image, output_text = create_char_image(
                    resized_image, self.enable_gpu, self.char_set, self.char_count, 
                    self.color_mode, self.font, self.torch, self.device, None,
                    should_stop=self.should_stop_callback
                )
            # 保存字符画文本
            if self.with_text and self.text_path is not None:
                input_content_text = {'output_text': output_text}
                save_file(input_content_text, self.text_path, "保存字符画文本", SaveModes.TEXT,
                          should_stop=self.should_stop_callback)
            # 保存字符画图片
            input_content_image: dict[str, Any] = {'output_image': output_image}
            assert self.image_path is not None, ASSERT_MESSAGES['image_path_not_none']
            save_file(input_content_image, self.image_path, "保存字符画图片", SaveModes.STATIC_IMAGE,
                      should_stop=self.should_stop_callback)
        except KeyboardInterrupt:
            # 键盘中断时也清理输出目录
            if self.output_dir is not None:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
        except Exception as e:
            logger.error(ERROR_MESSAGES['image_processing_failed'].format(e))
            logger.debug(f"原始异常: {e} (类型: {type(e)})")
            # 异常时清理输出目录
            if self.output_dir is not None:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
    def process_animated_image(self) -> None:
        """处理动画图像并转换为字符画动图
        
        该方法处理GIF等动画图像，将每一帧转换为字符画，然后合成为新的动画字符画。
        包含动画加载、单帧并行处理、进度显示、结果保存和临时文件清理等完整流程。
        
        Returns:
            None: 无返回值
        
        Raises:
            KeyboardInterrupt: 当用户中断处理时抛出
            FileNotFoundError: 当读取临时帧文件失败时抛出
            Exception: 当处理动画图像过程中遇到其他错误时抛出
        """
        try:
            with Image.open(self.input_path) as image:
                # 获取动图帧数
                frame_count = getattr(image, 'n_frames', 1)
                logger.info(f"帧数: {frame_count}")
                calculate_resized_dimensions(self.limit_size, image, None, True)
                input_stem = self.input_path.stem



                # 定义帧处理函数
                def process_frame_func(frame_index: int,
                                      color_mode_value: str) -> Tuple[Image.Image, int, SaveModes, Path, Path]:
                    """
                    处理动画图像帧的函数
                    
                    Args:
                        frame_index: int 要处理的帧索引
                        color_mode_value: str 当前使用的颜色模式值
                    
                    Returns:
                        Tuple[Image.Image, int, SaveModes, Path, Path]:
                            - 处理后的图像对象
                            - 帧持续时间（毫秒）
                            - 保存模式
                            - 文本文件保存路径
                            - 图像帧保存路径
                    """
                    # 定位到当前帧
                    image.seek(frame_index)
                    
                    # 获取帧持续时间
                    duration = image.info.get('duration', 0)
                    # 设置保存模式和临时帧路径
                    save_mode = SaveModes.ANIMATED_IMAGE_TMP_FRAME
                    input_ext = get_file_extension(self.input_path)
                    
                    # 确保路径不为None
                    assert self.text_path is not None, ASSERT_MESSAGES['text_path_not_none']
                    assert self.temp_dir is not None, ASSERT_MESSAGES['temp_dir_not_none']
                    text_full_path = self.text_path / f"{input_stem}_{color_mode_value}_char_art_frame{frame_index}.txt"
                    image_frame_path = self.temp_dir / f"{input_stem}_{color_mode_value}_char_art_frame{frame_index}{input_ext}"
                    return image.copy(), duration, save_mode, text_full_path, image_frame_path

                # 根据是否启用多线程选择处理方式
                if self.enable_multithread:
                    logger.info(f"使用多线程处理，最大线程数: {self.max_workers}")
                    durations = super().process_frames_multithreaded(frame_count,"处理帧", process_frame_func)
                else:
                    logger.info("使用顺序处理")
                    durations = super().process_frames_sequentially(frame_count,"处理帧",process_frame_func)
                
                if self.should_stop:
                    raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])

                # 检测是否可以一次性将所有帧加载到内存
                def can_load_all_frames_to_memory() -> bool:
                    """检测是否有足够的可用内存一次性加载所有帧
                    
                    Returns:
                        bool: 如果有足够内存返回True，否则返回False
                    """
                    try:
                        # 获取系统可用内存（字节）
                        available_memory = psutil.virtual_memory().available
                        
                        # 估算每帧内存占用（简化估算：假设每帧占用约1MB）
                        estimated_frame_size_mb = self.temp_frame_paths[1].stat().st_size / (1024 * 1024)
                        estimated_total_memory_mb = frame_count * estimated_frame_size_mb
                        estimated_total_memory_bytes = estimated_total_memory_mb * 1024 * 1024
                        
                        # 预留200MB额外内存作为安全边际
                        safety_margin_bytes = 200 * 1024 * 1024
                        
                        # 如果可用内存大于估算总内存的2倍（增加安全系数）加上安全边际，则认为可以一次性加载
                        return available_memory > (estimated_total_memory_bytes * 2 + safety_margin_bytes)
                    except Exception as err:
                        logger.warning(WARNING_MESSAGES['memory_check_failed'].format(f"{err}"))
                        logger.info("默认使用FFMPEG生成动图")
                        return False
                
                if can_load_all_frames_to_memory():
                    logger.info("使用ImageIO生成动图")
                    # 从临时文件夹读取帧图片
                    frames = super().load_frames_from_temp(frame_count)
                    logger.info(f"字符画图像尺寸: {frames[0].width}x{frames[0].height}")
                    # 保存字符画图片
                    input_content = {'frames': frames,'durations':durations}
                    assert self.image_path is not None, ASSERT_MESSAGES['image_path_not_none']
                    save_file(input_content, self.image_path, "保存字符画动图", SaveModes.ANIMATED_IMAGE,
                              should_stop=self.should_stop_callback)
                else:
                    logger.info("使用FFMPEG生成动图")
                    with Image.open(self.temp_frame_paths[1]) as temp_image:
                        logger.info(f"字符画图像尺寸: {temp_image.width}x{temp_image.height}")
                    frame_paths = [] # 从字典值中按照顺序提取路径列表
                    for tmp_frame_index in sorted(self.temp_frame_paths.keys()):
                        frame_paths.append(self.temp_frame_paths[tmp_frame_index])
                    # 确保参数类型正确
                    assert self.image_path is not None, ASSERT_MESSAGES['image_path_not_none']
                    assert self.temp_dir is not None, ASSERT_MESSAGES['temp_dir_not_none']
                    # 将durations转换为float类型列表
                    float_durations = [float(d) for d in durations]
                    create_animated_image(frame_paths, self.image_path, self.temp_dir, float_durations, self.max_workers, should_stop=self.should_stop_callback)
        except KeyboardInterrupt:
            # 键盘中断时清理输出目录
            if self.output_dir is not None and not self.is_debug:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
        except Exception as e:
            logger.error(ERROR_MESSAGES['animated_image_processing_failed'].format(e))
            logger.debug(f"原始异常: {e} (类型: {type(e)})")
            # 异常时清理输出目录
            if self.output_dir is not None and not self.is_debug:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
        finally:
            # 清理临时文件
            if self.temp_dir is not None and not self.with_image and not self.is_debug:
                cleanup_files(self.temp_dir)
                logger.info(SUCCESS_MESSAGES['temp_folder_cleaned'].format(self.temp_dir))


