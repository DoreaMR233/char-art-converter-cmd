"""
视频处理器模块

该模块定义了 VideoProcessor 类，负责将输入视频转换为字符画视频。
它是字符画转换的核心组件之一，处理视频加载、帧提取、字符映射、
颜色处理、音频提取与重组、字符画视频生成等功能。

主要功能：
- 支持各种格式的视频文件处理
- 提供多线程帧处理能力，提高处理效率
- 支持GPU加速，提升转换性能
- 保留原始视频音频，生成有声字符画视频
- 支持同时保存字符画图像序列
- 支持自定义输出视频的帧率、质量等参数
- 自动处理视频的不同分辨率和格式

依赖：
- logging: 用于日志记录
- time: 用于时间计算
- enum: 用于枚举处理
- pathlib: 用于路径操作
- cv2: 用于视频处理
- PIL: 用于图像处理
- ffmpeg: 用于视频和音频处理
"""
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from typing import Tuple, Any

import cv2
import jsonpath  # type: ignore
from PIL import Image

from .based_processor import BasedProcessor
from ..configs import SUCCESS_MESSAGES, ERROR_MESSAGES
from ..configs.message_config import ASSERT_MESSAGES
from ..configs.video_config import DEFAULT_FPS, DEFAULT_FRAME_EXTENSIONS, DEFAULT_VIDEO_CODEC, DEFAULT_BITRATE
from ..enums.file_type import FileType
from ..enums.save_modes import SaveModes
from ..utils import (
    get_output_path, create_temp_dir, format_time, cleanup_files,
    calculate_resized_dimensions, extract_audio,
    show_project_status_progress, create_video
)
from ..utils import video_utils

logger = logging.getLogger(__name__)

class VideoProcessor(BasedProcessor):
    """
    视频处理器类，负责将输入视频转换为字符画视频
    
    Args:
        self.audio_path: Optional[Path] 音频路径
        self.video_path: Path 视频输出路径
        self.temp_image_dir: Path 临时图像目录
        self.temp_audio_dir: Path 临时音频目录
        self.video_info: str 视频信息
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
        初始化视频处理器
        设置视频转换所需的所有参数和资源，包括输入输出路径、视频处理参数、临时目录等。
        
        Args:
            args: Any 包含所有处理参数的对象，包括输入路径、输出路径、颜色模式、字符密度等
        
        Returns:
            None: 无返回值
        
        Raises:
            ValueError: 当获取输出路径无效时抛出
        """
        # 调用父类构造函数，初始化共用属性和参数
        super().__init__(args, FileType.VIDEO)

        # 仅保留视频特有属性
        # 输出路径 - 视频特有
        self.audio_path: Optional[Path] = None
        # 重新设置输出路径，添加视频特有路径
        output_tuple = get_output_path(False, self.input_path, self.color_mode, args.output, FileType.VIDEO)
        # 类型检查，确保是4元组
        if len(output_tuple) == 4:
            self.output_dir, self.text_path, self.image_path, self.video_path = output_tuple
        else:
            raise ValueError(ERROR_MESSAGES['get_output_path_video_invalid'])
        # 临时文件夹路径 - 视频特有
        self.temp_image_dir: Path = self.image_path if (self.with_image and self.image_path is not None) else create_temp_dir()
        # 临时帧路径字典 - 用于存储处理后的帧路径
        self.temp_frame_paths: Dict[int, Path] = {}
        # 临时音频文件路径 - 视频特有
        self.temp_audio_dir: Path = create_temp_dir()
        # 视频信息 - 视频特有
        self.video_info: str =  ""

    def start(self) -> None:
        """
        启动视频转换处理流程
        记录处理参数，调用视频处理方法，处理完成后记录处理时间和结果信息。
        
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
        # 动态构建输出类型描述
        output_types = "视频"
        if self.with_text:
            output_types += "、文本"
        if self.with_image:
            output_types += "、图片"
        logger.info(f"输出类型: {output_types}")
        logger.info(f"视频输出路径: {self.video_path}")
        if self.with_text:
            logger.info(f"文本输出路径: {self.text_path}")
        if self.with_image:
            logger.info(f"视频帧输出路径: {self.image_path}")
        else:
            logger.info(f"临时视频帧夹路径: {self.temp_image_dir}")
        if self.temp_audio_dir:
            logger.info(f"临时音视频文件夹路径: {self.temp_audio_dir}")

        logger.info(f"GPU加速: {'启用' if self.enable_gpu else '禁用'}")
        if self.enable_gpu and self.gpu_memory_limit:
            logger.info(f"GPU内存限制: {self.gpu_memory_limit}MB")

        self.process_video()

        end_time: float = time.time()
        processing_time: float = end_time - start_time
        logger.info(f"{SUCCESS_MESSAGES['processing_complete']}")
        logger.info(f"处理时间: {format_time(processing_time)}")

    def process_video(self) -> None:
        """
        处理视频并转换为字符画视频
        该方法处理视频文件，将每一帧转换为字符画，然后合成为新的视频字符画。
        包含视频加载、单帧并行处理、进度显示、结果保存和临时文件清理等完整流程。
        
        Returns:
            None: 无返回值
        
        Raises:
            KeyboardInterrupt: 当用户中断处理时抛出
            FileNotFoundError: 当读取临时帧文件失败时抛出
            Exception: 当处理视频过程中遇到其他错误时抛出
        """
        try:
            # 使用OpenCV打开视频文件
            cap = cv2.VideoCapture(str(self.input_path))
            
            # 获取视频信息
            estimated_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 0.0
            frame_count = 0
            # 先尝试获取视频总帧数进行估算
            estimated_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.should_stop:
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            with show_project_status_progress(estimated_frame_count+1, description="获取视频信息：", verbose=True, position=0,hide_counter=False) as pbar:
                pbar.set_description("统计视频帧数")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    pbar.update(1)
                # 使用video_utils获取视频编码器信息
                self.video_info = video_utils.get_video_info(self.input_path, True, True, False, True)
                pbar.set_description("获取视频编码器信息")
                # 更新进度条到完成
                update_num = estimated_frame_count+1 - frame_count
                pbar.update(update_num)
            video_info_json = json.loads(self.video_info)
            codec_name = jsonpath.jsonpath(video_info_json, "$.streams[?(@.codec_type == \"video\")].codec_name")
            codec_name = codec_name[0] if codec_name else DEFAULT_VIDEO_CODEC
            bit_rate = jsonpath.jsonpath(video_info_json, "$.streams[?(@.codec_type == \"video\")].bit_rate")
            bit_rate = int(bit_rate[0]) if bit_rate else DEFAULT_BITRATE
            ffprobe_frame_info = jsonpath.jsonpath(video_info_json,
                                      "$.frames[?(@.media_type == \"video\")]")
            ffprobe_frame_count = len(ffprobe_frame_info) if ffprobe_frame_info else 0
            if ffprobe_frame_count > 0 and ffprobe_frame_count != frame_count:
                logger.warning(ERROR_MESSAGES['ffprobe_frame_count_mismatch'].format(ffprobe_frame_count, frame_count))
                frame_count = ffprobe_frame_count
            else:
                frame_count = frame_count if frame_count > 0 else estimated_frame_count
            avg_frame_rate_str = jsonpath.jsonpath(video_info_json, "$.streams[?(@.codec_type == \"video\")].avg_frame_rate")
            if avg_frame_rate_str and avg_frame_rate_str != "":
                avg_frame_rate_nums = avg_frame_rate_str[0].split("/")
                if len(avg_frame_rate_nums) == 2 and int(avg_frame_rate_nums[0]) > 0 and int(avg_frame_rate_nums[1]) > 0:
                    fps = float(avg_frame_rate_nums[0]) / float(avg_frame_rate_nums[1])
                else:
                    video_start_time = jsonpath.jsonpath(video_info_json, "$.format.start_time")
                    video_start_time = float(video_start_time[0]) if video_start_time else 0
                    video_end_time = jsonpath.jsonpath(video_info_json, "$.format.duration")
                    video_end_time = float(video_end_time[0]) if video_end_time else 0
                    video_duration = video_end_time - video_start_time
                    if video_duration > 0:
                        fps = frame_count / video_duration
            else:
                fps = estimated_fps
            logger.info(f"视频编码器: {codec_name}")
            logger.info(f"视频码率: {bit_rate}bps")
            logger.info(f"视频帧率: {fps:.2f}")
            logger.info(f"视频总帧数: {frame_count}")

            # 重置视频读取位置到开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 提取视频音频
            self.audio_path = extract_audio(self.input_path, self.temp_audio_dir,self.video_info,should_stop=self.should_stop_callback)
            
            # 读取第一帧用于计算调整后的尺寸
            all_ret, first_frame = cap.read()
            if not all_ret:
                raise ValueError(ERROR_MESSAGES['video_read_first_error'])
            
            # 将OpenCV图像转换为PIL图像进行尺寸计算
            first_pil_image = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            calculate_resized_dimensions(self.limit_size, first_pil_image, None, True)
            
            # 重置视频读取位置到开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            input_stem = self.input_path.stem

            
            # 定义帧处理函数
            def process_frame_func(frame_index: int, color_mode_value: str) -> Tuple[Image.Image, int, SaveModes, Path, Path]:
                """
                处理视频帧的函数
                从视频中读取指定索引的帧，将其转换为PIL图像，并计算相关参数
                
                Args:
                    frame_index: int 帧索引，用于标识当前处理的帧
                    color_mode_value: str 颜色模式值，用于生成输出文件的名称
                
                Returns:
                    Tuple[Image.Image, int, SaveModes, Path, Path]: 一个包含以下元素的元组：
                    - Image.Image: 处理后的PIL图像对象
                    - int: 帧持续时间（毫秒），基于视频的帧率计算
                    - SaveModes: 保存模式，设置为VIDEO_TMP_FRAME
                    - Path: 文本文件路径，用于保存字符画文本
                    - Path: 图像帧路径，用于保存临时图像帧
                
                Raises:
                    AssertionError: 当fps为None、text_path为None或temp_image_dir为None时抛出
                """
                # 读取当前帧
                frame_ret, frame_frame = cap.read()
                if not frame_ret or frame_frame is None:
                    raise ValueError(ERROR_MESSAGES['video_read_frame_error'])
                else:
                    # 将OpenCV图像转换为PIL图像
                    pil_image = Image.fromarray(cv2.cvtColor(frame_frame, cv2.COLOR_BGR2RGB))
                
                # 计算帧持续时间（基于视频的fps）
                # 确保fps不为None
                assert fps is not None, ASSERT_MESSAGES['fps_not_none']
                duration = int(1000 / fps) if fps > 0 else int(DEFAULT_FPS)
                # 设置保存模式和临时帧路径
                save_mode = SaveModes.VIDEO_TMP_FRAME
                input_ext = DEFAULT_FRAME_EXTENSIONS
                # 确保text_path和temp_image_dir不为None
                assert self.text_path is not None, ASSERT_MESSAGES['text_path_not_none']
                assert self.temp_image_dir is not None, ASSERT_MESSAGES['temp_image_dir_not_none']
                text_full_path = self.text_path / f"{input_stem}_{color_mode_value}_char_art_frame{frame_index}.txt"
                image_frame_path = self.temp_image_dir / f"{input_stem}_{color_mode_value}_char_art_frame{frame_index}{input_ext}"
                # 确保返回的路径不为None
                assert text_full_path is not None, ASSERT_MESSAGES['text_full_path_not_none']
                assert image_frame_path is not None, ASSERT_MESSAGES['image_frame_path_not_none']
                return pil_image, duration, save_mode, text_full_path, image_frame_path

            # 根据是否启用多线程选择处理方式

            if self.enable_multithread:
                logger.info(f"使用多线程处理，最大线程数: {self.max_workers}")
                try:
                    self.process_frames_multithreaded(frame_count,"处理视频帧",process_frame_func)
                except StopIteration as e:
                    logger.info(e)
                    pass
            else:
                logger.info("使用顺序处理")
                try:
                    self.process_frames_sequentially(frame_count,"处理视频帧",process_frame_func)
                except StopIteration as e:
                    logger.info(e)
                    # 视频帧读取完毕
                    pass
            
            # 关闭视频文件
            cap.release()
            if self.should_stop:
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            logger.info("将字符画视频帧合称为视频文件")

            frame_paths = []  # 从字典值中按照顺序提取路径列表
            for temp_frame_index in sorted(self.temp_frame_paths.keys()):
                frame_paths.append(self.temp_frame_paths[temp_frame_index])
            with Image.open(self.temp_frame_paths[1]) as temp_image:
                logger.info(f"字符画图像尺寸: {temp_image.width}x{temp_image.height}")
                create_video(frame_paths, self.audio_path, self.video_path, self.temp_audio_dir, self.video_info, fps,
                         codec_name, bit_rate, self.max_workers,should_stop=self.should_stop_callback)


                
        except KeyboardInterrupt:
            # 键盘中断时清理输出目录
            if self.output_dir is not None and not self.is_debug:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
        except Exception as e:
            logger.error(ERROR_MESSAGES['video_processing_failed'].format(e))
            logger.debug(f"原始异常: {e} (类型: {type(e)})")
            # 异常时清理输出目录
            if self.output_dir is not None and not self.is_debug:
                cleanup_files(self.output_dir)
                logger.info(SUCCESS_MESSAGES['output_folder_cleaned'].format(self.output_dir))
            raise
        finally:
            # 清理临时文件
            if self.temp_image_dir is not None and not self.with_image and not self.is_debug:
                # 清理所有临时文件（包括音频和视频）
                cleanup_files(self.temp_image_dir)
                logger.info(f"已清理临时图片文件夹：{self.temp_image_dir}")
            cleanup_files(self.temp_audio_dir)
            logger.info(f"已清理临时音视频文件夹：{self.temp_audio_dir}")