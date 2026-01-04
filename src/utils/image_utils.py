
"""
图像处理工具模块

该模块提供了图像处理相关的工具函数，负责图像格式检测、
动画图像识别等功能，为字符画转换提供图像预处理支持。

主要功能：
- 图像文件类型检测
- 动画图像判断
- 支持多种图像格式（JPEG、PNG、GIF、WebP、BMP、TIFF等）
- 动画格式支持（根据ANIMATED_FORMATS配置）

依赖：
- PIL (Pillow)：用于图像处理
- puremagic：用于文件类型检测
- pathlib：用于路径处理
- logging：用于日志记录

"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Callable

from PIL import Image

from .file_utils import get_file_extension, save_file
from .. import SaveModes
from ..configs import ANIMATED_FORMATS, DEFAULT_GIF_DURATION, GIF_LOOP_COUNT, GIF_EXTENSIONS, WEBP_EXTENSIONS, \
    WEBP_QUALITY, WEBP_METHOD
from ..configs.message_config import ERROR_MESSAGES

logger = logging.getLogger(__name__)

def is_animated_image(file_path: Path) -> bool:
    """
    判断图像文件是否为动画图像
    
    首先检查文件格式是否支持动画，然后判断文件是否包含多帧
    
    Args:
        file_path: Path Path 对象，表示要检查的图像文件路径
        
    Returns:
        bool: 布尔值，表示文件是否为动画图像
    
    Raises:
        TypeError: 当输入参数类型不符合要求时抛出
        Exception: 当检查过程中发生意外错误时抛出
    """
    # 首先检查文件格式是否在支持动画的格式列表中
    file_extension = file_path.suffix.lower()
    if file_extension not in ANIMATED_FORMATS:
        logging.debug(f"文件格式{file_extension}不在支持动画的格式列表中，直接返回False")
        return False
    
    # 如果是支持动画的格式，再检查是否包含多帧
    try:
        with Image.open(file_path) as img:
            # 检查是否有多帧
            return getattr(img, 'is_animated', False) or getattr(img, 'n_frames', 1) > 1
    except Exception as e :
        logging.debug(f"判断是否有多帧失败{e}，返回默认值False")
        return False

def create_animated_image(frames_paths: List[Path], output_path: Path, temp_dir: Path,
                          durations: Optional[List[float]] = None,threads_num: int = 1,
                          should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    使用ffmpeg将图像帧合成为动画图像（GIF或WebP）
    
    Args:
        frames_paths: List[Path] 图像帧路径列表
        output_path: Path 输出动画图像路径
        temp_dir: Path 临时文件目录
        durations: Optional[List[float]] 每帧持续时间列表（毫秒），如果为None则使用默认帧延迟
        threads_num: int 线程数，默认1
        should_stop: Optional[Callable[[], bool]] 用于检查是否应停止处理的回调函数
    Returns:
        None
        
    Raises:
        FileNotFoundError: 当输入文件不存在时
        ValueError: 当输出格式不支持时
        RuntimeError: 当ffmpeg执行失败时
    """
    # 检查输出格式
    # 检查所有输入文件是否存在
    for input_file in frames_paths:
        if not os.path.exists(input_file):
            logger.error(ERROR_MESSAGES['file_not_found'].format(input_file))
            raise FileNotFoundError(ERROR_MESSAGES['file_not_found'].format(input_file))
    try:
        logging.info(f"将 {len(frames_paths)} 个图像帧合成为动画图像 {output_path}")
        # 创建临时文件列表
        filelist_path = temp_dir / f"{output_path.stem}_filelist.txt"
        # 如果未提供持续时间列表，使用默认值
        if durations is None:
            fps = 1000.0 / DEFAULT_GIF_DURATION  # 计算帧率
            durations = [DEFAULT_GIF_DURATION / 1000.0] * len(frames_paths)  # 转换为秒
        else:
            fps = 1000.0 /(sum(durations) / len(durations))  # 计算平均帧率
        logger.info(f"动图帧率: {fps:.2f}")
        with open(filelist_path, mode='w', encoding='utf-8') as f:
            for frame_file in frames_paths:
                # 添加文件路径和持续时间
                index = frames_paths.index(frame_file)
                duration = durations[index] / 1000.0
                f.write(f"file '{str(frame_file.absolute())}'\nduration {duration}\n")
            # 最后一帧需要再写一次以确保持续正确的时间
            if frames_paths:
                f.write(f"file '{str(frames_paths[-1].absolute())}'\n")

        file_ext = get_file_extension(output_path)
        # 构建ffmpeg命令参数
        cmd = ['ffmpeg', '-f', 'concat', '-safe', '0',
               '-i', str(filelist_path),
               '-threads', str(threads_num)
               ]
        if file_ext in GIF_EXTENSIONS:
            # GIF格式参数
            cmd.extend(['-c:v', 'gif',
                   '-vf', f'fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                   '-loop', str(GIF_LOOP_COUNT)
                   ])
        elif file_ext in WEBP_EXTENSIONS:
            # WebP格式参数
            cmd.extend(['-c:v', 'libwebp', '-loop', str(GIF_LOOP_COUNT),
                        '-quality', str(WEBP_QUALITY), '-method', str(WEBP_METHOD)])

        # 添加输出文件路径和覆盖选项
        cmd.extend(['-y', str(output_path)])

        # 打印完整的FFmpeg命令
        logger.debug(f"执行FFmpeg命令: {' '.join(cmd)}")

        # 构建save_file所需的content字典
        content = {
            'cmd': cmd,
            'log_path' : temp_dir / f"{output_path.stem}_log.txt",
        }

        save_file(
            content=content,
            file_path=output_path,
            description="动画图像合成",
            save_mode=SaveModes.MERGE_ANIMATE,
            should_stop=should_stop
        )
        
    except KeyboardInterrupt:
        # 处理用户中断
        logging.info("动画图像合成被用户中断")
        raise
    except Exception as e:
        logging.error(ERROR_MESSAGES['create_animated_image_error'].format(e))
        raise
