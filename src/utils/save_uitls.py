"""
保存工具模块

该模块提供了各种文件保存功能，负责将字符画以不同格式保存到文件系统。

主要功能：
- 字符画文本保存
- 静态字符画图像保存
- 动态字符画图像保存
- 使用FFmpeg保存多媒体文件

依赖：
- logging：日志记录库
- pathlib：路径处理库
- typing：类型注解支持
- PIL (Pillow)：图像处理库
- better_ffmpeg_progress：FFmpeg进度显示库

"""
import logging
from pathlib import Path
from typing import Optional, List

from PIL import Image
from better_ffmpeg_progress import FfmpegProcess, FfmpegProcessError  # type: ignore

from src import ERROR_MESSAGES, SaveModes
from src.configs import JPEG_EXTENSIONS, JPEG_QUALITY, PNG_EXTENSIONS, PNG_COMPRESSION, WEBP_EXTENSIONS, WEBP_QUALITY, \
    WEBP_METHOD, TIFF_EXTENSIONS, TIFF_COMPRESSION, AVIF_EXTENSIONS, AVIF_QUALITY, AVIF_METHOD, GIF_EXTENSIONS, \
    GIF_LOOP_COUNT, WARNING_MESSAGES, DEFAULT_GIF_DURATION
from src.configs.image_config import APNG_EXTENSIONS, APNG_COMPRESSION

logger = logging.getLogger(__name__)

def save_char_art_text(text: str, file_path: Path, frame_index: Optional[int] = None) -> None:
    """
    保存字符画文本

    参数:
        text str:
            要保存的文本内容
        file_path Path:
            输出文件路径
        frame_index Optional[int]:
            帧索引，用于日志记录
    
    返回:
        None
    """
    try:
        with file_path.open('w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        logger.error(ERROR_MESSAGES['save_failed'].format(f'第{frame_index}帧字符画文本' if frame_index is not None else '字符画文本', str(file_path), e))
        raise
def save_static_char_art_image(image: Image.Image, file_path: Path, file_ext: str, frame_index: Optional[int] = None) -> None:
    """
    保存静态字符画图像

    参数:
        image Image.Image:
            要保存的图像对象
        file_path Path:
            输出文件路径
        file_ext str:
            文件扩展名，用于指定保存格式
        frame_index Optional[int]:
            帧索引，用于日志记录
    
    返回:
        None
    """
    try:
        if file_ext in JPEG_EXTENSIONS:
            image.save(file_path, quality=JPEG_QUALITY, optimize=True)
        elif file_ext in PNG_EXTENSIONS:
            image.save(file_path, compress_level=PNG_COMPRESSION, optimize=True)
        elif file_ext in WEBP_EXTENSIONS:
            image.save(file_path, quality=WEBP_QUALITY, method=WEBP_METHOD)
        elif file_ext in TIFF_EXTENSIONS:
            image.save(file_path, compression=TIFF_COMPRESSION)
        elif file_ext in AVIF_EXTENSIONS:
            image.save(file_path, quality=AVIF_QUALITY, method=AVIF_METHOD)
        elif file_ext in APNG_EXTENSIONS:
            image.save(file_path, compress_level=APNG_COMPRESSION, optimize=True)
        else:
            image.save(file_path, optimize=True)
    except Exception as e:
        logger.error(ERROR_MESSAGES['save_failed'].format(f'第{frame_index}帧字符画图像' if frame_index is not None else '字符画图像', str(file_path), e))
        raise

def save_animated_char_art_image(frames: List[Image.Image], durations: List[float], file_path: Path, file_ext: str) -> None:
    """
    保存动态字符画图像

    参数:
        frames List[Image.Image]:
            要保存的图像帧列表
        durations List[float]:
            每个帧的持续时间列表，单位为毫秒
        file_path Path:
            输出文件路径
        file_ext str:
            文件扩展名，用于指定保存格式
    
    返回:
        None
    """
    try:
        if durations is None or not durations:
            durations = [DEFAULT_GIF_DURATION for _ in range(len(frames))]
        first_frame = frames[0]
        if file_ext in GIF_EXTENSIONS:
            first_frame.save(
                file_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=GIF_LOOP_COUNT,
                optimize=True
            )
        elif file_ext in WEBP_EXTENSIONS:
            first_frame.save(
                file_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=GIF_LOOP_COUNT,
                quality=WEBP_QUALITY,
                method=WEBP_METHOD
            )
        elif file_ext in APNG_EXTENSIONS:
            first_frame.save(
                file_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                compress_level=APNG_COMPRESSION,
                optimize=True
            )
        else:
            # 不支持的动画格式，保存第一帧
            logger.warning(WARNING_MESSAGES['animation_format_unsupported'].format(file_ext))
            first_frame.save(file_path, optimize=True)
    except Exception as e:
        logger.error(ERROR_MESSAGES['save_failed'].format('字符画动图', file_path, e))
        raise

def save_file_by_ffmpeg(cmd: List[str], log_path: Path, save_mode: SaveModes) -> None:
    """
    使用FFmpeg保存文件

    参数:
        cmd List[str]:
            执行的命令字符串列表
        log_path Path:
            日志文件路径
        save_mode SaveModes:
            保存模式，用于日志记录
    
    返回:
        None
    """
    try:
        process = FfmpegProcess(cmd, ffmpeg_log_file=log_path)
        process.use_tqdm = True
        process.run()
        if process.return_code != 0:
            raise FfmpegProcessError(f"FFmpeg命令执行失败，返回码: {process.return_code}")
    except Exception as e:
        if save_mode == SaveModes.AUDIO:
            logger.error(ERROR_MESSAGES['save_failed'].format('提取的音频', str(log_path), e))
        elif save_mode == SaveModes.MERGE_ANIMATE:
            logger.error(ERROR_MESSAGES['save_failed'].format('字符画动图', str(log_path), e))
        elif save_mode == SaveModes.VIDEO:
            logger.error(ERROR_MESSAGES['save_failed'].format('字符画视频', str(log_path), e))
        else:
            logger.error(ERROR_MESSAGES['save_failed'].format('文件', str(log_path), e))
        raise

