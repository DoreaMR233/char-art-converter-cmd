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
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, List, Callable, Union

from PIL import Image
from better_ffmpeg_progress import FfmpegProcess, FfmpegProcessError  # type: ignore
from better_ffmpeg_progress.utils import (  # type: ignore
    check_shell_needed_for_command, get_media_duration, parse_ffmpeg_progress_line, validate_ffmpeg_command,
)
from better_ffmpeg_progress.terminate_process import terminate_ffmpeg_process  # type: ignore

from .progress_bar_utils import get_progress_sink

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
def save_static_char_art_image(image: Image.Image, file_path: Path, file_ext: str, frame_index: Optional[int] = None,
                               optimize: bool = True, compress_level: Optional[int] = None) -> None:
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
        optimize bool:
            是否启用编码器的尺寸优化，对PNG/APNG/JPEG会成倍增加编码耗时（解码结果完全相同）。
            内部临时帧可传False以节省时间。
        compress_level Optional[int]:
            zlib压缩级别，仅对PNG/APNG生效，None表示使用默认值（PNG_COMPRESSION/APNG_COMPRESSION）。
            内部临时帧可传1以进一步加快编码速度（文件更大，解码结果完全相同）。
    
    返回:
        None
    """
    try:
        if file_ext in JPEG_EXTENSIONS:
            image.save(file_path, quality=JPEG_QUALITY, optimize=optimize)
        elif file_ext in PNG_EXTENSIONS:
            image.save(file_path, compress_level=PNG_COMPRESSION if compress_level is None else compress_level,
                       optimize=optimize)
        elif file_ext in WEBP_EXTENSIONS:
            image.save(file_path, quality=WEBP_QUALITY, method=WEBP_METHOD)
        elif file_ext in TIFF_EXTENSIONS:
            image.save(file_path, compression=TIFF_COMPRESSION)
        elif file_ext in AVIF_EXTENSIONS:
            image.save(file_path, quality=AVIF_QUALITY, method=AVIF_METHOD)
        elif file_ext in APNG_EXTENSIONS:
            image.save(file_path, compress_level=APNG_COMPRESSION if compress_level is None else compress_level,
                       optimize=optimize)
        else:
            image.save(file_path, optimize=optimize)
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

def _emit_ffmpeg_progress(stream, duration_secs: Optional[float], total_ms: int, description: str,
                          sink: Optional[Callable]) -> None:
    """
    读取FFmpeg的-progress输出并转成进度汇事件（秒 → 毫秒）

    参数:
        stream:
            FFmpeg的stdout流（-progress pipe:1 输出）
        duration_secs Optional[float]:
            媒体总时长（秒），无法获取时为None
        total_ms int:
            上报给进度汇的总量（毫秒），为0表示不确定进度
        description str:
            阶段描述，用于进度汇的文案
        sink Optional[Callable]:
            进度汇回调
    """
    if stream is None or sink is None:
        return
    for line in stream:
        progress_secs = parse_ffmpeg_progress_line(line.strip(), duration_secs)
        if progress_secs is not None:
            sink(int(progress_secs * 1000), total_ms, description, '', 0)


def _run_ffmpeg_with_progress(cmd: List[str], log_path: Path, description: str,
                             should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    在当前进程运行FFmpeg，并把进度上报给进度汇（GUI模式）

    与FfmpegProcess使用完全相同的FFmpeg参数（-hide_banner/-loglevel/-progress pipe:1/-nostats），
    只把进度消费端从tqdm换成进度汇；进度汇回调必须在本进程触发，因此该路径不经子进程执行。

    参数:
        cmd List[str]:
            执行的命令字符串列表
        log_path Path:
            日志文件路径
        description str:
            进度汇阶段描述
        should_stop Optional[Callable[[], bool]]:
            中断检查函数，返回True时终止FFmpeg并抛出KeyboardInterrupt

    异常:
        FfmpegProcessError: FFmpeg进程执行失败
        KeyboardInterrupt: 用户中断
    """
    sink = get_progress_sink()
    # 仅做校验（与FfmpegProcess一致：输入文件必须存在），输入路径取第一个 -i，行为与CLI完全一致
    validate_ffmpeg_command(cmd)
    duration_secs = get_media_duration(cmd[cmd.index('-i') + 1])
    total_ms = int(duration_secs * 1000) if duration_secs else 0
    full_cmd: Union[List[str], str] = [cmd[0], '-hide_banner', '-loglevel', 'verbose', '-progress', 'pipe:1',
                                       '-nostats', *cmd[1:]]
    use_shell = check_shell_needed_for_command(full_cmd)
    if use_shell:
        full_cmd = ' '.join(full_cmd)

    with log_path.open('w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            full_cmd,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        reader = threading.Thread(target=_emit_ffmpeg_progress,
                                  args=(process.stdout, duration_secs, total_ms, description, sink),
                                  daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                if should_stop is not None and should_stop():
                    terminate_ffmpeg_process(process)
                    raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'].format('FFmpeg处理'))
                time.sleep(0.1)
        finally:
            reader.join(timeout=2.0)
        if process.returncode != 0:
            raise FfmpegProcessError(f"FFmpeg命令执行失败，返回码: {process.returncode}")

def save_file_by_ffmpeg(cmd: List[str], log_path: Path, save_mode: SaveModes,
                        description: Optional[str] = None,
                        should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    使用FFmpeg保存文件

    参数:
        cmd List[str]:
            执行的命令字符串列表
        log_path Path:
            日志文件路径
        save_mode SaveModes:
            保存模式，用于日志记录
        description Optional[str]:
            进度汇阶段描述（仅进度汇模式使用）
        should_stop Optional[Callable[[], bool]]:
            中断检查函数（仅进度汇模式使用）
    
    返回:
        None
    """
    try:
        if get_progress_sink() is not None:
            # GUI模式：进度汇只能在本进程回调，直接在本地同步执行FFmpeg并上报进度
            _run_ffmpeg_with_progress(cmd, log_path, description or '合成文件', should_stop)
        else:
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

