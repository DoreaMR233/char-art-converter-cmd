"""
视频处理工具模块

该模块提供了一系列用于视频处理的工具函数，包括视频格式检测、从帧创建视频以及为视频添加音频等功能。

主要功能：
- 视频格式检测：支持多种视频格式的自动检测，采用多层级检测机制确保准确性
- 帧序列转视频：将字符画帧列表转换为无声视频文件
- 视频音频合成：将音频文件添加到视频中，创建完整的有声视频
- 硬件解码友好编码：按输出容器自动选择硬件解码器可解码的编码器、像素格式和编码档次

依赖：
- cv2 (OpenCV)：用于视频帧处理和格式验证
- ffmpeg：用于视频格式检测和音视频合成
- python-ffmpeg：用于调用FFmpeg命令行工具
- puremagic：用于基于文件内容的视频格式检测
- PIL (Pillow)：用于图像帧处理
- pathlib：用于文件路径操作
- logging：用于日志记录

主要函数：
- detect_video_type：检测视频文件类型，返回标准扩展名
- create_video_from_frames：从PIL图像帧列表创建无声视频
- add_audio_to_video：将音频添加到视频文件中
- get_hw_decode_encode_config：按输出容器获取硬件解码友好的编码器与编码档次
- get_hw_decode_video_args：构建硬件解码友好的视频编码参数
"""
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Callable

import jsonpath  # type: ignore

from .file_utils import save_file
from ..configs import DEFAULT_FPS, DEFAULT_VIDEO_CODEC, DEFAULT_BITRATE, DEFAULT_AUDIO_CODEC, DEFAULT_BIT_RATE_STRING, \
    DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS, EXTENSION_TO_CODEC, HW_DECODE_PIX_FMT, HW_DECODE_EVEN_SIZE_FILTER, \
    HW_DECODE_DEFAULT_PROFILE, HW_DECODE_FASTSTART_EXTENSIONS, HW_DECODE_VP9_SPEED_ARGS, \
    HW_DECODE_UNSUPPORTED_EXTENSIONS, HW_DECODE_PROFILE_BY_EXTENSION
from ..configs.message_config import ERROR_MESSAGES, WARNING_MESSAGES
from ..enums.save_modes import SaveModes

# 配置日志
logger = logging.getLogger(__name__)

def get_video_info(file_path: Path | str, show_format: bool = True, show_streams: bool = True, show_packets: bool = False, show_frames: bool = False) -> str:
    """
    获取媒体文件的视频信息
    
    Args:
        file_path: Path | str 媒体文件路径
        show_format: bool 是否显示封装格式信息，默认True
        show_streams: bool 是否显示所有媒体流信息，默认True
        show_packets: bool 是否显示每个数据包信息，默认False
        show_frames: bool 是否显示单帧信息，默认False
    
    返回:
        字符串，包含所选媒体信息的JSON格式字符串
    
    Raises:
        subprocess.SubprocessError: 当ffprobe命令执行失败时
    """
    # 构建基础命令
    cmd = ['ffprobe',
           '-i', (file_path if isinstance(file_path, str) else str(file_path)),
           '-v', 'quiet',
           '-print_format', 'json']
    
    # 根据参数动态添加显示选项
    if show_format:
        cmd.append('-show_format')
    if show_streams:
        cmd.append('-show_streams')
    if show_packets:
        cmd.append('-show_packets')
    if show_frames:
        cmd.append('-show_frames')

    # 打印完整的FFmpeg命令
    logger.debug(f"执行FFmpeg命令: {' '.join(cmd)}")

    # 执行命令 - 添加encoding='utf-8'以解决Windows编码问题
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
    probe_result = json.loads(result.stdout)


    return json.dumps(probe_result, indent=2, ensure_ascii=False)


def get_hw_decode_encode_config(output_path: Path) -> tuple[str, Optional[str]]:
    """
    获取硬件解码友好的视频编码配置

    根据输出容器的扩展名，选择主流硬件解码器（NVDEC/DXVA2/D3D11VA/QuickSync/VAAPI 等）
    能够解码的视频编码器及其编码档次。扩展名没有对应编码时使用默认编码器。

    Args:
        output_path: Path 输出视频文件路径，其扩展名决定输出容器格式

    Returns:
        tuple[str, Optional[str]]: 视频编码器名称与编码档次，编码档次为 None 表示该编码器不指定档次

    Raises:
        None
    """
    suffix = output_path.suffix.lower()
    encoder = EXTENSION_TO_CODEC.get(suffix, DEFAULT_VIDEO_CODEC)
    profile = HW_DECODE_PROFILE_BY_EXTENSION.get(suffix, HW_DECODE_DEFAULT_PROFILE)
    return encoder, profile


def get_hw_decode_video_args(output_path: Path, encoder: Optional[str] = None) -> list[str]:
    """
    构建硬件解码友好的视频编码参数

    硬件解码器只支持主流编码格式的 8bit 4:2:0 色度采样码流，并且要求宽高为偶数，
    因此这里统一附带 `-pix_fmt yuv420p`、`-profile:v` 以及宽高取偶数的缩放滤镜。

    Args:
        output_path: Path 输出视频文件路径，其扩展名决定输出容器格式
        encoder: Optional[str] 指定的视频编码器，为 None 时按输出容器自动选择

    Returns:
        list[str]: ffmpeg 输出参数列表（-c:v、-pix_fmt、-profile:v、-vf 等）

    Raises:
        None
    """
    suffix = output_path.suffix.lower()
    default_encoder, default_profile = get_hw_decode_encode_config(output_path)
    if encoder is None:
        encoder, profile = default_encoder, default_profile
    else:
        # 显式指定编码器时，仅当它与该容器的默认编码器一致才套用编码档次，
        # 否则档次可能不被该编码器支持（如 mpeg4 不支持 high 档次而直接报错）
        profile = default_profile if encoder == default_encoder else None

    args = ['-c:v', encoder, '-pix_fmt', HW_DECODE_PIX_FMT]
    if profile is not None:
        args.extend(['-profile:v', str(profile)])
    # libvpx-vp9 默认编码速度极慢，放宽速度参数以便实际可用
    if encoder == 'libvpx-vp9':
        args.extend(HW_DECODE_VP9_SPEED_ARGS)
    # 宽高取偶数，否则 yuv420p 无法编码（如 321x241 会直接报错）
    args.extend(['-vf', HW_DECODE_EVEN_SIZE_FILTER])
    # mp4/mov 需要 moov 前置才能边下边播
    if suffix in HW_DECODE_FASTSTART_EXTENSIONS:
        args.extend(['-movflags', '+faststart'])
    return args


def create_video(frames_paths: list[Path], audio_path: Optional[Path], output_path: Path, temp_dir: Path, video_info: str, fps: float = DEFAULT_FPS,
                 codec: Optional[str] = None, bitrate: int = DEFAULT_BITRATE, threads_num: int = 1,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    从帧序列创建视频文件

    输出视频统一采用硬件解码友好的编码参数（编码器按输出容器选择、8bit 4:2:0 像素格式、
    主流编码档次、偶数宽高），以保证生成的视频可以被硬件解码器解码。

    Args:
        frames_paths: list[Path] 帧文件路径列表
        audio_path: Optional[Path] 音频文件路径，可选
        output_path: Path 输出视频文件路径
        temp_dir: Path 临时目录路径
        video_info: str 视频信息的JSON字符串
        fps: float 帧率，默认使用配置中的默认值
        codec: Optional[str] 视频编码器，为 None 时按输出容器自动选择硬件解码友好的编码器
        bitrate: int 视频比特率，默认使用配置中的默认值
        threads_num: int 使用的线程数，默认1
        should_stop: Optional[Callable[[], bool]] 检查是否应该停止处理的回调函数，可选

    返回:
        None

    Raises:
        FileNotFoundError: 当输入文件不存在时
        Exception: 当视频创建失败时
    """
    # 检查所有输入文件是否存在
    for input_file in frames_paths:
        if not os.path.exists(input_file):
            logger.error(ERROR_MESSAGES['file_not_found'].format(input_file))
            raise FileNotFoundError(ERROR_MESSAGES['file_not_found'].format(input_file))
    try:
        logger.debug(f"视频码率: {bitrate}")
        logger.info(f"将 {len(frames_paths)} 个视频帧合成视频文件 {output_path}")
        # 解析视频信息
        video_info_json = json.loads(video_info)
        frames_info = jsonpath.jsonpath(video_info_json,
                                      "$.frames[?(@.media_type == \"video\")]")
        if not frames_info:
            logger.debug(f"视频帧信息中未找到视频流的持续时间，尝试使用数据包信息")
            frames_info = jsonpath.jsonpath(video_info_json,
                                          "$.packets_and_frames[?(@.type== \"frame\" && @.media_type == \"video\")]")
        durations = []
        if frames_info:
            for frame in frames_info:
                durations.append(frame.get("duration_time", 1 / fps))

        else:
            logger.debug(f"视频帧信息中未找到视频流的持续时间，尝试使用默认帧率 {fps}")

        durations = durations if durations else [1 / fps for _ in range(len(frames_paths))]
        # 创建临时文件列表
        with open(temp_dir / f"{output_path.stem}_filelist.txt", mode='w', encoding='utf-8') as f:
            for frame_file in frames_paths:
                f.write(f"file '{str(frame_file.absolute())}'\nduration {durations[frames_paths.index(frame_file)]}\n")
            filelist_path = f.name
        # 硬件解码友好的编码参数：编码器按输出容器选择，并约束像素格式/编码档次/宽高
        hw_args = get_hw_decode_video_args(output_path, encoder=codec)
        output_suffix = output_path.suffix.lower()
        if output_suffix in HW_DECODE_UNSUPPORTED_EXTENSIONS:
            logger.warning(WARNING_MESSAGES['hw_decode_unsupported_container'].format(
                output_suffix, hw_args[hw_args.index('-c:v') + 1]))
        logger.info(f"视频编码参数: {' '.join(hw_args)}")
        # 构建ffmpeg命令参数
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-threads', str(threads_num),
            '-i', filelist_path,
        ]
        # 音频参数：必须先声明所有输入（-i），再统一追加输出参数，
        # 否则输出参数会被当作第二个输入文件的输入选项而报错
        audio_args = []
        if audio_path and  audio_path.exists():
            logger.debug(f"音频文件存在: {audio_path}")
            audio_info = jsonpath.jsonpath(video_info_json, "$.streams[?(@.codec_type == \"audio\")]")[0]
            acodec = audio_info.get("codec_name", DEFAULT_AUDIO_CODEC)
            audio_bit_rate = audio_info.get("bit_rate",DEFAULT_BIT_RATE_STRING)
            sample_rate = audio_info.get("sample_rate",DEFAULT_SAMPLE_RATE)
            channels = audio_info.get("channels", DEFAULT_CHANNELS)
            logger.debug(f"音频编码器: {acodec}, 音频比特率: {audio_bit_rate}, 采样率: {sample_rate}, 声道数: {channels}")
            cmd.extend([
                '-i', str(audio_path),  # 输入音频文件
            ])
            audio_args = ['-b:a', str(audio_bit_rate),
                          '-ar', str(sample_rate),
                          '-ac', str(channels),
                          '-c:a', acodec]
        else:
            logger.debug(f"音频文件不存在: {audio_path}")
        cmd.extend(hw_args)
        cmd.extend(audio_args)
        cmd.extend([
            '-r', str(fps),
            '-y',  # 覆盖现有文件
            str(output_path)])

        # 打印完整的FFmpeg命令
        logger.debug(f"执行FFmpeg命令: {' '.join(str(arg) for arg in cmd)}")

        # 使用save_file函数进行视频合并
        content = {
            'cmd': cmd,
            'log_path' : temp_dir / f"{output_path.stem}_log.txt"
        }
        save_file(content, output_path, f"合成视频文件", SaveModes.VIDEO, should_stop=should_stop)


    except Exception as e:
        logger.error(ERROR_MESSAGES['create_video_error'].format(e))
        raise