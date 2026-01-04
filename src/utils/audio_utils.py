"""
音频工具模块

该模块提供了音频处理相关的工具函数，主要用于从视频文件中提取音频。
支持使用FFmpeg进行音频提取，并能根据音频编码类型自动选择合适的文件扩展名。

主要功能：
- 从视频文件中提取音频流
- 自动检测音频编码类型
- 支持多种音频格式
- 错误处理和日志记录

依赖：
- json: 用于解析JSON数据
- logging: 用于日志记录
- pathlib: 用于路径操作
- jsonpath: 用于从JSON中提取数据
- file_utils: 文件操作工具
- audio_config: 音频配置参数
- message_config: 错误和警告消息
- save_modes: 保存模式枚举
"""
import json
import logging
from pathlib import Path
from typing import Optional, Callable

import jsonpath  # type: ignore

from .file_utils import save_file
from ..configs.audio_config import CODEC_TO_EXTENSION, DEFAULT_AUDIO_EXTENSION, DEFAULT_BIT_RATE, DEFAULT_CHANNELS, \
    DEFAULT_SAMPLE_RATE, DEFAULT_AUDIO_CODEC
from ..configs.message_config import ERROR_MESSAGES, WARNING_MESSAGES
from ..enums.save_modes import SaveModes

logger = logging.getLogger(__name__)

def extract_audio(input_path: Path, temp_dir: Path, video_info: str,
              should_stop: Optional[Callable[[], bool]] = None) -> Optional[Path]:
    """
    从视频中提取音频
    
    Args:
        input_path Path: 输入视频文件路径
        temp_dir Path: 临时文件目录
        video_info str: 视频信息JSON字符串
        should_stop Optional[Callable[[], bool]]: 用于检查是否应停止处理的回调函数
    
    Returns:
        Optional[Path]: 临时音频文件路径，或者如果视频没有音频则返回None
    
    Raises:
        FileNotFoundError: 当视频文件不存在时抛出
        RuntimeError: 当FFmpeg不可用或音频提取失败时抛出
    """
    if not input_path.exists():
        logger.error(ERROR_MESSAGES['file_not_found'].format(input_path))
        raise FileNotFoundError(ERROR_MESSAGES['file_not_found'].format(input_path))

        
    try:
        logger.debug(f"检查视频是否包含音频流: {input_path}")
        video_info_json = json.loads(video_info)
        audio_info = jsonpath.jsonpath(video_info_json, "$.streams[?(@.codec_type == \"audio\")]")
        
        # 检查是否有音频流
        if not audio_info:
            logger.warning(f"{WARNING_MESSAGES['video_no_audio'].format(input_path)}")
            return None
        audio_info = audio_info[0]
        # 记录音频信息用于调试
        logger.debug(f"音频流信息: {audio_info}")

        # 根据编码器类型决定音频文件后缀名
        codec_name = audio_info.get('codec_name', DEFAULT_AUDIO_CODEC)
        # 确保 codec_name 是 str 类型
        if isinstance(codec_name, str):
            extension = CODEC_TO_EXTENSION.get(codec_name, DEFAULT_AUDIO_EXTENSION)
        else:
            extension = DEFAULT_AUDIO_EXTENSION
        
        # 创建临时音频文件路径
        audio_path = temp_dir / f"temp_audio{extension}"
        log_path = temp_dir / f"temp_log{extension}_log.txt"
        logger.debug(f"开始从视频中提取音频: {input_path}")

        # 构建ffmpeg命令参数
        cmd = [
            'ffmpeg', '-i', str(input_path), '-vn',
            '-acodec', codec_name,
            '-ab', str(audio_info.get('bit_rate',str(DEFAULT_BIT_RATE))),
            '-ar', str(audio_info.get('sample_rate', str(DEFAULT_SAMPLE_RATE))),
            '-ac', str(audio_info.get('channels', str(DEFAULT_CHANNELS))),
            '-y', str(audio_path),
            # '-progress', 'pipe:1'
        ]
        logger.debug(f"执行FFmpeg命令: {' '.join(cmd)}")
        input_content = {'cmd': cmd, 'log_path': log_path}
        save_file(input_content, audio_path, f"提取原视频的音频部分", SaveModes.AUDIO, should_stop=should_stop)

        return audio_path
    except Exception as e:
        logger.error(ERROR_MESSAGES['audio_extraction_error'].format(e))
        logger.debug(f"原始异常: {e} (类型: {type(e)})")
        raise RuntimeError(ERROR_MESSAGES['audio_extraction_error'].format(e))
