"""
FFmpeg工具模块

该模块提供了与FFmpeg相关的工具函数，用于检查FFmpeg可用性，为音视频处理提供支持。

主要功能：
- 检查系统是否安装了FFmpeg
- 为视频和音频处理提供FFmpeg支持验证

依赖：
- subprocess: 用于执行系统命令

"""
import logging
import subprocess

from src import ERROR_MESSAGES

logger = logging.getLogger(__name__)

def check_ffmpeg_available() -> None:
    """
    检查系统是否安装了FFmpeg
    
    通过执行'ffmpeg -version'命令来检查FFmpeg是否可用
    
    Returns:
        None: 如果FFmpeg可用则无返回值
        
    Raises:
        RuntimeError: 当FFmpeg不可用时抛出
    """
    try:
        # 使用subprocess检查ffmpeg是否可用
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error(ERROR_MESSAGES['ffmpeg_not_available'])
        raise RuntimeError(ERROR_MESSAGES['ffmpeg_not_available'])
