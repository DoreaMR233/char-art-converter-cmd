"""
文件类型枚举模块

该模块定义了 FileType 枚举类，用于识别和处理不同类型的输入文件。

主要功能：
- 定义四种文件类型：图像(IMAGE)、视频(VIDEO)、文本(TEXT)和音频(AUDIO)
- 提供通过字符串获取对应枚举值的功能
- 实现通过文件路径和扩展名自动识别文件类型
- 提供文件类型验证和错误处理
- 确保字符画转换过程中文件类型的统一识别

依赖：
- enum: Python标准库枚举模块
- pathlib: 用于文件路径操作
- ..configs: 导入错误消息配置
- ..configs.common_config: 导入支持的格式列表
- ..configs.image_config: 导入支持的图像格式
- ..configs.video_config: 导入支持的视频格式
- ..configs.audio_config: 导入支持的音频格式
- ..utils.file_utils: 导入获取文件扩展名的工具函数
"""
from enum import Enum
from pathlib import Path

from ..configs import ERROR_MESSAGES



class FileType(Enum):
    """
    文件类型枚举类，定义了字符画生成过程中支持的文件类型
    
    Args:
        IMAGE: str 图像文件类型，值为'image'
        VIDEO: str 视频文件类型，值为'video'
        TEXT: str 文本文件类型，值为'text'
        AUDIO: str 音频文件类型，值为'audio'
    """
    IMAGE = 'image'  # 图像文件类型
    VIDEO = 'video'  # 视频文件类型
    TEXT = 'text'  # 文本文件类型
    AUDIO = 'audio' # 音频文件类型
    
    @staticmethod
    def from_string(value: str) -> 'FileType':
        """
        通过字符串获取对应的枚举值
        Args:
            value: str 字符串值
            
        Returns:
            FileType: 对应的FileType枚举值
            
        Raises:
            ValueError: 如果找不到对应的枚举值

        """
        for file_type in FileType:
            if file_type.value == value:
                return file_type

        raise ValueError(ERROR_MESSAGES['invalid_file_type'].format(f"{value}。可用的类型: {[file_type.value for file_type in FileType]}"))
    
    @staticmethod
    def from_path(input_path: Path) -> 'FileType':
        """
        通过文件路径获取对应的FileType枚举值
        根据文件的扩展名判断文件类型，先检查是否为支持的图像格式，再检查是否为支持的视频格式
        
        Args:
            input_path: Path Path 对象，表示文件路径
            
        Returns:
            FileType: 对应的FileType枚举值
            
        Raises:
            ValueError: 如果文件扩展名不在支持的格式列表中

        """
        from ..configs.common_config import ALL_SUPPORTED_FORMATS
        from ..configs.image_config import SUPPORTED_IMAGE_FORMATS
        from ..configs.video_config import SUPPORTED_VIDEO_FORMATS
        from ..configs.audio_config import SUPPORTED_AUDIO_FORMATS
        from ..utils.file_utils import get_file_extension
        # 使用get_file_extension获取文件后缀名
        file_ext = get_file_extension(input_path)
        
        # 检查是否为支持的图像格式
        if file_ext in SUPPORTED_IMAGE_FORMATS:
            return FileType.IMAGE
        
        # 检查是否为支持的视频格式
        if file_ext in SUPPORTED_VIDEO_FORMATS:
            return FileType.VIDEO
        # 检查是否为支持的文本格式
        if file_ext in ['.txt']:
            return FileType.TEXT

        # 检查是否为支持的音频格式
        if file_ext in SUPPORTED_AUDIO_FORMATS:
            return FileType.AUDIO

        # 如果扩展名不在支持的格式列表中，抛出异常
        raise ValueError(ERROR_MESSAGES['unsupported_format'].format(
            file_ext, ', '.join(ALL_SUPPORTED_FORMATS)
        ))
