"""
保存模式枚举模块

该模块定义了 SaveModes 枚举类，用于指定字符画生成后的输出格式和保存方式。

主要功能：
- 定义九种保存模式，包括文本文件、动画文本、静态图像、动画图像、音频等
- 提供通过字符串获取对应枚举值的功能
- 实现枚举值验证和错误处理
- 确保字符画生成过程中保存模式的统一使用和类型安全

依赖：
- enum: Python标准库枚举模块
- ..configs: 导入错误消息配置
"""
from enum import Enum

from ..configs import ERROR_MESSAGES


class SaveModes(Enum):
    """
    保存模式枚举类，定义了字符画生成过程中支持的保存格式
    
    Args:
        TEXT: str 文本文件，值为'text'
        ANIMATED_TEXT: str 动画文本文件，值为'animated_text'
        STATIC_IMAGE: str 静态图像文件，值为'static_image'
        ANIMATED_IMAGE: str 动画图像文件，值为'animated_image'
        AUDIO: str 音频文件，值为'audio'
        ANIMATED_IMAGE_TMP_FRAME: str 动画图像临时帧文件，值为'animated_image_tmp_frame'
        VIDEO_TMP_FRAME: str 视频临时帧文件，值为'video_tmp_frame'
        VIDEO: str 视频文件，值为'video'
        MERGE_ANIMATE: str 合并动画图像文件，值为'merge_animate'
    """
    TEXT = 'text'  # 文本文件
    ANIMATED_TEXT = 'animated_text'  # 动画文本文件
    STATIC_IMAGE = 'static_image'  # 静态图像文件
    ANIMATED_IMAGE = 'animated_image'  # 动画图像文件
    AUDIO = 'audio'  # 音频文件
    ANIMATED_IMAGE_TMP_FRAME = 'animated_image_tmp_frame'  # 动画图像临时帧文件
    VIDEO_TMP_FRAME = 'video_tmp_frame' # 视频临时帧文件
    VIDEO = 'video' # 视频文件
    MERGE_ANIMATE = 'merge_animate'  # 合并动画图像文件

    @staticmethod
    def from_string(value: str) -> 'SaveModes':
        """
        通过字符串获取对应的枚举值
        Args:
            value: str 字符串值
            
        Returns:
            SaveModes: 对应的SaveModes枚举值
            
        Raises:
            ValueError: 如果找不到对应的枚举值

        """
        for save_mode in SaveModes:
            if save_mode.value == value:
                return save_mode

        raise ValueError(ERROR_MESSAGES['invalid_save_mode'].format(f"{value}。可用的类型: {[save_mode.value for save_mode in SaveModes]}"))