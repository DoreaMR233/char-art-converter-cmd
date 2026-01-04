"""
颜色模式枚举模块

该模块定义了 ColorModes 枚举类，用于指定字符画生成过程中支持的颜色处理模式。

主要功能：
- 定义三种颜色模式：灰度模式(GRAYSCALE)、彩色模式(COLOR)、彩色背景模式(COLOR_BACKGROUND)
- 提供通过字符串获取对应枚举值的功能
- 实现枚举值验证和错误处理
- 确保字符画生成过程中颜色模式的统一使用和类型安全

依赖：
- enum: Python标准库枚举模块
- ..configs: 导入错误消息配置
"""
from enum import Enum




class ColorModes(Enum):
    """
    颜色模式枚举类，定义了字符画生成过程中支持的颜色处理模式
    
    Args:
        GRAYSCALE: str 灰度模式，生成纯灰度字符画，值为'grayscale'
        COLOR: str 彩色模式，字符本身着色，值为'color'
        COLOR_BACKGROUND: str 彩色背景模式，字符背景着色，值为'colorBackground'
    """
    GRAYSCALE = 'grayscale'  # 灰度模式
    COLOR = 'color'  # 彩色模式
    COLOR_BACKGROUND = 'colorBackground'  # 彩色背景模式
    
    @staticmethod
    def from_string(value: str) -> 'ColorModes':
        """
        通过字符串获取对应的枚举值
        Args:
            value: str 字符串值
            
        Returns:
            ColorModes: 对应的ColorModes枚举值
            
        Raises:
            ValueError: 如果找不到对应的枚举值

        """
        for mode in ColorModes:
            if mode.value == value:
                return mode
        from ..configs import ERROR_MESSAGES
        raise ValueError(ERROR_MESSAGES['invalid_color_mode'].format(f"{value}。可用的模式: {[mode.value for mode in ColorModes]}"))



