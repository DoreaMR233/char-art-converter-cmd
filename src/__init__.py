"""
字符画转换器核心包

该包提供了字符画转换的核心功能，包括图像处理、视频处理、
字符艺术生成等功能。

Modules:
    configs: 配置文件，定义了默认参数和常量
    enums: 枚举类，定义了颜色模式、文件类型等枚举
    processors: 处理器模块，实现了图像和视频的字符画转换
    utils: 工具函数模块，提供了各种辅助功能

Exported Functions/Classes:
    DEFAULT_DENSITY: 默认字符密度
    DEFAULT_COLOR_MODE: 默认颜色模式
    DEFAULT_LIMIT_SIZE: 默认图像尺寸限制
    DEFAULT_WITH_TEXT: 默认文本输出标志
    DEFAULT_WITH_IMAGE: 默认图像输出标志
    ERROR_MESSAGES: 错误信息字典
    COLOR_MODES: 支持的颜色模式列表
    CHAR_DENSITY_CONFIG: 字符密度配置
    ENABLE_GPU: GPU加速启用标志
    ColorModes: 颜色模式枚举类
    FileType: 文件类型枚举类
    SaveModes: 保存模式枚举类
    ImageProcessor: 图像处理器类
    VideoProcessor: 视频处理器类
    setup_logging: 日志设置工具函数
"""
# 导出配置项
from .configs import (
    DEFAULT_DENSITY, DEFAULT_COLOR_MODE, DEFAULT_LIMIT_SIZE, DEFAULT_WITH_TEXT,
    DEFAULT_WITH_IMAGE, ERROR_MESSAGES, COLOR_MODES, CHAR_DENSITY_CONFIG, ENABLE_GPU
)

# 导出枚举类
from .enums import ColorModes, FileType, SaveModes

# 导出处理器
from .processors import ImageProcessor, VideoProcessor

# 导出工具函数
from .utils import setup_logging
