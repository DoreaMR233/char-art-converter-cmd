
"""
通用配置模块

该模块定义了应用程序中跨组件共享的通用配置参数，包含字符密度、颜色模式、默认设置、字体、日志和GPU加速等配置。

主要功能：
- 定义字符密度和字符集配置（低密度、中密度、高密度）
- 提供颜色模式支持列表
- 整合所有支持的图像和视频格式
- 设置默认参数（密度、颜色模式、尺寸限制、输出选项等）
- 配置日志格式和级别
- 管理GPU加速相关设置

依赖：
- typing：用于类型注解
- ..enums.color_modes：提供ColorModes枚举类
- .image_config：导入支持的图像格式
- .video_config：导入支持的视频格式
"""

from typing import Dict, List

from ..enums.color_modes import ColorModes
from .image_config import SUPPORTED_IMAGE_FORMATS
from .video_config import SUPPORTED_VIDEO_FORMATS

# 字符密度和字符集配置
CHAR_DENSITY_CONFIG: Dict[str, str] = {
    'low': " .:-=+*#%@",  # 低密度
    'medium': " ,:;i1tfLCG08@",  # 中密度
    'high': " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"  # 高密度
}

# 颜色模式
COLOR_MODES: List[str] = [mode.value for mode in ColorModes]

# 所有支持的格式列表（用于帮助信息显示）
ALL_SUPPORTED_FORMATS: List[str] = list(SUPPORTED_IMAGE_FORMATS)+list(SUPPORTED_VIDEO_FORMATS)

# 默认设置
# 计算中间位置的key作为默认密度
keys = list(CHAR_DENSITY_CONFIG.keys())
DEFAULT_DENSITY: str = keys[len(keys) // 2]  # 如果key数量是奇数，取中间的；如果是偶数，取中间两个的第一个
DEFAULT_COLOR_MODE: ColorModes = ColorModes.COLOR
DEFAULT_LIMIT_SIZE: None | List[int] = None
DEFAULT_WITH_TEXT: bool = False
DEFAULT_WITH_IMAGE: bool = False

# 字体设置
DEFAULT_FONT_SIZE: int = 12  

# 日志设置
DEFAULT_LOG_LEVEL: str = 'INFO'  # 默认日志等级 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# GPU并行计算配置
ENABLE_GPU: bool = True                    # 是否启用GPU加速（默认启用，如果GPU不可用会自动回退到CPU）
GPU_MEMORY_LIMIT: float = 0.8              # GPU内存使用限制：80%
