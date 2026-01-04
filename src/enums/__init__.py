

"""
枚举类型模块集合

该模块包含字符画转换过程中使用的各种枚举类型定义，用于规范和统一
不同功能模块之间的参数传递和状态表示。

主要功能：
- ColorModes: 定义支持的颜色模式，如RGB、灰度、单色等
- SaveModes: 定义支持的输出保存模式，如仅文本、仅图像、组合保存等
- FileType: 定义支持的输入文件类型，如静态图像、动态图像、视频等

依赖：
- color_modes: 颜色模式枚举模块
- save_modes: 保存模式枚举模块
- file_type: 文件类型枚举模块
"""

# 导出颜色模式枚举
from .color_modes import ColorModes

# 导出保存模式枚举
from .save_modes import SaveModes

# 导出文件类型枚举
from .file_type import FileType

