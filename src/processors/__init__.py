

"""
处理器模块集合

该模块包含字符画转换的核心处理组件，提供了图像和视频的处理能力。

主要功能：
- BasedProcessor: 所有处理器的基类，提供通用功能和方法
- ImageProcessor: 处理静态和动态图像的转换
- VideoProcessor: 处理视频文件的转换

依赖：
- based_processor: 处理器基类模块
- image_processor: 图像处理器模块
- video_processor: 视频处理器模块
"""

# 导出处理器基类
from .based_processor import BasedProcessor

# 导出图像处理器
from .image_processor import ImageProcessor

# 导出视频处理器
from .video_processor import VideoProcessor

