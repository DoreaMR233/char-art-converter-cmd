
"""
图像配置模块

该模块定义了程序可以处理的各种图像格式及其相关参数配置，包含了多种图像格式的扩展名、质量设置、压缩级别和支持的格式列表。

主要功能：
- 定义多种图像格式的扩展名（JPEG、PNG、BMP、GIF、WebP、TIFF、HEIF、AVIF）
- 配置格式特定参数（质量设置、压缩级别、压缩方法等）
- 设置动画格式参数（GIF循环次数、帧延迟）
- 提供支持的图像格式列表
- 定义支持动画的文件格式

依赖：
- typing：用于类型注解
"""

from typing import List

# PIL图像大小限制设置
# 设置PIL的最大图像像素限制，防止解压炸弹攻击警告
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 100000000  # 10000*10000像素，约100MB图像

# 图像处理设置

# JPEG格式 - 最常用有损压缩格式
JPEG_EXTENSIONS: List[str] = ['.jpg', '.jpeg']
JPEG_QUALITY: int = 95  # JPEG质量

# PNG格式 - 无损压缩，支持透明度
PNG_EXTENSIONS: List[str] = ['.png']
PNG_COMPRESSION: int = 6  # PNG压缩级别

# BMP格式 - Windows位图格式
BMP_EXTENSIONS: List[str] = ['.bmp']

# GIF设置
# GIF格式 - 支持动画和透明度
GIF_EXTENSIONS: List[str] = ['.gif']
GIF_LOOP_COUNT: int = 0  # 0表示无限循环
DEFAULT_GIF_DURATION: int = 100  # 默认帧延迟（毫秒）

# WebP设置
# WebP格式 - Google开发的现代图像格式
WEBP_EXTENSIONS: List[str] = ['.webp']
WEBP_QUALITY: int = 80
WEBP_METHOD: int = 6  # 压缩方法（0-6，6为最佳质量）

# APNG格式 - 支持动画的PNG格式
APNG_EXTENSIONS: List[str] = ['.apng']
APNG_COMPRESSION: int = 6  # APng压缩级别

# TIFF格式 - 高质量图像格式
TIFF_EXTENSIONS: List[str] = ['.tiff','.tif']
TIFF_COMPRESSION: str = 'tiff_lzw'  # LZW压缩，保留图像质量

# HEIF格式 - 高效图像格式，苹果设备常用
HEIF_EXTENSIONS: List[str] = ['.heif', '.heic']

# AVIF格式 - 基于AV1的现代图像格式
AVIF_EXTENSIONS: List[str] = ['.avif']
AVIF_QUALITY: int = 80  # AVIF质量
AVIF_METHOD: int = 6  # 压缩方法（0-6，6为最佳质量）


# 支持的文件格式配置
# 定义程序能够处理的图像文件格式
SUPPORTED_IMAGE_FORMATS: List[str] = JPEG_EXTENSIONS+PNG_EXTENSIONS+ BMP_EXTENSIONS+GIF_EXTENSIONS+WEBP_EXTENSIONS+TIFF_EXTENSIONS+HEIF_EXTENSIONS+AVIF_EXTENSIONS+APNG_EXTENSIONS


# 动画格式配置
# 定义支持动画的文件格式
# 目前支持GIF和WebP动画
ANIMATED_FORMATS: List[str] = GIF_EXTENSIONS+WEBP_EXTENSIONS+APNG_EXTENSIONS

