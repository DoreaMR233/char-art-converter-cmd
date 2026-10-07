"""
视频配置模块

该模块定义了程序可以处理的各种视频格式及其相关参数配置，包含了多种视频格式的扩展名、支持的编解码器、默认配置和通用参数。

主要功能：
- 定义多种视频格式的扩展名（MP4、AVI、MOV、MKV、WebM、FLV、MPEG、WMV）
- 配置各种视频编解码器参数
- 设置通用视频参数（帧率、比特率等）
- 提供支持的视频格式列表
- 建立扩展名到默认编码的映射字典
- 建立扩展名到硬件解码友好编码参数的映射字典（像素格式、编码档次、偶数尺寸约束）
- 配置默认视频帧图片格式

依赖：
- typing：用于类型注解
- .image_config：导入PNG扩展名配置
"""

from typing import Dict, List, Optional

from src.configs.image_config import PNG_EXTENSIONS

# 视频格式配置
# 本配置文件定义了可以读取和生成的各种视频格式及其相关参数
# 每个格式都包含扩展名、支持的编解码器和默认配置
#
# 注意：以下 *_CODEC_* 常量的值必须是 ffmpeg 能够识别的**编码器名称**
# （即 `ffmpeg -encoders` 中列出的名字），而不是容器中的编码 tag
# （如 avc1/hevc/mp4v/VP90/FLV1/WMV3/apcn）。这些常量会被直接作为
# `-c:v` 的参数使用，写错会导致 ffmpeg 编码失败。

# 视频编解码器设置

# MP4格式 - 最常用的通用视频格式
# 特性：广泛兼容、高压缩率、良好的质量、支持流媒体
# 适用场景：网络视频、移动设备、社交媒体、通用存储
MP4_EXTENSIONS: List[str] = ['.mp4']
MP4_CODEC_H264: str = 'libx264'  # H.264编码 - 广泛兼容的高效编码，支持硬件解码
MP4_CODEC_H265: str = 'libx265'  # H.265/HEVC编码 - 更高压缩率，新一代编码
MP4_CODEC_MPEG4: str = 'mpeg4'  # MPEG-4 Part 2编码 - 兼容性更好但压缩率较低
MP4_DEFAULT_CODEC: str = MP4_CODEC_H264  # 默认使用H.264编码，保证主流硬件解码器可解
MP4_QUALITY: int = 8  # 质量设置（0-9，越高质量越好）

# AVI格式 - 经典视频格式
# 特性：Windows原生支持、容器格式灵活、无流式传输支持
# 适用场景：本地播放、Windows环境、传统视频编辑
AVI_EXTENSIONS: List[str] = ['.avi']
AVI_CODEC_XVID: str = 'libxvid'  # XVID编码 - 开源MPEG-4实现，良好的画质体积比
AVI_CODEC_MJPG: str = 'mjpeg'  # MJPEG编码 - 简单但文件较大，适合快速处理
AVI_CODEC_DIVX: str = 'msmpeg4'  # DivX编码 - 商业MPEG-4实现，广泛支持
AVI_DEFAULT_CODEC: str = MP4_CODEC_H264  # 默认使用H.264编码，MPEG-4 Part 2/MJPEG均无法通用硬解

# MOV格式 - Apple QuickTime格式
# 特性：高质量、专业编辑友好、Apple设备原生支持
# 适用场景：专业视频编辑、Apple生态系统、高质量存储
MOV_EXTENSIONS: List[str] = ['.mov']
MOV_CODEC_PRORES: str = 'prores'  # Apple ProRes编码 - 高质量专业编码，文件较大
MOV_CODEC_H264: str = 'libx264'  # H.264编码 - 平衡质量和文件大小
MOV_DEFAULT_CODEC: str = MOV_CODEC_H264  # 默认使用H.264编码

# MKV格式 - 开源容器格式
# 特性：开源、支持几乎所有编码、多音轨和字幕、章节标记
# 适用场景：高清视频存储、多语言内容、动漫、蓝光备份
MKV_EXTENSIONS: List[str] = ['.mkv']
MKV_CODEC_H264: str = 'libx264'  # H.264编码 - 良好的兼容性
MKV_CODEC_H265: str = 'libx265'  # H.265/HEVC编码 - 更高压缩率和质量
MKV_DEFAULT_CODEC: str = MKV_CODEC_H264  # 默认使用H.264编码

# WebM格式 - 开源Web视频格式
# 特性：开源、为Web优化、良好的压缩率、HTML5支持
# 适用场景：网页视频、在线流媒体、低带宽环境、YouTube
WEBM_EXTENSIONS: List[str] = ['.webm']
WEBM_CODEC_VP8: str = 'libvpx'  # VP8编码 - 早期开源编码
WEBM_CODEC_VP9: str = 'libvpx-vp9'  # VP9编码 - 新一代开源编码，压缩率更高
WEBM_DEFAULT_CODEC: str = WEBM_CODEC_VP9  # 默认使用VP9编码

# FLV格式 - Flash视频格式
# 特性：文件小、流式传输支持、早期Web视频标准
# 适用场景：早期网络视频、流媒体服务器、直播流
FLV_EXTENSIONS: List[str] = ['.flv']
FLV_CODEC_FLV1: str = 'flv'  # Sorenson Spark编码 - 传统Flash编码，无硬件解码器支持
FLV_CODEC_H264: str = 'libx264'  # H.264编码 - 现代Flash支持的高效编码
FLV_DEFAULT_CODEC: str = FLV_CODEC_H264  # 默认使用H.264编码

# MPEG格式 - 传统MPEG格式
# 特性：广泛的兼容性、标准清晰度、DVD使用的格式
# 适用场景：老式设备兼容、DVD视频、标准清晰度内容
MPEG_EXTENSIONS: List[str] = ['.mpg', '.mpeg']
MPEG_CODEC_MPEG1: str = 'mpeg1video'  # MPEG-1编码 - VCD质量
MPEG_CODEC_MPEG2: str = 'mpeg2video'  # MPEG-2编码 - DVD质量，支持硬件解码
MPEG_DEFAULT_CODEC: str = MPEG_CODEC_MPEG2  # 默认使用MPEG-2编码

# WMV格式 - Windows Media Video
# 特性：Windows原生支持、良好的压缩率、DRM支持
# 适用场景：Windows应用、流媒体、数字版权保护内容
# 注意：ffmpeg 没有 VC-1（WMV3）编码器，WMV1/WMV2 也不被硬件解码器支持，
# 因此 wmv 容器只能保证可播放，无法保证硬件解码。
WMV_EXTENSIONS: List[str] = ['.wmv']
WMV_CODEC_WMV1: str = 'wmv1'  # Windows Media Video 7 - 早期版本
WMV_CODEC_WMV2: str = 'wmv2'  # Windows Media Video 8 - 中等质量
WMV_CODEC_WMV3: str = 'wmv3'  # Windows Media Video 9/VC-1 - 仅可作为解码器名，ffmpeg 无对应编码器
WMV_DEFAULT_CODEC: str = WMV_CODEC_WMV2  # 默认使用WMV8编码

# 通用视频配置参数
# 这些参数用于视频创建时的默认设置
DEFAULT_FPS: float = 30.0  # 默认帧率 - 大多数标准视频使用30fps
DEFAULT_BITRATE: int = 5000000  # 默认比特率（5Mbps）- 高清视频推荐
DEFAULT_VIDEO_CODEC: str = MP4_DEFAULT_CODEC  # 默认视频编解码器 - 当扩展名没有对应编码时使用

# 硬件解码（硬解）友好配置
# 硬件解码器（NVDEC / DXVA2 / D3D11VA / QuickSync / VideoToolbox / VAAPI）只支持
# 主流编码格式的 8bit 4:2:0 色度采样码流，并且要求视频宽高为偶数。
# 字符画视频由 PNG（RGB）逐帧合成，如果不显式约束，x264 会默认输出
# yuv444p（High 4:4:4 Predictive），任何硬件解码器都无法解码。
# 因此合成视频时统一按以下配置输出，保证生成结果可以被硬件解码。
HW_DECODE_PIX_FMT: str = 'yuv420p'  # 硬件解码器通用支持的像素格式：8bit 4:2:0
HW_DECODE_EVEN_SIZE_FILTER: str = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'  # 宽高取偶数，否则 yuv420p 无法编码
HW_DECODE_DEFAULT_PROFILE: str = 'high'  # 未知扩展名时使用的编码档次
HW_DECODE_FASTSTART_EXTENSIONS: List[str] = MP4_EXTENSIONS + MOV_EXTENSIONS  # 需要 moov 前置以便流式播放

# VP9 编码速度参数：libvpx-vp9 默认速度极慢（cpu-used=0），这里放宽到实时可用的速度
HW_DECODE_VP9_SPEED_ARGS: List[str] = ['-deadline', 'good', '-cpu-used', '2', '-row-mt', '1']

# 没有可用硬件解码器的输出容器扩展名
# wmv 容器下的 WMV1/WMV2 编码格式目前没有硬件解码器支持（ffmpeg 也没有 VC-1 编码器），
# 只能保证视频可以正常播放，无法保证硬件解码。
HW_DECODE_UNSUPPORTED_EXTENSIONS: List[str] = list(WMV_EXTENSIONS)

# 扩展名到视频编码档次的映射字典
# 用于在硬件解码友好的前提下，为不同容器选择合适的 `-profile:v` 参数
# 值为 None 表示该编码器不支持/不需要指定编码档次
HW_DECODE_PROFILE_BY_EXTENSION: Dict[str, Optional[str]] = {}
for ext in MP4_EXTENSIONS + MKV_EXTENSIONS + AVI_EXTENSIONS + MOV_EXTENSIONS + FLV_EXTENSIONS:
    HW_DECODE_PROFILE_BY_EXTENSION[ext] = 'high'  # H.264 High Profile
for ext in WEBM_EXTENSIONS:
    HW_DECODE_PROFILE_BY_EXTENSION[ext] = '0'  # VP9 Profile 0：8bit 4:2:0
for ext in MPEG_EXTENSIONS:
    HW_DECODE_PROFILE_BY_EXTENSION[ext] = 'main'  # MPEG-2 Main Profile
for ext in WMV_EXTENSIONS:
    HW_DECODE_PROFILE_BY_EXTENSION[ext] = None

# 支持的视频格式配置
# 定义程序能够处理的视频文件格式列表
# 注意：实际支持的格式可能取决于系统上安装的编解码器
SUPPORTED_VIDEO_FORMATS: List[str] = MP4_EXTENSIONS + AVI_EXTENSIONS + MOV_EXTENSIONS + MKV_EXTENSIONS + WEBM_EXTENSIONS + FLV_EXTENSIONS + MPEG_EXTENSIONS + WMV_EXTENSIONS

# 扩展名到默认编码的映射字典
# 用于根据文件扩展名自动选择合适的视频编码器
# 此映射确保不同格式的视频使用最佳的默认编码器
# 默认编码器均已选择主流硬件解码器支持的编码格式（H.264 / VP9 / MPEG-2）
EXTENSION_TO_CODEC: dict = {}
# 为每种格式的所有扩展名添加对应的默认编码器
for ext in MP4_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = MP4_DEFAULT_CODEC
for ext in AVI_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = AVI_DEFAULT_CODEC
for ext in MOV_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = MOV_DEFAULT_CODEC
for ext in MKV_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = MKV_DEFAULT_CODEC
for ext in WEBM_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = WEBM_DEFAULT_CODEC
for ext in FLV_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = FLV_DEFAULT_CODEC
for ext in MPEG_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = MPEG_DEFAULT_CODEC
for ext in WMV_EXTENSIONS:
    EXTENSION_TO_CODEC[ext] = WMV_DEFAULT_CODEC

# 默认视频帧图片格式
DEFAULT_FRAME_EXTENSIONS: str = PNG_EXTENSIONS[0]