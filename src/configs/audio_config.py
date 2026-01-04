"""
音频配置模块

该模块定义了程序可以处理的各种音频格式及其相关参数配置，包含了多种音频格式的扩展名、支持的编解码器、默认配置和通用参数。

主要功能：
- 支持多种音频格式的配置（AAC、MP3、WAV、FLAC、OGG、WMA、AIFF等）
- 定义各格式的扩展名和支持的编解码器
- 配置通用音频参数（采样率、声道数、比特率等）
- 提供编码器到扩展名的映射字典
- 设置默认音频格式和编解码器

依赖：
- typing：用于类型注解
"""

from typing import List, Dict

# 音频格式配置
# 本配置文件定义了程序可以处理的各种音频格式及其相关参数
# 每个格式都包含扩展名、支持的编解码器和默认配置

# AAC格式 - 高级音频编码
# 特性：高压缩率、良好的音质、广泛支持的现代音频格式
# 适用场景：网络音频、移动设备、流媒体、数字电视
AAC_EXTENSIONS: List[str] = ['.aac']
AAC_CODEC: str = 'aac'  # AAC编码

# MP3格式 - 动态图像专家组音频层III
# 特性：最广泛使用的音频格式、良好的兼容性、中等压缩率
# 适用场景：通用音频存储、便携式播放器、网络音频
MP3_EXTENSIONS: List[str] = ['.mp3']
MP3_CODEC: str = 'mp3'  # MP3编码

# WAV格式 - 波形音频文件格式
# 特性：无压缩、高质量、大文件体积
# 适用场景：音频编辑、专业音频处理、原始录音存储
WAV_EXTENSIONS: List[str] = ['.wav']
WAV_CODEC_PCM: str = 'pcm_s16le'  # 16位PCM编码 - 最常见的无压缩格式
WAV_CODEC_PCM_24: str = 'pcm_s24le'  # 24位PCM编码 - 更高质量
WAV_CODEC_PCM_32: str = 'pcm_s32le'  # 32位PCM编码 - 最高质量
WAV_DEFAULT_CODEC: str = WAV_CODEC_PCM  # 默认使用16位PCM编码

# FLAC格式 - 免费无损音频编解码器
# 特性：无损压缩、高质量、开源、较小的文件体积
# 适用场景：无损音频存储、音乐收藏、高质量音频流
FLAC_EXTENSIONS: List[str] = ['.flac']
FLAC_CODEC: str = 'flac'  # FLAC编码

# OGG格式 - 自由开放的容器格式
# 特性：开源、支持多种编码、良好的压缩率
# 适用场景：开源项目、游戏音频、网络音频
OGG_EXTENSIONS: List[str] = ['.ogg']
OGG_CODEC_VORBIS: str = 'vorbis'  # Vorbis编码 - 最常用的OGG编码
OGG_CODEC_OPUS: str = 'opus'  # Opus编码 - 高效的现代编码
OGG_DEFAULT_CODEC: str = OGG_CODEC_VORBIS  # 默认使用Vorbis编码

# WMA格式 - Windows Media音频
# 特性：Windows原生支持、良好的压缩率、DRM支持
# 适用场景：Windows应用、流媒体、数字版权保护内容
WMA_EXTENSIONS: List[str] = ['.wma']
WMA_CODEC: str = 'wmav2'  # WMA编码

# AIFF格式 - 音频交换文件格式
# 特性：Apple设备原生支持、高质量、专业音频处理
# 适用场景：Apple生态系统、专业音频编辑、高质量存储
AIFF_EXTENSIONS: List[str] = ['.aiff', '.aif']
AIFF_CODEC_PCM: str = 'pcm_s16be'  # 16位大端PCM编码

# 通用音频配置参数
# 这些参数用于音频创建时的默认设置
DEFAULT_SAMPLE_RATE: int = 44100  # 默认采样率（44.1kHz）- CD质量
DEFAULT_CHANNELS: int = 2  # 默认声道数 - 立体声
DEFAULT_BIT_RATE: int = 192000  # 默认比特率（192kbps）- 高质量音频
DEFAULT_BIT_RATE_STRING: str = "192k"  # 默认比特率字符串形式（用于FFmpeg命令）

# 支持的音频格式配置
# 定义程序能够处理的音频文件格式列表
# 注意：实际支持的格式可能取决于系统上安装的编解码器
SUPPORTED_AUDIO_FORMATS: List[str] = AAC_EXTENSIONS + MP3_EXTENSIONS + WAV_EXTENSIONS + FLAC_EXTENSIONS + OGG_EXTENSIONS + WMA_EXTENSIONS + AIFF_EXTENSIONS

# 编码器到扩展名的映射字典
# 用于根据音频编码器自动选择合适的文件扩展名
# 此映射确保不同编码的音频使用最佳的默认扩展名
CODEC_TO_EXTENSION: Dict[str, str] = {
    # AAC系列
    'aac': AAC_EXTENSIONS[0],
    'mp4a': AAC_EXTENSIONS[0],  # MP4音频编码
    'aac_latm': AAC_EXTENSIONS[0],
    
    # MP3系列
    'mp3': MP3_EXTENSIONS[0],
    'mp3float': MP3_EXTENSIONS[0],
    'libmp3lame': MP3_EXTENSIONS[0],
    
    # WAV/PCM系列
    'pcm_s16le': WAV_EXTENSIONS[0],
    'pcm_s24le': WAV_EXTENSIONS[0],
    'pcm_s32le': WAV_EXTENSIONS[0],
    'pcm_s16be': AIFF_EXTENSIONS[0],  # 大端PCM使用AIFF格式
    'pcm_s24be': AIFF_EXTENSIONS[0],
    'pcm_s32be': AIFF_EXTENSIONS[0],
    'pcm_u8': WAV_EXTENSIONS[0],
    'pcm_f32le': WAV_EXTENSIONS[0],
    'pcm_f64le': WAV_EXTENSIONS[0],
    
    # FLAC系列
    'flac': FLAC_EXTENSIONS[0],
    'flac24': FLAC_EXTENSIONS[0],
    
    # OGG系列
    'vorbis': OGG_EXTENSIONS[0],
    'opus': OGG_EXTENSIONS[0],
    
    # WMA系列
    'wmav1': WMA_EXTENSIONS[0],
    'wmav2': WMA_EXTENSIONS[0],
    'wmav3': WMA_EXTENSIONS[0],
}

# 默认音频格式
DEFAULT_AUDIO_EXTENSION: str = AAC_EXTENSIONS[0]  # 默认使用AAC格式
DEFAULT_AUDIO_CODEC: str = AAC_CODEC  # 默认使用AAC编码