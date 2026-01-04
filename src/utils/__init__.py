"""
工具函数模块

该模块提供了字符画转换所需的各种工具函数，是字符画转换功能的基础，被其他模块广泛使用。

主要功能：
- 字符艺术生成：像素到字符的转换、字符图像创建
- 颜色处理：RGB转灰度、对比度增强、颜色获取
- 文件操作：路径处理、目录创建、文件保存
- 字体管理：字体加载、字符尺寸计算
- 格式化工具：时间、速度、大小格式化
- GPU加速：PyTorch初始化、内存限制设置
- 图像处理：图像类型检测、动画图像识别
- 日志设置：日志配置、获取日志信息
- 进度条显示：进度条配置和显示
- 参数验证：输入文件、输出路径、参数有效性验证
- 视频处理：视频类型检测、信息获取、创建视频
- FFmpeg工具：FFmpeg可用性检查
- 音频处理：音频提取

依赖：
- char_art_utils: 字符艺术生成工具
- color_utils: 颜色处理工具
- file_utils: 文件操作工具
- font_utils: 字体管理工具
- format_utils: 格式化工具
- gpu_utils: GPU加速工具
- image_utils: 图像处理工具
- logging_utils: 日志设置工具
- progress_bar_utils: 进度条显示工具
- validate_utils: 参数验证工具
- video_utils: 视频处理工具
- ffmpeg_utils: FFmpeg工具
- audio_utils: 音频处理工具
"""

# 从各个子模块导入函数
from .char_art_utils import (
    calculate_resized_dimensions,
    resize_image_for_chars,
    pixel_to_char,
    pixel_to_char_gpu,
    image_to_char_text,
    create_char_image,
    process_image_gpu,
    process_image_cpu
)
from .color_utils import (
    rgb_to_gray,
    enhance_color,
    get_contrast_color,
    ensure_rgb_mode
)
from .file_utils import (
    get_output_path,
    ensure_dir_exists,
    create_temp_dir,
    get_file_extension,
    cleanup_files,
    save_file
)
from .font_utils import (
    FontManager,
    load_font,
    calculate_char_size
)
from .format_utils import (
    format_time,
    format_speed,
    format_size
)
from .gpu_utils import (
    init_pytorch_and_gpu,
    setup_gpu_memory_limit
)
from .image_utils import (
    is_animated_image
)
from .logging_utils import (
    setup_logging,
    get_current_log_config
)
from .progress_bar_utils import (
    get_tqdm_kwargs,
    show_value_file_save_progress,
    no_value_file_save_progress,
    show_project_status_progress
)
from .validate_utils import (
    validate_input_file,
    validate_output_path,
    validate_arguments
)
from .video_utils import (
    get_video_info,
    create_video
)
from .ffmpeg_utils import (
    check_ffmpeg_available
)
from .audio_utils import (
    extract_audio
)
