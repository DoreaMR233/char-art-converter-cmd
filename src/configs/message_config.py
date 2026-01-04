
"""
消息配置模块

该模块定义了程序中使用的各种消息模板，包括错误消息、成功消息、断言消息和警告消息。所有消息都支持中文，便于用户理解和调试。

主要功能：
- 定义错误消息模板（文件操作、格式不支持、处理失败、参数错误等）
- 提供成功消息模板（处理完成、文件保存、文件夹清理等）
- 设置断言消息模板（用于开发和调试阶段）
- 配置警告消息模板（GPU不可用、音频处理失败、字体加载问题等）
- 所有消息支持格式化参数，可动态插入相关信息

依赖：
- typing：用于类型注解
"""

from typing import Dict

# 错误消息模板
# 定义程序中使用的各种错误消息，支持格式化参数
ERROR_MESSAGES: Dict[str, str] = {
    'file_not_found': '文件不存在: {}',                    # 输入文件不存在时的错误消息
    'path_not_file': '路径不是文件: {}',                    # 路径不是文件时的错误消息
    'path_not_directory': '路径不是目录: {}',               # 路径不是目录时的错误消息
    'directory_creation_failed': '目录创建失败: {}',        # 目录创建失败时的错误消息
    'unsupported_format': '不支持的文件格式: {}',            # 文件格式不支持时的错误消息
    'output_dir_not_writable': '输出目录不可写: {}',         # 输出目录无写入权限时的错误消息
    'processing_failed': '图像处理失败: {}',                   # 图像处理失败时的错误消息
    'invalid_limit_size': '无效的大小限制参数: {}',           # 大小限制参数无效时的错误消息
    'read_frame_from_temp_dir_failed': '从临时文件夹读取动图帧失败：{}',  # 从临时文件夹读取动图帧失败时的错误消息
    'invalid_color_mode': '无效的颜色模式: {}',                # 色彩模式参数无效的错误信息
    'invalid_save_mode': '无效的保存模式: {}',                # 无效的保存模式的错误消息
    'invalid_file_type': '无效的文件类型: {}',                # 文件类型参数无效的错误信息
    'ffmpeg_not_available': 'FFmpeg未安装，无法提取音频',      # FFmpeg不可用时的错误消息
    'video_read_first_error':"无法读取视频文件的第一帧",        # 无法读取视频文件的第一帧时的错误消息
    'video_read_frame_error': '读取视频帧时发生错误',        # 读取视频帧过程中发生错误时的消息
    'audio_extraction_error': '提取音频时发生错误: {}',        # 提取音频过程中发生错误时的消息
    'create_video_error': '创建视频文件时发生错误: {}',        # 创建视频文件时发生错误时的消息
    'get_output_path_image_invalid': 'IMAGE模式的get_output_path方法应返回4个值',  # 图像输出路径返回值错误
    'get_output_path_video_invalid': 'VIDEO模式的get_output_path方法应返回4个值',  # 视频输出路径返回值错误
    'value_error': '参数错误: {}',                       # 参数值错误
    'general_error': '处理失败: {}',                     # 一般错误
    'gpu_processing_image_failed': 'GPU处理{}失败: {}',       # GPU处理特定图像失败的错误
    'frame_processing_failed': '处理帧时出错: {}',             # 处理帧时出错的错误
    'serialize_params_failed': '无法序列化保存参数: {}',        # 无法序列化保存参数错误
    'save_process_error': '保存过程中发生错误: {}',            # 保存过程中发生错误
    'force_terminate_process_error': '强制终止进程时出错: {}',   # 强制终止进程时出错
    'kill_process_error': '使用kill终止进程时出错: {}',         # 使用kill终止进程时出错
    'save_failed': '保存 {} 到 {} 时出错: {}',            # 保存特定类型时出错的错误消息
    'save_operation_interrupted': '保存操作已中断: {}',             # 保存操作被中断
    'processing_interrupted': '处理已被用户中断',              # 处理被用户中断
    'video_processing_failed': '视频处理失败: {}',             # 视频处理失败错误
    'image_processing_failed': '图像处理失败: {}',             # 图像处理失败的错误
    'animated_image_processing_failed': '动图处理失败: {}',  # 动图处理失败的错误
}

# 成功消息模板
# 定义程序中使用的成功提示消息
SUCCESS_MESSAGES: Dict[str, str] = {
    'processing_complete': '✓ 处理完成！',                 # 处理完成时的成功消息
    'file_saved': '文件已保存: {}',                       # 文件保存成功时的消息模板
    'output_folder_cleaned': '已清理输出文件夹：{}',        # 清理输出文件夹后的消息
    'temp_folder_cleaned': '已清理临时文件夹：{}'           # 清理临时文件夹后的消息
}

# 断言消息模板
# 定义程序中使用的断言消息，支持中文翻译
ASSERT_MESSAGES: Dict[str, str] = {
    'fps_not_none': 'fps 不应为 None',                    # 帧率不应为None
    'text_path_not_none': 'text_path 不应为 None',  # text_path 不应为 Non
    'temp_image_dir_not_none': 'temp_image_dir 不应为 None',  # temp_image_dir 不应为 None
    'text_full_path_not_none': 'text_full_path 不应为 None',  # text_full_path不应为None
    'image_frame_path_not_none': 'image_frame_path 不应为 None',  # image_frame_path不应为None
    'image_path_not_none': 'image_path 不应为 None',  # image_path 不应为 None
    'temp_dir_not_none': 'temp_dir 不应为 None',  # temp_dir 不应为 None
}

# 警告消息模板
# 定义程序中使用的各种警告消息，支持格式化参数
WARNING_MESSAGES: Dict[str, str] = {
    'gpu_check_error': '检查CUDA可用性时出错: {}',  # 检查CUDA可用性时的警告消息
    'gpu_info_error': '获取GPU {} 信息时出错: {}',   # 获取GPU信息时的警告消息
    'video_no_audio': '视频 {} 不包含音频流，跳过音频提取',  # 视频不包含音频时的警告消息
    'gpu_acceleration_unavailable': 'GPU加速已请求但不可用，将使用CPU处理',  # GPU加速不可用时的警告消息
    'gpu_count_error': '获取GPU数量信息时出错: {}',    # 获取GPU数量错误
    'default_device_error': '设置默认设备时出错: {}',    # 设置默认设备错误
    'gpu_details_init_error': '初始化GPU详情时出错: {}',  # 初始化GPU详情错误
    'pytorch_init_error': '初始化PyTorch环境时出错: {}',  # 初始化PyTorch环境错误
    'gpu_memory_limit_failed': '设置GPU内存限制失败: {}',  # 设置GPU内存限制失败
    'gpu_cache_clean_failed': '清理GPU内存缓存失败: {}',   # 清理GPU内存缓存失败
    'fonttools_detection_failed': 'fonttools字体检测失败: {}',  # fonttools字体检测失败
    'font_load_failed': '无法加载字体 {}: {}',          # 字体加载失败
    'using_pil_default_font': '使用PIL默认字体',         # 使用PIL默认字体
    'font_manager_load_failed': 'FontManager加载字体失败: {}，使用默认字体',  # FontManager加载失败
    'image_size_exceeds_limit': '调整后原图尺寸({}x{})超过原图适合的字符尺寸({}x{})，将使用原图适合的尺寸',  # 图像尺寸超过限制
    'gpu_processing_failed': 'GPU处理失败，回退到CPU: {}',   # GPU处理失败，回退到CPU
    'animation_format_unsupported': '不支持的动画格式 {}，仅保存第一帧',  # 不支持的动画格式
    'process_interrupt_wait_error': '等待进程响应中断时出错: {}',  # 进程中断等待错误
    'process_force_termination': '保存进程未能及时响应中断，正在强制终止...',  # 强制终止进程
    'force_kill_failed': '强制终止失败，使用kill',        # 强制终止失败
    'subprocess_keyboard_interrupt': '子进程捕获到KeyboardInterrupt，正在清理资源...',  # 子进程捕获到KeyboardInterrupt
    'memory_check_failed': '内存检测失败: {}',  # 内存检测失败
    'ffprobe_frame_count_mismatch': 'FFprobe 统计的视频帧数({})与OpenCV读取的帧数({})不一致，使用FFprobe统计的帧数',  # FFprobe统计的视频帧数与OpenCV读取的帧数不一致
}
