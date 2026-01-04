"""
验证工具函数模块

这个模块提供了用于验证输入文件、输出路径和命令行参数的工具函数
确保字符画转换工具接收到有效的输入和参数

主要功能：
- 输入文件验证（存在性、类型、格式）
- 输出路径验证（目录创建、可写性）
- 命令行参数有效性验证

依赖：
- os：操作系统接口
- pathlib：路径处理
- typing：类型注解支持
- src.configs：错误消息和支持的格式配置
- src.utils.file_utils：文件扩展名获取工具
"""

import os
from pathlib import Path
from typing import Any

from ..configs import ERROR_MESSAGES, ALL_SUPPORTED_FORMATS
from .file_utils import get_file_extension


def validate_input_file(file_path: Path) -> None:
    """
    验证输入文件是否有效
    
    检查文件是否存在、是否为文件类型，以及是否为支持的格式
    
    参数:
        file_path Path:
            Path 对象，表示要验证的文件路径
    
    返回:
        None
    
    Raises:
        FileNotFoundError: 当文件不存在时抛出
        ValueError: 当路径不是文件类型或文件格式不受支持时抛出
    """
    if not file_path.exists():
        raise FileNotFoundError(ERROR_MESSAGES['file_not_found'].format(file_path))
    
    if not file_path.is_file():
        raise ValueError(ERROR_MESSAGES['path_not_file'].format(file_path))
    
    file_ext = get_file_extension(file_path)
    if file_ext not in ALL_SUPPORTED_FORMATS:
        raise ValueError(
            ERROR_MESSAGES['unsupported_format'].format(
                file_ext, ', '.join(ALL_SUPPORTED_FORMATS)
            )
        )

def validate_output_path(output_path: Path) -> None:
    """
    验证输出路径是否有效
    
    检查输出路径的父目录是否存在，不存在则尝试创建，并验证父目录是否可写
    
    参数:
        output_path Path:
            Path 对象，表示输出文件或目录的路径
    
    返回:
        None
    
    Raises:
        ValueError: 当输出目录创建失败或目录不可写时抛出
    """
    parent_dir = output_path.parent
    
    # 确保父目录存在
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError(ERROR_MESSAGES['output_dir_not_writable'].format(parent_dir))
    
    # 检查目录是否可写
    if not os.access(parent_dir, os.W_OK):
        raise ValueError(ERROR_MESSAGES['output_dir_not_writable'].format(parent_dir))

def validate_arguments(args: Any) -> None:
    """
    验证命令行参数的有效性
    
    对输入文件、输出目录和limit_size参数进行全面验证
    
    参数:
        args Any:
            命令行参数对象，包含input、output和limit_size等属性
    
    返回:
        None
    
    Raises:
        FileNotFoundError: 当输入文件不存在时抛出
        ValueError: 当参数格式错误或路径无效时抛出
    """
    # 验证输入文件
    input_path = Path(args.input)
    validate_input_file(input_path)
    
    # 验证输出目录
    if args.output:
        output_path = Path(args.output)
        validate_output_path(output_path)
    
    # 验证limit_size参数
    if args.limit_size is not None:
        if len(args.limit_size) == 2:
            width, height = args.limit_size
            if width <= 0 or height <= 0:
                raise ValueError(ERROR_MESSAGES['invalid_limit_size'].format("宽度和高度必须大于0"))
        elif len(args.limit_size) != 0:
            raise ValueError(ERROR_MESSAGES['invalid_limit_size'].format("--limit-size参数格式错误，应为：无参数、或两个整数(宽度 高度)"))
