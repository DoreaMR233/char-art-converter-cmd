"""
字符画转换器命令行工具

该脚本是字符画转换器的命令行入口，用于将图像或视频转换为字符画。
支持多种配置选项，包括字符密度、颜色模式、输出大小限制等。

主要功能：
- 支持图像和视频格式输入
- 多种字符密度级别可选
- 支持彩色和黑白字符画
- 可配置输出大小
- 支持GPU加速
- 支持多线程处理

依赖：
- argparse: 命令行参数解析
- logging: 日志记录
- signal: 信号处理
- sys: 系统相关功能
- threading: 线程处理
- pathlib: 路径操作
- typing: 类型注解
- src.configs: 配置模块
- src.enums: 枚举类型模块
- src.processors: 处理器模块
- src.utils: 工具函数模块

示例用法：
  char_art_converter.py input.jpg
  char_art_converter.py input.gif --color-mode color
  char_art_converter.py input.png --with-text
"""
# 首先导入最基础的模块和设置信号处理
import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

# 使用模块级别的变量替代builtins.exit_flag
exit_flag = False

def global_signal_handler(signum: int, frame) -> None:
    """
    全局信号处理器，在导入其他模块前设置
    Args:
        signum : int 接收到的信号编号
        frame : 当前执行帧对象
    
    Returns:
        None
    """
    global exit_flag
    print(f"\n收到信号 {signum}，当前帧: {frame}")
    if not exit_flag:
        exit_flag = True
        print("\n接收到中断信号，正在停止处理...", file=sys.stderr)
        # 设置较短的超时时间，确保能够退出
        def force_exit() -> None:
            """
            强制退出程序
            
            该方法用于在2秒内未收到其他信号时，强制退出程序。
            主要用于处理程序在处理过程中收到中断信号后，确保程序能够及时退出。
            
            Returns:
                None
            """
            print("\n强制退出程序...", file=sys.stderr)
            sys.exit(1)
        # 启动一个定时器，如果2秒后还没退出就强制退出
        timer = threading.Timer(2.0, force_exit)
        timer.daemon = True
        timer.start()

# 注册全局信号处理器
signal.signal(signal.SIGINT, global_signal_handler)

# 导入配置和工具
from src.configs import (
    DEFAULT_DENSITY, COLOR_MODES, DEFAULT_COLOR_MODE, DEFAULT_LIMIT_SIZE,
    DEFAULT_WITH_TEXT, ENABLE_GPU, ERROR_MESSAGES, CHAR_DENSITY_CONFIG, DEFAULT_WITH_IMAGE
)

# 创建logger
logger = logging.getLogger(__name__)

# 延迟导入较重的模块，避免在信号处理器设置前阻塞
from src.enums import FileType
from src.processors import ImageProcessor, VideoProcessor
from src.utils import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    该方法用于构建并配置命令行参数解析器，定义了所有可用的命令行选项，
    包括输入/输出设置、处理参数、性能优化选项等。
    
    Returns:
        argparse.ArgumentParser: 配置好的命令行参数解析器对象
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="将图像转换为字符画",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s input.jpg output.png
  %(prog)s input.gif output.gif --color-mode color
  %(prog)s input.png output.png --with-text """
    )

    # 必需参数
    parser.add_argument(
        'input',
        type=str,
        help='输入图像文件路径'
    )

    # 可选参数
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='设置输出目录（如果不指定，将在输入文件同目录生成）'
    )

    parser.add_argument(
        '-d', '--density',
        choices=list(CHAR_DENSITY_CONFIG.keys()),
        default=DEFAULT_DENSITY,
        help=f'字符密度级别 (默认: {DEFAULT_DENSITY}，可选值: {", ".join(CHAR_DENSITY_CONFIG.keys())})'
    )

    parser.add_argument(
        '-c', '--color-mode',
        choices=COLOR_MODES,
        default=DEFAULT_COLOR_MODE,
        help=f'颜色模式 (默认: {DEFAULT_COLOR_MODE}，可选值: {", ".join(COLOR_MODES)})'
    )

    parser.add_argument(
        '-l', '--limit-size',
        nargs='*',
        type=int,
        metavar='[LIMIT_WIDTH,LIMIT_HEIGHT]',
        default=DEFAULT_LIMIT_SIZE,
        help='调整输入图片尺寸。不带参数时使用默认大小(若指定了字体大小则为其1/2，否则原图宽度的1/4和原图高度的1/6)，带两个参数时指定宽度和高度'
    )

    parser.add_argument(
        '-t', '--with-text',
        action='store_true',
        default=DEFAULT_WITH_TEXT,
        help='同时输出字符画图像和文本文件'
    )

    parser.add_argument(
        '-i', '--with-image',
        action='store_true',
        default=DEFAULT_WITH_IMAGE,
        help='同时输出字符画视频/动图和字符画图像（仅当输入文件为视频或动图时有效）'
    )

    parser.add_argument(
        '--no-multithread',
        action='store_true',
        help='禁用多线程处理动画帧（默认启用多线程）'
    )

    parser.add_argument(
        '--disable-gpu',
        action='store_false',
        dest='enable_gpu',
        default=ENABLE_GPU,
        help='禁用GPU并行计算加速，使用CPU处理（默认启用GPU）'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='启用DEBUG级别日志输出'
    )

    parser.add_argument(
        '--gpu-memory-limit',
        type=int,
        metavar='MB',
        help='设置GPU内存限制（MB），默认使用配置文件中的值'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser

def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    程序主入口函数
    
    该方法是程序的核心执行函数，负责：
    1. 解析命令行参数
    2. 配置日志记录
    3. 创建ImageProcessor或VideoProcessor实例
    4. 启动图像处理流程
    5. 处理各种可能的异常情况
    6. 响应全局中断信号
    
    Args:
        args : Optional[argparse.Namespace] 命令行参数命名空间对象，如果为None则从命令行解析
    
    Returns:
        int: 执行状态码，0表示成功，非0表示失败
            0: 成功
            1: 一般错误
            2: 文件未找到错误
            130: 用户中断
    
    Raises:
        KeyboardInterrupt: 用户中断处理
        FileNotFoundError: 找不到指定的文件
        ValueError: 参数值无效
        Exception: 其他未预期的异常
    """
    global exit_flag
    
    # 解析命令行参数
    if args is None:
        parser: argparse.ArgumentParser = create_parser()
        parsed_args: argparse.Namespace = parser.parse_args()
    else:
        parsed_args = args

    input_path: Path = Path(parsed_args.input)
    file_type: FileType = FileType.from_path(input_path)
    # 设置日志
    setup_logging(verbose=parsed_args.debug)  # 根据debug参数设置日志级别
    
    # 传递全局exit_flag给处理器
    parsed_args.exit_flag = exit_flag
    
    try:
        # 检查是否已经收到中断信号
        if exit_flag:
            logger.info(ERROR_MESSAGES['processing_interrupted'])
            return 130
            
        if file_type == FileType.IMAGE:
            image_processor: ImageProcessor = ImageProcessor(parsed_args)
            image_processor.start()
        elif file_type == FileType.VIDEO:
            video_processor: VideoProcessor = VideoProcessor(parsed_args)
            video_processor.start()
        return 0
    except KeyboardInterrupt:
        # 确保在KeyboardInterrupt中也设置exit_flag

        exit_flag = True
        logger.info(ERROR_MESSAGES['processing_interrupted'])
        return 130
    
    except FileNotFoundError as e:
        logger.error(ERROR_MESSAGES['file_not_found'].format(e))
        return 2
    
    except ValueError as e:
        logger.error(ERROR_MESSAGES['value_error'].format(e))
        return 1
    
    except Exception as e:
        logger.error(ERROR_MESSAGES['general_error'].format(e))
        return 1


if __name__ == '__main__':
    sys.exit(main())