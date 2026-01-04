
"""
日志工具模块

该模块提供了日志系统相关的工具函数，负责配置和管理应用程序的日志记录功能。

主要功能：
- 应用程序日志系统配置
- 日志级别设置和管理
- 日志格式定制
- 当前日志配置获取
- 第三方库日志级别控制

依赖：
- logging：Python标准日志库
- typing：类型注解支持

"""
import logging
from typing import Optional
import colorama

# 初始化colorama
colorama.init(autoreset=True, strip=False, convert=True)

from ..configs import DEFAULT_LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# 自定义彩色日志格式化器
class ColoredFormatter(logging.Formatter):
    """
    自定义彩色日志格式化器类，为不同级别的日志添加对应的颜色前缀
    Args:
        继承自logging.Formatter，使用父类的参数进行初始化
    """
    
    def format(self, record):
        """
        格式化日志记录，为不同级别添加彩色前缀

        Args:
            record : logging.LogRecord 日志记录对象，包含日志消息、级别、时间等信息

        Returns:
            str : 格式化后的日志消息，包含颜色前缀

        """
        # 调用父类的format方法获取格式化的消息
        formatted_message = super().format(record)
        
        # 根据日志级别添加颜色
        if record.levelno == logging.ERROR:
            return f"{colorama.Fore.RED}{formatted_message}{colorama.Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            return f"{colorama.Fore.YELLOW}{formatted_message}{colorama.Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            return f"{colorama.Fore.CYAN}{formatted_message}{colorama.Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            return f"{colorama.Fore.MAGENTA}{formatted_message}{colorama.Style.RESET_ALL}"
        
        # 其他级别保持原样
        return formatted_message


def setup_logging(verbose: bool = False, log_level: Optional[str] = None) -> None:
    """配置应用程序日志系统
    
    该函数初始化和配置应用程序的日志系统，设置适当的日志级别、格式和处理程序。
    它会清除现有的处理器以避免重复输出，并设置第三方库的日志级别。
    
    Args:
        verbose: bool 是否启用详细日志模式（DEBUG级别）
        log_level: Optional[str] 指定的日志级别字符串（如 'INFO', 'WARNING', 'ERROR' 等）。
                  如果为None且verbose为False，则使用默认日志级别。
    
    Returns:
        None
    """
    if verbose:
        level = logging.DEBUG
    else:
        # 使用指定的日志级别或默认日志级别
        level_str = log_level or DEFAULT_LOG_LEVEL
        level = getattr(logging, level_str.upper(), logging.INFO)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 清除现有的处理器，避免重复输出
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建彩色格式化器
    formatter = ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # 创建并配置流处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # 直接配置根日志记录器
    root_logger.setLevel(level)
    
    # 添加我们的彩色处理器
    root_logger.addHandler(stream_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    if verbose:
        logging.getLogger('src').setLevel(logging.DEBUG)


def get_current_log_config() -> dict:
    """获取当前使用的日志配置
    
    Returns:
        dict: 包含当前日志配置的字典，包括日志级别、格式和日期格式
    """
    root_logger = logging.getLogger()
    
    # 获取当前日志级别
    # 先检查是否有处理器，如果有则从处理器获取级别，否则使用根日志记录器的级别
    current_level = logging.getLevelName(root_logger.level)
    if root_logger.handlers:
        # 处理器的级别可能比根日志记录器更具体
        for handler in root_logger.handlers:
            if handler.level > logging.NOTSET:
                current_level = logging.getLevelName(handler.level)
                break
    
    # 获取当前日志格式和日期格式
    current_format: Optional[str] = LOG_FORMAT
    current_date_format: str = LOG_DATE_FORMAT
    
    # 从第一个处理器获取当前格式（如果存在）
    if root_logger.handlers:
        first_handler = root_logger.handlers[0]
        if hasattr(first_handler, 'formatter') and first_handler.formatter:
            # 类型断言：确保 _fmt 是 str 类型
            current_format = first_handler.formatter._fmt  # type: ignore[assignment]
            datefmt: Optional[str] = first_handler.formatter.datefmt
            # 如果datefmt为None，使用默认格式
            if datefmt is None:
                current_date_format = LOG_DATE_FORMAT
            else:
                # 类型断言：确保 datefmt 是 str 类型
                current_date_format = datefmt  # type: ignore[assignment]
    
    return {
        'level': current_level,
        'format': current_format,
        'datefmt': current_date_format
    }
