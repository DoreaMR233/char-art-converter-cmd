


"""
Windows 控制台中断处理工具模块

该模块解决 Windows 上 Ctrl+Break（CTRL_BREAK_EVENT，对应 SIGBREAK）无法触发应用
优雅中断的问题。PyTorch 会间接载入 Intel Fortran 运行时（随 MKL 一起分发），该运行时
在 Python 之后注册自己的控制台控制处理器；Windows 按"后注册先调用"的顺序调用控制台
控制处理器，因此 CTRL_BREAK 会被该运行时抢先处理，打印
"forrtl: error (200): program aborting due to control-BREAK event" 并进入 CRT 原生
终止路径。当 CUDA 上下文处于活跃状态时该路径不会返回，进程将永久挂起，Python 层的
中断逻辑（`BasedProcessor.signal_handler`）完全没有机会执行。

主要功能：
- 在 Windows 上注册控制台控制处理器，抢占 CTRL_C / CTRL_BREAK
- 将控制台事件转发为 SIGINT，复用应用已有的优雅中断逻辑
- 非 Windows 平台或重复调用时安全退化

依赖：
- ctypes：调用 kernel32.SetConsoleCtrlHandler
- signal：转发 SIGINT
- sys：平台判断
- logging：用于日志记录

"""
import logging
import signal
import sys
from typing import Any, Optional

from src.configs import WARNING_MESSAGES

# 初始化日志器
logger = logging.getLogger(__name__)

# Windows 控制台控制事件类型（wincon.h）
CTRL_C_EVENT: int = 0
CTRL_BREAK_EVENT: int = 1
CTRL_CLOSE_EVENT: int = 2
CTRL_LOGOFF_EVENT: int = 5
CTRL_SHUTDOWN_EVENT: int = 6

# 需要转发为 SIGINT 的控制台事件
_INTERRUPT_EVENTS: tuple = (CTRL_C_EVENT, CTRL_BREAK_EVENT)

# ctypes 回调对象引用，必须保持存活，否则回调可能崩溃
_console_handler: Optional[Any] = None


def _handle_console_event(ctrl_type: int) -> bool:
    """
    控制台控制处理器回调，将中断类控制台事件转发为 SIGINT

    该回调由 Windows 在系统创建的控制台线程中调用（不是主线程）。返回 True 表示事件
    已被处理，不再交给后续处理器（Python 默认处理器与 Intel Fortran 运行时）。

    Args:
        ctrl_type: int 控制台控制事件类型，取值见 CTRL_C_EVENT / CTRL_BREAK_EVENT 等常量

    Returns:
        bool: True 表示已处理该事件（中断类事件已转发为 SIGINT）；
            False 表示不处理，交给后续控制台处理器
    """
    if ctrl_type in _INTERRUPT_EVENTS:
        signal.raise_signal(signal.SIGINT)
        return True
    return False


def install_console_interrupt_handler() -> bool:
    """
    注册 Windows 控制台控制处理器，使 CTRL_C / CTRL_BREAK 走应用的优雅中断流程

    必须在所有可能注册控制台控制处理器的库（PyTorch 及其 MKL / Intel Fortran 运行时）
    载入之后调用，否则会被这些后注册的处理器抢先。函数幂等，重复调用不会重复注册。

    Args:
        无

    Returns:
        bool: True 表示当前进程已注册本模块的控制台处理器（包含重复调用）；
            False 表示非 Windows 平台或注册失败
    """
    global _console_handler

    if sys.platform != "win32":
        return False
    if _console_handler is not None:
        return True

    try:
        import ctypes

        handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        _console_handler = handler_type(_handle_console_event)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.SetConsoleCtrlHandler(_console_handler, True):
            _console_handler = None
            logger.warning(f"{WARNING_MESSAGES['console_ctrl_handler_register_failed'].format(ctypes.get_last_error())}")
            return False
    except Exception as e:
        _console_handler = None
        logger.warning(f"{WARNING_MESSAGES['console_ctrl_handler_register_failed'].format(e)}")
        return False

    logger.debug("已注册 Windows 控制台中断处理器（CTRL_BREAK / CTRL_C 将转发为 SIGINT）")
    return True
