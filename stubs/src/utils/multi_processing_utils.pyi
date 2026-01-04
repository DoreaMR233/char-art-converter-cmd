import multiprocessing
from typing import Callable, Any, Optional, Dict, Tuple


def _execute_with_exception_capture(func: Callable, result_queue: multiprocessing.Queue, args: Tuple = None, kwargs: Dict = None) -> None:
    """
    在子进程中执行函数，并捕获可能的异常
    
    Args:
        func: 要执行的函数
        result_queue: 用于传递结果或异常的队列
        args: 函数的位置参数
        kwargs: 函数的关键字参数
    """
    ...

def run_with_interrupt_support(func: Callable, should_stop: Optional[Callable[[], bool]] = None,
                               timeout: Optional[float] = None, **kwargs) -> Any:
    """
    使用多进程执行函数，支持中断和异常传递，具有增强的资源释放机制
    
    Args:
        func: 要执行的函数
        should_stop: 可选的函数，用于检查是否应该停止执行
        timeout: 可选的超时时间（秒）
        **kwargs: 传递给func的关键字参数
        
    Returns:
        函数的执行结果
        
    Raises:
        Exception: 如果子进程中发生异常，会将异常传递到主进程并抛出
        KeyboardInterrupt: 如果should_stop返回True或执行超时，会中断执行并抛出KeyboardInterrupt
    
    注意：此函数实现了增强的资源释放机制：
    1. 在中断时使用3秒等待时间确保子进程正常终止
    2. 如进程仍存活，使用kill强制终止并等待2秒
    3. 添加显式休眠确保操作系统完全释放文件句柄
    4. 关闭队列并等待线程完成
    5. 在所有资源清理完成后再抛出异常
    
    这些措施有效避免了文件清理时出现的"另一个程序正在使用此文件"错误。
    """
    ...
