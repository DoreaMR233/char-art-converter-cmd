"""
多进程处理工具模块

该模块提供了多进程执行函数的工具函数，支持中断控制、异常传递和超时处理等功能。

主要功能：
- 子进程异常捕获和传递
- 支持外部中断信号处理
- 函数执行超时控制
- 资源安全清理

依赖：
- multiprocessing: 用于创建和管理子进程
- queue: 用于进程间通信
- traceback: 用于异常堆栈捕获
- typing: 用于类型注解
- logging: 用于日志记录
- time: 用于超时控制
- ..configs.message_config: 用于错误消息引用
"""
import multiprocessing
import queue
import traceback
from typing import Callable, Any, Optional, Dict, Tuple
import logging
import time
from ..configs.message_config import ERROR_MESSAGES

logger = logging.getLogger(__name__)


def _execute_with_exception_capture(func: Callable, result_queue: multiprocessing.Queue, args: Optional[Tuple] = None, kwargs: Optional[Dict] = None) -> None:
    """
    在子进程中执行函数，并捕获可能的异常
    
    Args:
        func: Callable 要执行的函数
        result_queue: multiprocessing.Queue 用于传递结果或异常的队列
        args: Optional[Tuple] 函数的位置参数
        kwargs: Optional[Dict] 函数的关键字参数
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    
    try:
        # 执行函数并获取结果
        result = func(*args, **kwargs)
        # 将结果放入队列，格式为 (True, result)
        result_queue.put((True, result))
    except Exception as e:
        # 捕获异常，获取异常的堆栈信息
        error_trace = traceback.format_exc()
        logger.error(f"子进程执行失败:\n{error_trace}")
        # 将异常信息放入队列，格式为 (False, (exception_type, exception_value, error_trace))
        result_queue.put((False, (type(e), e, error_trace)))


def run_with_interrupt_support(func: Callable, should_stop: Optional[Callable[[], bool]] = None, 
                               timeout: Optional[float] = None, **kwargs) -> Any:
    """
    使用多进程执行函数，支持中断和异常传递
    
    Args:
        func: Callable 要执行的函数
        should_stop: Optional[Callable[[], bool]] 可选的函数，用于检查是否应该停止执行
        timeout: Optional[float] 可选的超时时间（秒）
        **kwargs: Any 传递给func的关键字参数
        
    Returns:
        Any: 函数的执行结果
        
    Raises:
        Exception: 如果子进程中发生异常，会将异常传递到主进程并抛出
        KeyboardInterrupt: 如果should_stop返回True或执行超时，会中断执行并抛出KeyboardInterrupt
    """
    # 创建队列用于传递结果和异常
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    
    # 创建子进程
    process = multiprocessing.Process(
        target=_execute_with_exception_capture,
        args=(func, result_queue),
        kwargs={'kwargs': kwargs}
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 存储需要抛出的异常
    exception_to_raise = None
    
    try:
        # 启动子进程
        process.start()
        
        # 等待子进程完成或检查是否需要中断
        while process.is_alive():
            # 检查是否需要中断
            if should_stop is not None and should_stop():
                logger.info("检测到中断信号，终止子进程")
                # 设置需要抛出的异常，但先不抛出
                exception_to_raise = KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'].format(func.__name__))
                break
            
            # 检查是否超时
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.info("执行超时，终止子进程")
                # 设置需要抛出的异常，但先不抛出
                exception_to_raise = KeyboardInterrupt(f"函数 {func.__name__} 执行超时")
                break
            
            # 检查是否有结果可用
            try:
                if not result_queue.empty():
                    success, result = result_queue.get_nowait()
                    if success:
                        # 函数执行成功，返回结果
                        return result
                    else:
                        # 函数执行失败，存储异常信息，但先不抛出
                        exception_type, exception_value, error_trace = result
                        exception_to_raise = exception_type(str(exception_value))
                        break
                
                # 使用超时等待，避免阻塞
                process.join(timeout=0.05)  # 增加检查间隔，减少CPU使用率
            except queue.Empty:
                pass
            except EOFError:
                logger.warning("队列已关闭，可能是子进程异常终止")
                break
            except KeyboardInterrupt:
                # 捕获主进程的Ctrl+C中断
                logger.info("捕获到键盘中断，准备终止子进程")
                exception_to_raise = KeyboardInterrupt()
                break
        
        # 检查进程是否正常结束
        if process.exitcode != 0 and exception_to_raise is None:
            logger.error(f"子进程异常结束，退出码: {process.exitcode}")
            # 尝试从队列获取异常信息
            try:
                if not result_queue.empty():
                    success, result = result_queue.get_nowait()
                    if not success:
                        exception_type, exception_value, error_trace = result
                        exception_to_raise = exception_type(str(exception_value))
            except (queue.Empty, ValueError):
                pass
            
            # 如果没有异常信息，设置一个通用异常
            if exception_to_raise is None:
                exception_to_raise = KeyboardInterrupt(ERROR_MESSAGES['save_failed'].format(func.__name__, "未知目标", f"进程异常退出，退出码: {process.exitcode}"))
        
        # 再次检查队列，确保获取所有结果
        if exception_to_raise is None and not result_queue.empty():
            success, result = result_queue.get_nowait()
            if success:
                return result
            else:
                exception_type, exception_value, error_trace = result
                exception_to_raise = exception_type(str(exception_value))
                
    finally:
        # 确保子进程被终止
        if process.is_alive():
            try:
                logger.debug("确保子进程被终止")
                # 首先尝试正常终止
                process.terminate()
                
                # 增加更长的等待时间，确保文件资源被释放
                process.join(timeout=3.0)  # 增加超时时间到3秒
                
                # 如果进程仍然存活，尝试强制终止
                if process.is_alive():
                    logger.warning("子进程没有在指定时间内终止，强制终止")
                    process.kill()
                    process.join(timeout=2.0)  # 增加kill后的等待时间到2秒
                    
                # 再次检查进程状态
                if process.is_alive():
                    logger.error("子进程仍然存活，可能存在资源泄漏")
            except Exception as e:
                logger.error(f"终止子进程时出错: {str(e)}")
        
        # 显式等待以确保操作系统完全释放文件句柄
        time.sleep(0.5)  # 短暂休眠以确保文件句柄释放
        
        # 关闭队列
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception as e:
            logger.error(f"关闭队列时出错: {str(e)}")
        
        # 再次等待一小段时间，确保所有资源都已释放
        time.sleep(0.2)  # 额外的短暂等待
        
        # 所有资源清理完成后，再抛出异常
        if exception_to_raise is not None:
            logger.debug(f"资源清理完成后抛出异常: {type(exception_to_raise).__name__}")
            raise exception_to_raise
