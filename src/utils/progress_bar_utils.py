"""
进度条工具模块

该模块提供了用于显示各种任务进度的工具函数，主要使用tqdm库实现进度条功能，
支持多种进度条样式和配置，为字符画转换过程提供可视化进度反馈。

主要功能：
- 标准化的tqdm配置参数生成
- 文件保存进度显示（带实时大小和速度）
- 字符转换进度显示
- 图像处理进度显示
- 视频帧处理进度显示
- 自定义动画式进度条
- 支持多进度条同时显示
- 支持不同位置的进度条布局
- 进度单位和格式定制

依赖：
- tqdm：用于进度条实现
- threading：用于多线程进度更新
- time：用于时间计算
- logging：用于日志记录
- format_utils：用于数据格式化

"""
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any
from typing import Optional, Callable

import colorama

# 初始化colorama
colorama.init(autoreset=True, strip=False, convert=True)

from .format_utils import format_size, format_speed

# 初始化日志器
logger = logging.getLogger(__name__)

def get_tqdm_kwargs(total: int, desc: str, verbose: bool = True, unit: str = 'it', position: int = 0) -> Dict[str, Any]:
    """
    获取tqdm进度条配置参数
    
    生成标准化的tqdm进度条配置，确保在不同场景下的进度条显示一致。
    包含进度条格式、行为和显示选项的完整配置。
    
    参数:
        total: int
            总进度值
        desc: str
            进度条描述文本
        verbose: bool
            是否显示进度条
        unit: str
            进度单位
        position: int
            进度条在终端中的位置
    
    返回:
        Dict[str, Any]:
            tqdm进度条的配置参数字典
    """

    # 设置绿色的进度条格式
    green_bar = f'{colorama.Fore.GREEN}{{bar}}{colorama.Style.RESET_ALL}'
    
    return {
        'total': total,
        'desc': desc,
        'unit': unit,
        'leave': position == 0,  # 主进度条保留，子进度条不保留
        'ncols': None,  # 让tqdm自动适应终端宽度
        'ascii': True,  # 在Windows中使用ASCII字符
        'dynamic_ncols': True,  # 启用动态列宽，适应不同终端
        'file': sys.stdout,  # 统一使用stdout
        'bar_format': f'{{desc}}: {{percentage:3.0f}}%|{green_bar}| {{n_fmt}}/{{total_fmt}} {{postfix}}',
        'position': position,
        'disable': not verbose,
        'smoothing': 0.1,  # 添加平滑处理以提高显示效果
        'miniters': 1,  # 确保每次迭代都更新进度条
        'mininterval': 0.1  # 最小更新间隔，提高响应速度
    }

def show_value_file_save_progress(file_path: Path, save_completed: threading.Event, description: str, verbose: bool, position: int = 0, should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    显示带值的文件保存进度
    
    创建一个带有进度值和百分比显示的文件保存进度条，实时估算文件大小和保存速度
    
    参数:
        file_path: Path
            文件路径，用于获取当前保存大小
        save_completed: threading.Event
            保存完成事件，用于通知进度条停止更新
        description: str
            进度条描述文本
        verbose: bool
            是否显示进度条
        position: int
            进度条在终端中的显示位置
        should_stop: Optional[Callable[[], bool]]
            可选的中断检查函数，返回True时中断进度条
    
    返回:
        None
    """
    # 如果不显示进度条，直接等待完成
    if not verbose:
        save_completed.wait()
        return

    # 初始化变量
    start_time = time.time()
    last_size = 0
    last_time = start_time
    estimated_total_size = None
    # 添加稳定性参数
    size_history = []  # 用于跟踪大小变化历史
    size_estimation_count = 0  # 限制估算频率

    # 使用tqdm显示进度
    tqdm_kwargs = get_tqdm_kwargs(1, description, verbose, 'it', position)
    tqdm_kwargs.update({
        'ncols': 100,
    })

    # 直接从tqdm模块导入，避免变量名冲突
    from tqdm import tqdm as tqdm_func
    with tqdm_func(**tqdm_kwargs) as pbar:
        while not save_completed.is_set():
            try:  
                # 检查是否需要中断
                if should_stop and should_stop():
                    raise KeyboardInterrupt
                
                try:
                    # 获取当前文件大小
                    current_size = 0
                    if file_path.exists():
                        try:
                            current_size = file_path.stat().st_size
                        except (OSError, IOError):
                            current_size = 0
                except (OSError, IOError):
                    current_size = 0

                current_time = time.time()
                elapsed_time = current_time - start_time

                # 计算写入速度
                time_diff = current_time - last_time
                if time_diff > 0:
                    speed = (current_size - last_size) / time_diff
                else:
                    speed = 0

                # 存储大小历史，用于更稳定的估算
                size_history.append((current_time, current_size))
                # 只保留最近的10个记录
                if len(size_history) > 10:
                    size_history.pop(0)
                
                # 更新最后记录的大小和时间
                last_size = current_size
                last_time = current_time
                
                # 避免CPU占用过高
                time.sleep(0.1)
                # 改进的文件大小估算算法
                if speed > 0 and elapsed_time > 1:  # 至少等待1秒再估算
                    size_estimation_count += 1
                    # 限制估算频率，避免频繁跳动
                    if size_estimation_count >= 3 or estimated_total_size is None:
                        size_estimation_count = 0

                        # 使用历史数据进行更稳定的估算
                        if len(size_history) > 3:
                            # 计算平均增长率
                            growth_rates = []
                            for i in range(1, len(size_history)):
                                time_diff = size_history[i][0] - size_history[i - 1][0]
                                size_diff = size_history[i][1] - size_history[i - 1][1]
                                if time_diff > 0 and size_diff > 0:
                                    growth_rates.append(size_diff / time_diff)

                            if growth_rates:
                                avg_growth_rate = sum(growth_rates) / len(growth_rates)
                                # 考虑最近的增长趋势
                                recent_growth = (current_size - size_history[0][1]) / (
                                            current_time - size_history[0][0])
                                # 加权平均
                                weighted_growth = (avg_growth_rate * 0.7) + (recent_growth * 0.3)

                                # 使用加权增长率估算剩余时间
                                if estimated_total_size is None:
                                    # 初始估算：基于当前大小和增长率
                                    estimated_total_size = current_size * 2
                                else:
                                    # 动态调整估算，避免过大波动
                                    new_estimate = current_size + weighted_growth * 5  # 预估5秒内完成
                                    # 平滑过渡到新的估算值
                                    estimated_total_size = int(estimated_total_size * 0.7 + new_estimate * 0.3)
                        else:
                            # 数据不足时的简单估算
                            if estimated_total_size is None:
                                estimated_total_size = int(current_size * 2)
                            elif current_size > estimated_total_size * 0.8:
                                estimated_total_size = int(estimated_total_size * 1.2)  # 更保守的增长

                # 计算进度百分比
                if estimated_total_size and estimated_total_size > 0:
                    progress = min(95, int((current_size / estimated_total_size) * 100))
                else:
                    # 没有估算值时，使用时间基础的进度，但更平滑
                    progress = min(90, int((elapsed_time / 60) * 100))  # 假设1分钟内完成，最多90%

                # 更新进度条，避免剧烈跳动
                if abs(pbar.n - progress) > 2:  # 只有变化超过2%才更新
                    pbar.n = progress

                # 设置后缀信息
                if estimated_total_size:
                    size_info = f"{format_size(current_size)}/{format_size(int(estimated_total_size))}"
                else:
                    size_info = f"{format_size(current_size)}"

                speed_info = format_speed(speed)
                pbar.set_postfix_str(f"{size_info} | {speed_info}")
                pbar.refresh()

                # 更新上次记录的值
                last_size = current_size
                last_time = current_time

                time.sleep(0.1)  # 每0.1秒更新一次
            except KeyboardInterrupt:
                logger.info("文本文件保存操作已中断")
                raise
            except Exception as e:
                # 捕获所有异常，确保进度条不会崩溃
                logger.debug(f"进度条更新异常: {e}")
                time.sleep(0.1)
                continue

        # 保存完成，获取最终文件大小
        try:
            if file_path.exists():
                final_size = file_path.stat().st_size
                total_time = time.time() - start_time
                avg_speed = final_size / total_time if total_time > 0 else 0

                # 更新进度到100%
                pbar.n = 1
                pbar.set_postfix_str(f"{format_size(final_size)} | 平均 {format_speed(avg_speed)}")
                pbar.refresh()
            else:
                pbar.n = 1
                pbar.set_postfix_str("完成")
                pbar.refresh()
        except (OSError, IOError):
            pbar.n = 1
            pbar.set_postfix_str("出现异常")
            pbar.refresh()

def show_project_status_progress(total: int, description: str, verbose: bool = True, unit: str = 'it', position: int = 0, hide_counter: bool = False) -> Any:
    """
    显示项目状态进度条（通用方法）
    
    创建并返回一个用于显示各种项目进度的tqdm进度条。
    这是一个通用方法，可以替代所有特定类型的进度条显示函数。
    
    参数:
        total: int
            总进度值，通常为需要处理的项目数量或步骤数
        description: str
            进度条描述文本
        verbose: bool
            是否显示进度条
        unit: str
            进度单位
        position: int
            进度条在终端中的位置
        hide_counter: bool
            是否隐藏计数器部分（已读取数量和总数）
    
    返回:
        Any:
            tqdm进度条对象，可用于在调用处更新进度
    """

    # 获取基本tqdm配置
    tqdm_kwargs = get_tqdm_kwargs(total, description, verbose, unit, position)
    
    # 如果需要隐藏计数器，使用特殊的bar_format
    if hide_counter:
        tqdm_kwargs['bar_format'] = '{desc} {percentage:3.0f}%|{bar}| {postfix}'
    
    # 直接从tqdm模块导入，避免变量名冲突
    from tqdm import tqdm as tqdm_func
    
    # 返回tqdm进度条对象，以便在调用处控制更新
    return tqdm_func(**tqdm_kwargs)

def no_value_file_save_progress(file_path: Path, save_completed: threading.Event, description: str, verbose: bool, position: int = 0, should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    显示无值的文件保存进度
    
    创建一个动画式的文件保存进度条，不显示具体进度百分比，而是通过滚动动画表示正在进行
    
    参数:
        file_path: Path
            正在保存的文件路径
        save_completed: threading.Event
            用于通知保存完成的事件对象
        description: str
            进度条描述文本
        verbose: bool
            是否显示进度条
        position: int
            进度条在终端中的显示位置
        should_stop: Optional[Callable[[], bool]]
            可选的停止检查函数，返回True时中断进度条
    
    返回:
        None
    """
    # 如果不显示进度条，直接等待完成
    if not verbose:
        save_completed.wait()
        return

    # 初始化变量
    start_time = time.time()
    
    # 动画相关变量
    animation_width = 30
    animation_pos = 0
    direction = 1
    block_width = 5
    block_char = f'{colorama.Fore.GREEN}█{colorama.Style.RESET_ALL}'
    
    # 导入tqdm
    from tqdm import tqdm
    
    try:
        # 获取统一的tqdm配置参数
        tqdm_kwargs = get_tqdm_kwargs(1, description, verbose, 'it', position)
        # 覆盖特定的配置以实现动画效果
        tqdm_kwargs.update({
            'bar_format': '{desc}',
            'disable': False
        })
        time.sleep(1)
        # 创建自定义的tqdm进度条，使用统一配置
        with tqdm(**tqdm_kwargs) as pbar:
            # 持续更新进度条直到保存完成
            while not save_completed.is_set():
                # 检查是否需要中断
                if should_stop and should_stop():
                    raise KeyboardInterrupt
                    
                # 创建左右滚动动画效果
                bar = [' '] * animation_width
                for i in range(block_width):
                    pos = animation_pos + i
                    if 0 <= pos < animation_width:
                        bar[pos] = block_char
                
                # 生成滚动条字符串
                bar_str = '|' + ''.join(bar) + '|'
                
                # 自定义设置进度条显示，不显示百分比和速度
                # 使用pbar.desc来更新描述，包含动画条
                pbar.desc = f"{description}: {bar_str}"
                pbar.refresh()
                
                # 更新动画位置和方向
                animation_pos += direction
                if animation_pos >= animation_width - block_width or animation_pos <= 0:
                    direction *= -1
                    animation_pos = max(0, min(animation_width - block_width, animation_pos))
                
                # 控制更新频率
                time.sleep(0.1)
            
            # 保存完成，获取最终文件大小
            final_size = 0
            avg_speed = 0
            try:
                if file_path.exists():
                    final_size = file_path.stat().st_size
                    total_time = time.time() - start_time
                    if total_time > 0 and final_size > 0:
                        avg_speed = int(final_size / total_time)  # 转换为整数
            except (OSError, IOError):
                pass
            
            # 更新进度条到完成状态，显示文件大小和速度
            if final_size > 0:
                # 完成时显示完整信息
                green_block = f'{colorama.Fore.GREEN}█{colorama.Style.RESET_ALL}'
                full_bar = '|' + green_block * animation_width + '|'
                pbar.desc = f"{description}: {full_bar} {format_size(final_size)} | 平均 {format_speed(avg_speed)} | 完成"
            else:
                green_block = f'{colorama.Fore.GREEN}█{colorama.Style.RESET_ALL}'
                full_bar = '|' + green_block * animation_width + '|'
                pbar.desc = f"{description}: {full_bar} 完成"
            

            # 设置进度为100%并刷新
            pbar.n = 1
            pbar.refresh()
            # 等待1秒，确保进度条显示完整
            time.sleep(1)
    
    except (OSError, IOError, KeyboardInterrupt) as e:
        # 捕获常见异常，包括键盘中断
        logger.debug(f"进度条显示异常: {e}")
        try:
            # 确保显示完成状态
            sys.stdout.write(f"\r{description}: 出现异常\n")
            sys.stdout.flush()
        except (IOError, OSError):
            pass
    except Exception as e:
        # 捕获其他未预期的异常
        logger.debug(f"进度条显示未预期异常: {e}")
    finally:
        # 确保事件已设置，防止线程阻塞
        if not save_completed.is_set():
            save_completed.set()




