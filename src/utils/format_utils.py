"""
格式转换工具模块

该模块提供了一系列用于数据格式转换的工具函数，
包括时间、速度和文件大小的格式化功能，
用于将数值转换为人类可读的字符串表示。

主要功能：
- 时间格式化：将秒数转换为可读的时间字符串（时:分:秒）
- 速度格式化：将字节数和时间转换为可读的速度格式（B/s到TB/s）
- 文件大小格式化：将字节数转换为可读的存储单位（B、KB、MB、GB、TB）

这些工具函数广泛应用于进度显示、性能统计和用户反馈等场景，
确保输出信息清晰易读。

依赖：
- 无第三方库依赖，仅使用Python标准库
"""


def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串
    
    根据秒数大小自动转换为合适的时间单位（秒、分、时）
    
    Args:
        seconds: float 需要格式化的秒数
        
    Returns:
        str: 格式化后的时间字符串，格式为：
            - 小于60秒："X.X秒"
            - 小于1小时："X分X.X秒"
            - 大于等于1小时："X时X分X.X秒"
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes: int = int(seconds // 60)
        secs: float = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours: int = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)  # 复用之前定义的变量
        secs = seconds % 60  # 复用之前定义的变量
        return f"{hours}时{minutes}分{secs:.1f}秒"

def format_speed(size_bytes_or_speed: float, seconds: float = 1.0) -> str:
    """
    计算并格式化传输速度
    
    将字节数和时间转换为可读的速度格式，支持多种单位（B/s到TB/s）
    
    Args:
        size_bytes_or_speed: float 如果seconds=1.0，则表示已计算好的速度；
                          否则表示字节数，需要根据seconds计算速度
        seconds: float 传输时间（秒），默认为1.0
        
    Returns:
        str: 格式化后的速度字符串，例如："10.50 KB/s"
    """
    if seconds <= 0:
        return "0 B/s"
    
    # 如果seconds=1.0，则认为传入的是已计算好的速度
    speed: float = size_bytes_or_speed if seconds == 1.0 else size_bytes_or_speed / seconds
    units: list[str] = ['B', 'KB', 'MB', 'GB', 'TB']
    
    unit: str
    for unit in units:
        if speed < 1024.0:
            return f"{speed:.2f} {unit}/s"
        speed /= 1024.0
    
    return f"{speed:.2f} {units[-1]}/s"


def format_size(size_bytes: float) -> str:
    """
    将字节数格式化为人类可读的文件大小字符串
    
    根据字节数大小自动转换为合适的存储单位（B、KB、MB、GB、TB）
    
    Args:
        size_bytes: float 需要格式化的字节数
        
    Returns:
        str: 格式化后的文件大小字符串，例如："1024.0 KB"或"1.0 MB"
    """
    # 直接格式化字节数，不再通过文件路径
    unit: str
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"