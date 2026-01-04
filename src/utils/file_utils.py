"""
文件操作工具模块

该模块提供了一系列用于文件和目录管理的工具函数，
包括路径生成、目录操作、文件保存、临时文件管理等功能，
是字符画转换过程中文件处理的核心组件。

主要功能：
- 生成输出文件路径
- 目录创建和管理
- 临时目录创建
- 文件扩展名处理
- 文件清理和删除
- 异步文件保存
- 进度跟踪和管理
- 视频和图像文件处理

依赖：
- pathlib: 用于路径操作
- cv2: 用于图像处理
- numpy: 用于数值计算
- shutil: 用于高级文件操作
- tempfile: 用于临时文件管理
- threading/multiprocessing: 用于并发处理
- ffmpeg: 用于视频处理（可选）
"""
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Callable, Any
import puremagic


from .progress_bar_utils import no_value_file_save_progress
from .save_uitls import save_char_art_text, save_static_char_art_image, save_animated_char_art_image, \
    save_file_by_ffmpeg
from .multi_processing_utils import run_with_interrupt_support
from ..configs import ERROR_MESSAGES, PNG_EXTENSIONS, APNG_EXTENSIONS
from ..enums import SaveModes, ColorModes, FileType

logger = logging.getLogger(__name__)
def get_output_path(is_animated: bool, input_path: Path, color_mode: ColorModes,
                   output_arg: Optional[str] = None, file_type: FileType = FileType.IMAGE) -> Tuple[Path, Path, Optional[Path], Path]:
    """
    计算输出文件路径
    
    根据输入路径、颜色模式、是否为动画和文件类型，生成相应的输出目录、文本文件、图像文件路径。
    对于视频文件，还会生成视频输出路径。支持自定义输出目录或使用默认的基于输入文件的命名规则。
    
    Args:
        is_animated : bool 输入图像/视频是否为动画
        input_path : Path 输入文件路径
        color_mode : ColorModes 颜色模式枚举值
        output_arg : Optional[str] = None 自定义输出目录路径
        file_type : FileType = FileType.IMAGE 文件类型枚举值，支持IMAGE和VIDEO类型，默认为图像类型
    
    Returns:
        Tuple[Path, Path, Optional[Path], Path]: 
            - 当file_type为IMAGE时: (输出目录路径, 文本文件路径, 图像文件路径或None, 图像文件路径)
            - 当file_type为VIDEO时: (输出目录路径, 文本文件路径, 图像文件路径, 视频文件路径)
    
    Raises:
        ValueError: 当指定的输出路径不是目录时抛出
        ValueError: 当提供的file_type不是有效的FileType枚举值时抛出
    """

    color_mode_value = color_mode.value
    input_stem = input_path.stem
    input_dir = input_path.parent
    if output_arg:
        output_path = Path(output_arg)
        # 检查是否为目录路径
        if output_path.exists() and not output_path.is_dir()  :
            raise ValueError(ERROR_MESSAGES['path_not_directory'].format(output_path))
    else:
        output_path = input_dir / f"{input_stem}_{color_mode_value}_char_art_{file_type.value}"

    # 确保目录存在
    ensure_dir_exists(output_path)
    if file_type == FileType.IMAGE:
        frame_path = None
        if is_animated:
            text_path = output_path / f"{input_stem}_{color_mode_value}_char_art_text"
            frame_path = output_path / f"{input_stem}_{color_mode_value}_char_art_frame"
        else:
            text_path = output_path / f"{input_stem}_{color_mode_value}_char_art.txt"
        input_ext = get_file_extension(input_path)
        image_path = output_path / f"{input_stem}_{color_mode_value}_char_art{input_ext}"
        return output_path, text_path, frame_path, image_path
    elif file_type == FileType.VIDEO:
        text_path = output_path / f"{input_stem}_{color_mode_value}_char_art_text"
        image_path = output_path / f"{input_stem}_{color_mode_value}_char_art_image"
        input_ext = get_file_extension(input_path)
        video_path = output_path / f"{input_stem}_{color_mode_value}_char_art{input_ext}"
        return output_path, text_path,image_path, video_path
    else:
        raise ValueError(ERROR_MESSAGES['invalid_file_type'].format(input_path))

def ensure_dir_exists(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    递归创建目录（包括父目录），如果目录已存在则不执行任何操作。
    
    Args:
        dir_path : Union[str, Path] 需要确保存在的目录路径
    
    Returns:
        Path: 规范化后的目录路径对象
    
    Raises:
        NotADirectoryError: 如果指定的路径已存在但不是一个目录
        PermissionError: 当创建目录时没有足够权限时抛出
        OSError: 当目录创建失败时抛出，包含详细错误信息
    """

    path = Path(dir_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        raise OSError(ERROR_MESSAGES['directory_creation_failed'].format(e))

def create_temp_dir(base_dir: Optional[str] = None, prefix: str = 'char_art_') -> Path:
    """
    创建临时目录
    
    常用于存储动画处理过程中的中间帧文件。
    
    Args:
        base_dir : Optional[str] = None 基础目录路径，如果为None则使用系统临时目录
        prefix : str = 'char_art_' 临时目录名称前缀
    
    Returns:
        Path: 创建的临时目录路径对象
    
    Raises:
        OSError: 当创建临时目录失败时抛出
    """

    if base_dir:
        base_path = ensure_dir_exists(base_dir)
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=base_path)
    else:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
    
    return Path(temp_dir)

def get_file_extension(file_path: Path) -> str:
    """
    根据文件类型获取文件扩展名
    
    根据指定的文件类型（图像或视频）调用相应的检测函数，返回标准的文件扩展名。
    支持处理不同类型文件的统一接口。
    
    Args:
        file_path : Path 文件路径对象
    
    Returns:
        str: 标准的文件扩展名（带点号，如.jpg、.mp4）
    
    Raises:
        None
    """
    from .image_utils import is_animated_image
    # 检查文件是否存在，如果不存在则直接返回文件扩展名
    if file_path.exists():
        # 使用puremagic库检测文件类型
        detected_types = puremagic.magic_file(str(file_path))
        # 获取置信度最高的文件类型
        suffix = detected_types[0].extension
        # 如果检测到PNG格式且文件是动画，返回APNG扩展名
        if suffix in PNG_EXTENSIONS:
            return APNG_EXTENSIONS[0] if is_animated_image(file_path) else suffix
        return suffix
    else:
        logging.debug(f"文件不存在: {file_path}，直接返回文件扩展名")
        suffix = file_path.suffix.lower()
        return suffix if suffix.startswith('.') else '.' + suffix


def cleanup_files(*paths: Union[str, Path]) -> None:
    """
    清理临时文件和目录

    递归删除指定的文件或目录。支持同时清理多个路径，并具有容错机制，
    即使某些路径删除失败，也会继续处理其他路径。

    Args:
        *paths : Union[str, Path] 需要清理的文件或目录路径（可变参数）

    Returns:
        None

    Note:
        成功清理的路径将记录debug日志，清理失败的路径将记录warning日志
    """

    for path in paths:
        path_obj = Path(path)
        try:
            if path_obj.exists():
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
                logging.debug(f"已清理临时文件: {path_obj}")
        except Exception as e:
            logging.warning(f"清理临时文件失败 {path_obj}: {e}")


def save_file(content: dict[str,Any], file_path: Path,
              description: str, save_mode: SaveModes, position: int = 0,
              should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    保存不同类型的文件
    
    根据指定的保存模式，将内容保存为文本、图像、动画或视频文件。支持进度显示和中断操作。
    
    Args:
        content : dict[str,Any] 包含要保存内容的字典，根据保存模式需要不同的键值对
        file_path : Path 保存文件的路径对象
        description : str 保存操作的描述信息，用于进度显示
        save_mode : SaveModes 保存模式枚举，决定如何处理和保存内容
        position : int = 0 进度条位置偏移量
        should_stop : Optional[Callable[[], bool]] 可选的停止检查函数，用于中断保存操作
    
    Returns:
        None
    
    Raises:
        ValueError: 如果提供了无效的保存模式
        KeyboardInterrupt: 如果保存操作被中断
        Exception: 如果在保存过程中发生其他错误
    """
    # 确保目录存在
    ensure_dir_exists(file_path.parent)
    # 使用提供的should_stop函数或默认函数
    should_stop_check = should_stop if should_stop else lambda: False
    try:
        if save_mode == SaveModes.TEXT or save_mode == SaveModes.ANIMATED_TEXT:
            output_text = content["output_text"]
            # 保存文本文件
            if should_stop_check():
                raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'])
            completed_event = threading.Event()
            process_thread = threading.Thread(target=no_value_file_save_progress,
                                              args=(file_path, completed_event, description, True, position, should_stop_check))
            process_thread.start()
            # 使用多进程保存文本文件，支持中断和异常传递
            run_with_interrupt_support(
                save_char_art_text,
                should_stop=should_stop_check,
                text=output_text,
                file_path=file_path,
                frame_index=content.get("index", None)
            )
            completed_event.set()
            while process_thread.is_alive():
                time.sleep(0.1)
                pass
            if content.get("index") is None:
                logger.info(f"文本已保存: {file_path}")
            else:
                logger.debug(f"帧{content['index']}文本已保存: {file_path}")
        elif save_mode == SaveModes.STATIC_IMAGE or save_mode == SaveModes.ANIMATED_IMAGE_TMP_FRAME or save_mode == SaveModes.VIDEO_TMP_FRAME:
            output_image = content["output_image"]
            completed_event = threading.Event()
            process_thread = threading.Thread(target=no_value_file_save_progress,
                                              args=(file_path, completed_event, description, True, position,
                                                    should_stop_check))
            process_thread.start()
            # 保存静态图像
            if should_stop_check():
                raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'])
            file_ext = get_file_extension(file_path)
            # 使用多进程保存静态图像，支持中断和异常传递
            run_with_interrupt_support(
                save_static_char_art_image,
                should_stop=should_stop_check,
                image=output_image,
                file_path=file_path,
                file_ext=file_ext,
                frame_index=content.get("index", None)
            )
            completed_event.set()
            while process_thread.is_alive():
                time.sleep(0.1)
                pass
            if content.get("index") is None:
                logger.info(f"图像已保存: {file_path}")
            else:
                logger.debug(f"帧{content['index']}图像已保存: {file_path}")
        elif save_mode == SaveModes.ANIMATED_IMAGE:
            # 保存动画图像
            if should_stop_check():
                raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'])
            completed_event = threading.Event()
            process_thread = threading.Thread(target=no_value_file_save_progress,
                                              args=(file_path, completed_event, description, True, position,
                                                    should_stop_check))
            process_thread.start()
            frames = content["frames"]
            durations = content["durations"]
            file_ext = get_file_extension(file_path)

            # 在保存前再次检查是否需要中断
            if should_stop_check():
                raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'])

            # 使用多进程保存动画图像，支持中断和异常传递
            run_with_interrupt_support(
                save_animated_char_art_image,
                should_stop=should_stop_check,
                frames=frames,
                durations=durations,
                file_path=file_path,
                file_ext=file_ext
            )
            completed_event.set()
            while process_thread.is_alive():
                time.sleep(0.1)
                pass
            logger.info(f"图像已保存: {file_path}")
        elif save_mode == SaveModes.AUDIO or save_mode == SaveModes.MERGE_ANIMATE or save_mode == SaveModes.VIDEO:
            # 使用到FFMPEG的操作（保存音频/将图像帧合成为动画图像/保存视频）
            if should_stop_check():
                raise KeyboardInterrupt(ERROR_MESSAGES['save_operation_interrupted'])

            cmd = content.get('cmd')
            log_path = content.get('log_path')

            # 使用多进程保存FFmpeg处理的文件，支持中断和异常传递
            run_with_interrupt_support(
                save_file_by_ffmpeg,
                should_stop=should_stop_check,
                cmd=cmd,
                log_path=log_path,
                save_mode=save_mode
            )

            if save_mode == SaveModes.AUDIO:
                logger.info(f"音频已提取: {file_path}")
            elif save_mode == SaveModes.MERGE_ANIMATE:
                logger.info(f"动图帧已合并: {file_path}")
            elif save_mode == SaveModes.VIDEO:
                logger.info(f"视频已保存: {file_path}")
        else:
            raise ValueError(ERROR_MESSAGES['invalid_save_mode'].format(save_mode.value, type(content).__name__))
    except Exception as e:
        logger.error(f"保存视频时出错: {e}")
        raise

