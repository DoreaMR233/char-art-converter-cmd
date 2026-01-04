"""
字符艺术生成工具模块

该模块提供了字符画生成的核心功能，包括图像尺寸计算、
图像重采样、像素转字符、GPU加速处理等功能。

主要功能：
- 计算适合字符画的图像尺寸
- 重采样图像以适应字符显示
- 将像素值转换为对应字符
- GPU加速的字符转换
- 生成字符画文本
- 创建字符画图像
- 支持CPU和GPU处理

依赖：
- numpy: 用于数值计算
- PIL (Pillow): 用于图像处理
- torch: 用于GPU加速（可选）
"""
import logging
from typing import List, Optional, Any, Tuple, Union, Callable

import numpy as np
from PIL import Image, ImageDraw

from .color_utils import rgb_to_gray, ensure_rgb_mode, enhance_color, get_contrast_color
from .font_utils import calculate_char_size
from .progress_bar_utils import show_project_status_progress
from ..configs.message_config import ERROR_MESSAGES, WARNING_MESSAGES
from ..enums.color_modes import ColorModes

# 初始化日志器
logger = logging.getLogger(__name__)

def calculate_resized_dimensions(limit_size: Optional[List[int]], image: Image.Image, frame_index: Optional[int] = None, is_show: bool = True, font_size: Optional[int] = None) -> Tuple[int, int]:
    """
    计算调整后的图像尺寸，保持宽高比

    根据原始图像的尺寸、字体大小和限制条件，计算适合字符画生成的最佳尺寸，
    同时保持图像的原始宽高比。

    Args:
        limit_size: Optional[List[int]] 限制尺寸的列表 [width, height]，如果为None则不限制，空列表则使用默认限制
        image: Image.Image 原始图像对象
        frame_index: Optional[int] 帧索引，用于日志记录（处理动画时使用）
        is_show: bool 是否显示日志信息
        font_size: Optional[int] 字体大小，用于计算默认缩放比例，默认为12

    Returns:
        Tuple[int, int]: 调整后的宽度和高度 (width, height)

    Raises:
        TypeError: 当输入参数类型不符合要求时抛出
        ValueError: 当输入参数值无效时抛出
    """

    original_width, original_height = image.size
    if is_show:
        if frame_index is None:
            logger.info(f"原图尺寸: {original_width}x{original_height}")
        else:
            logger.debug(f"帧{frame_index}原图尺寸: {original_width}x{original_height}")
    if limit_size is None:
        # 不限制大小
        max_width = original_width
        max_height = original_height
    elif isinstance(limit_size, list) and len(limit_size) == 0:
        # 使用默认限制，基于字体大小

        max_width = original_width // ( font_size // 2 if font_size is not None else 4)
        max_height = original_height // ( font_size // 2 if font_size is not None else 6)
    else:
        # 使用用户指定的尺寸
        user_width, user_height = limit_size
        # 计算原图适合的最大字符尺寸
        max_char_width = max(1, original_width)
        max_char_height = max(1, original_height)
            
        if user_width > max_char_width or user_height > max_char_height:
            logger.warning(WARNING_MESSAGES['image_size_exceeds_limit'].format(user_width, user_height, max_char_width, max_char_height))
            max_width = max_char_width
            max_height = max_char_height
        else:
            max_width = user_width
            max_height = user_height
        
    # 保持宽高比
    aspect_ratio = original_width / original_height
        
    if max_width / max_height > aspect_ratio:
        # 高度是限制因素
        new_height = float(max_height)
        new_width = new_height * aspect_ratio
    elif max_width / max_height < aspect_ratio:
        # 宽度是限制因素
        new_width = float(max_width)
        new_height = new_width / aspect_ratio
    else:
        # 宽高比一致则保持不变
        new_width = float(max_width)
        new_height = float(max_height)

    # 确保最小尺寸
    new_width = max(new_width, 1)
    new_height = max(new_height, 1)
    if is_show:
        if frame_index is None:
            logger.info(f"调整后原图尺寸: {int(new_width)}x{int(new_height)}")
        else:
            logger.debug(f"原图帧{frame_index}调整后尺寸: {int(new_width)}x{int(new_height)}")
    return int(new_width), int(new_height)

def resize_image_for_chars(limit_size: Optional[List[int]], image: Image.Image, frame_index: Optional[int] = None, is_show: bool = True, font_size: Optional[int] = None) -> Image.Image:
    """
    调整图像大小以适合字符画生成
    
    使用calculate_resized_dimensions计算最佳尺寸，然后调整图像大小，
    采用LANCZOS重采样方法以获得最佳质量。
    
    Args:
        limit_size: Optional[List[int]] 限制尺寸的列表 [width, height]，如果为None则不限制，空列表则使用默认限制
        image: Image.Image 原始图像对象
        frame_index: Optional[int] 帧索引，用于日志记录（处理动画时使用）
        is_show: bool 是否显示日志信息
        font_size: Optional[int] 字体大小，用于计算默认缩放比例，默认为12
            
    Returns:
        Image.Image: 调整大小后的图像对象
    
    Raises:
        TypeError: 当输入参数类型不符合要求时抛出
        ValueError: 当输入参数值无效时抛出
        Exception: 当图像调整大小失败时抛出
    """

    new_width, new_height = calculate_resized_dimensions(limit_size,image, frame_index,is_show,font_size)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def pixel_to_char(pixel: Union[int, float, Tuple[int, ...], Tuple[Tuple[int, ...], ...], None],
                  char_set: str, char_count: int) -> str:
    """
    将像素值转换为对应的字符
    
    根据像素的灰度值或RGB值，选择最适合表示该像素的字符。
    较暗的像素对应较密集的字符，较亮的像素对应较稀疏的字符。
    使用非线性映射算法确保字符集的均匀分布和更大的字符种类差异。
    
    Args:
        pixel: Union[int, float, Tuple[int, ...], None] 像素值，可以是灰度值（int/float）或RGB值（元组），或None
        char_set: str 字符集字符串，包含用于表示不同灰度级别的字符
        char_count: int 字符集的字符数量
            
    Returns:
        str: 与像素对应的字符
    
    Raises:
        TypeError: 当输入参数类型不符合要求时抛出
        IndexError: 当字符索引超出字符集范围时抛出
    """

    if pixel is None:
        return ' '  # 如果像素为空，返回空格

    if isinstance(pixel, tuple):
        # RGB像素，转换为灰度
        if len(pixel) >= 3:
            # 确保传入的是整数类型
            r = int(pixel[0]) if isinstance(pixel[0], (int, float)) else 0
            g = int(pixel[1]) if isinstance(pixel[1], (int, float)) else 0
            b = int(pixel[2]) if isinstance(pixel[2], (int, float)) else 0
            gray = rgb_to_gray(r, g, b)
        else:
            gray = 0  # 处理不完整的元组
    else:
        # 已经是灰度值，确保是整数类型
        gray = int(pixel) if isinstance(pixel, (int, float)) else 0

    # 使用更激进的对比度增强策略
    # 1. 使用sigmoid-like函数进行非线性映射，确保灰度值范围被充分利用
    # 首先将灰度值归一化到[-1, 1]范围，然后应用sigmoid-like函数
    normalized_gray = (gray - 127.5) / 127.5  # 归一化到[-1, 1]
    
    # 2. 使用反正切函数进行非线性映射，它比sigmoid更适合我们的需求
    # 因为它在中间区域有更好的线性度，而在两端有更强的压缩效果
    # 调整参数以增强对比度，使中间区域的差异更明显
    contrast_factor = 4.0  # 更高的对比度因子
    import math
    mapped_gray = 0.5 * (math.atan(contrast_factor * normalized_gray) / (math.pi/2) + 1.0)  # 映射回[0, 1]
    
    # 3. 直接将映射后的灰度值均匀分配到字符集中的每个字符
    # 这样可以确保每个字符都有机会被使用
    # 使用char_count而不是char_count-1可以确保覆盖整个范围
    char_index = int(mapped_gray * char_count)
    
    # 4. 反向映射：较暗的像素对应较密集的字符（索引较大的字符）
    char_index = char_count - 1 - char_index
    
    # 确保索引在有效范围内
    char_index = max(0, min(char_index, char_count - 1))

    return char_set[char_index]

def pixel_to_char_gpu(pixels_gpu: Any, char_count: int, torch: Any, device: Any) -> Any:
    """
    使用GPU将像素值批量转换为字符索引
    
    该函数在GPU上执行批量像素到字符索引的转换，是GPU加速图像处理的核心组件。
    支持RGB到灰度的转换、非线性映射增强，并将像素值映射到对应的字符索引。
    使用与CPU版本相同的非线性映射算法，确保更大的字符种类差异。
    
    Args:
        pixels_gpu: Any GPU上的像素张量，可以是灰度或RGB格式
        char_count: int 字符集的字符数量
        torch: Any PyTorch模块，用于GPU计算
        device: Any 计算设备（CPU或GPU）
    
    Returns:
        Any: GPU上的字符索引张量，每个像素对应一个字符索引
    
    Raises:
        TypeError: 当输入参数类型不符合要求时抛出
        RuntimeError: 当GPU计算失败时抛出
        ValueError: 当输入参数值无效时抛出
    """

    # 确保输入是float32类型
    pixels_gpu = pixels_gpu.to(torch.float32)

    # 如果是RGB图像，转换为灰度
    if pixels_gpu.ndim == 3 and pixels_gpu.shape[2] == 3:
        # RGB转灰度的权重
        weight_rgb = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=device)
        # 执行矩阵乘法进行RGB到灰度的转换
        gray_gpu = torch.matmul(pixels_gpu, weight_rgb)
    else:
        # 已经是灰度图像
        gray_gpu = pixels_gpu.squeeze()

    # 使用与CPU版本相同的非线性映射算法
    with torch.no_grad():
        # 1. 将灰度值归一化到[-1, 1]范围
        normalized_gray = (gray_gpu - 127.5) / 127.5
        
        # 2. 使用反正切函数进行非线性映射
        contrast_factor = 4.0  # 与CPU版本相同的对比度因子
        mapped_gray = 0.5 * (torch.atan(contrast_factor * normalized_gray) / (torch.pi / 2) + 1.0)
        
        # 3. 直接将映射后的灰度值均匀分配到字符集中的每个字符
        char_indices = (mapped_gray * char_count).to(torch.int32)
        
        # 4. 反向映射：较暗的像素对应较密集的字符（索引较大的字符）
        char_indices = char_count - 1 - char_indices
        
        # 确保索引在有效范围内
        char_indices = torch.clamp(char_indices, 0, char_count - 1)

    return char_indices

def image_to_char_text(resized_image: Image.Image,use_gpu: bool,torch: Any,device: Any,char_set: str,char_count: int, should_stop: Optional[Callable[[], bool]] = None) -> str:
    """
    将调整大小后的图像转换为字符文本
    
    该方法是核心转换函数，可以使用CPU或GPU加速将图像转换为字符文本。
    如果GPU处理失败，会自动回退到CPU处理模式。
    
    Args:
        resized_image: Image.Image 调整大小后的图像对象
        use_gpu: bool 是否使用GPU加速处理
        torch: Any PyTorch模块，用于GPU加速
        device: Any 计算设备（CPU或GPU）
        char_set: str 字符集字符串
        char_count: int 字符集的字符数量
        should_stop: Optional[Callable[[], bool]] 用于检查是否应停止处理的回调函数
            
    Returns:
        str: 由字符组成的文本，每行代表图像的一行像素
    """

    # 初始化字符行列表
    char_lines: List[str] = []
    
    # 调整图像大小
    width, height = resized_image.size
    
    # 转换为RGB模式以便处理
    resized_image = ensure_rgb_mode(resized_image)
    
    # 检查是否应该使用GPU加速
    logger.debug(f"字符转换 - 图像大小: {width}x{height}, 使用GPU: {use_gpu}")
    
    if use_gpu:
        try:
            # 使用GPU加速处理
            # 创建进度条，显示GPU处理进度
            with show_project_status_progress(3, description="GPU生成字符文本") as pbar:
                # 检查是否应该停止
                if should_stop and should_stop():
                    raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                    
                # 步骤1：GPU计算字符索引
                pbar.set_description("GPU生成字符文本 [GPU计算]")
                # 转换图像为numpy数组
                image_array = np.array(resized_image)
                
                # 将numpy数组转移到GPU
                gpu_array = torch.tensor(image_array, dtype=torch.float32, device=device)
                
                # 使用pixel_to_char_gpu函数进行像素到字符索引的转换
                char_indices = pixel_to_char_gpu(gpu_array, char_count, torch, device)
                
                # 将结果从GPU转移回CPU
                char_indices_cpu = char_indices.cpu().numpy()
                
                # 更新进度
                pbar.update(1)
                
                # 检查是否应该停止
                if should_stop and should_stop():
                    raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                
                # 步骤2：映射字符索引
                pbar.set_description("GPU生成字符文本 [映射字符]")
                # 映射字符索引到实际字符
                char_array = np.vectorize(lambda idx: char_set[idx])(char_indices_cpu)
                
                # 更新进度
                pbar.update(1)
                
                # 检查是否应该停止
                if should_stop and should_stop():
                    raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                
                # 步骤3：构造最终文本
                pbar.set_description("GPU生成字符文本 [构造文本]")
                # 构造最终文本
                char_lines = [''.join(row) for row in char_array]
                
                # 更新进度
                pbar.update(1)
            
            logger.debug("使用GPU成功完成字符转换")
            return "\n".join(char_lines)
            
        except (ImportError, RuntimeError) as e:
            # 出错时回退到CPU处理
            logger.warning(WARNING_MESSAGES['gpu_processing_failed'].format(e), exc_info=True)
    
    # CPU处理模式
    # 使用进度条显示转换进度
    with show_project_status_progress(height, description="CPU生成字符文本") as pbar:
        for y in range(height):
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                
            line = ""
            for x in range(width):
                pixel = resized_image.getpixel((x, y))
                char = pixel_to_char(pixel,char_set,char_count)
                line += char

            char_lines.append(line)
            pbar.update(1)
    
    return "\n".join(char_lines)

def create_char_image(resized_image: Image.Image, use_gpu: bool, char_set: str, char_count: int, 
                     color_mode: ColorModes, font: Any, torch: Any, device: Any,
                     frame_index: Optional[int] = None, position: int = 0, 
                     should_stop: Optional[Callable[[], bool]] = None) -> Tuple[Image.Image, str]:
    """
    创建字符画图像和对应的字符文本
    
    根据调整后的图像，生成字符画图像和对应的字符文本字符串。
    支持多种颜色模式和GPU加速处理。
    
    Args:
        resized_image: Image.Image 调整大小后的图像对象
        use_gpu: bool 是否使用GPU加速处理
        char_set: str 字符集字符串
        char_count: int 字符集的字符数量
        color_mode: ColorModes 颜色模式枚举值，决定如何处理字符颜色
        font: Any 用于绘制字符的字体对象
        torch: Any PyTorch模块，用于GPU加速
        device: Any 计算设备（CPU或GPU）
        frame_index: Optional[int] 帧索引，用于日志记录（处理动画时使用）
        position: int 进度条的位置
        should_stop: Optional[Callable[[], bool]] 用于检查是否应停止处理的回调函数
            
    Returns:
        Tuple[Image.Image, str]: 生成的字符画图像和对应的字符文本字符串
    """

    # 调整图像大小
    width, height = resized_image.size

    # 转换为RGB模式
    resized_image = ensure_rgb_mode(resized_image)

    # 计算字符尺寸
    char_width, char_height = calculate_char_size(font)

    # 计算输出图像尺寸
    output_width = int(width * char_width)
    output_height = int(height * char_height)

    # 创建输出图像
    output_image = Image.new('RGB', (output_width, output_height), 'white')
    draw = ImageDraw.Draw(output_image)

    if frame_index is None:
        logger.info(f"字符画图像尺寸: {output_image.width}x{output_image.height}")
    else:
        logger.debug(f"帧{frame_index}字符画图像尺寸: {output_image.width}x{output_image.height}")

    # 创建输出字符串数组
    char_lines: List[str] = []

    # 检查是否使用GPU加速
    logger.debug(f"字符图像生成 - 图像大小: {width}x{height}, 使用GPU: {use_gpu}")

    if use_gpu:
        # GPU加速处理
        try:
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                
            process_image_gpu(resized_image, draw, char_lines, width, height, char_width, char_height, torch, device, char_set, char_count, color_mode, font, frame_index, position, should_stop)
        except Exception as e:
            logger.warning(WARNING_MESSAGES['gpu_processing_failed'].format(e), exc_info=True)
            process_image_cpu(resized_image, draw, char_lines, width, height, char_width, char_height, font, char_set, char_count, color_mode, frame_index, position, should_stop)
    else:
        # CPU处理
        # 检查是否应该停止
        if should_stop and should_stop():
            raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            
        process_image_cpu(resized_image, draw, char_lines, width, height, char_width, char_height, font, char_set, char_count, color_mode, frame_index, position, should_stop)

    char_line = "\n".join(char_lines)
    if frame_index is None:
        logger.info(f"字符画图像与文本已生成")
    else:
        logger.debug(f"帧{frame_index}字符画图像与文本已生成")

    return output_image,char_line

def process_image_gpu(resized_frame: Image.Image, draw: Any, char_lines: List[str], width: int, height: int,
                      char_width: int, char_height: int, torch: Any, device: Any,
                      char_set: str, char_count: int, color_mode: ColorModes, font: Any,
                      frame_index: Optional[int], position: int = 0,
                      should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    使用GPU处理图像，生成字符画和字符文本
    
    该方法使用PyTorch进行GPU加速处理，将图像转换为字符画图像和字符文本。
    支持灰度、彩色和背景色三种颜色模式。
    
    Args:
        resized_frame: Image.Image 调整大小后的图像对象
        draw: Any 用于绘制字符的ImageDraw对象
        char_lines: List[str] 存储生成的字符行的列表
        width: int 图像宽度
        height: int 图像高度
        char_width: int 字符宽度
        char_height: int 字符高度
        torch: Any PyTorch模块
        device: Any 计算设备（GPU）
        char_set: str 字符集字符串
        char_count: int 字符集的字符数量
        color_mode: ColorModes 颜色模式枚举值
        font: Any 用于绘制字符的字体对象
        frame_index: Optional[int] 帧索引（用于动画处理）
        position: int 进度条位置
        should_stop: Optional[Callable[[], bool]] 用于检查是否应停止处理的回调函数
    """

    # 根据是否有帧编号使用不同的描述文本
    image_desc = f"帧 {frame_index}" if frame_index is not None  else "静态图像"
    # 预创建字符映射字典
    char_index_to_char = {i: char for i, char in enumerate(char_set)}

    # 计算总处理量，用于进度条
    total_pixels = width * height
    
    # 创建主进度条
    with show_project_status_progress(total=total_pixels+4, description=f"{image_desc} GPU生成",
                                        position=position) as pbar:
        try:
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            # 1. 数据准备
            pbar.set_description(f"{image_desc} GPU生成 [数据准备]")
            # 直接转换PIL图像为PyTorch张量并移至GPU
            pixels_gpu = torch.from_numpy(np.array(resized_frame, dtype=np.uint8)).to(device, dtype=torch.float32)
            torch.cuda.synchronize()

            # 更新进度条
            pbar.update(1)
            # 2. 批量GPU计算字符索引和颜色
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            pbar.set_description(f"{image_desc} GPU生成 [批量GPU计算]")
            char_indices_gpu = pixel_to_char_gpu(pixels_gpu, char_count, torch, device)

            text_color_mask = None
            bg_colors_gpu = None
            
            # 对于灰度模式，我们不需要颜色信息
            if color_mode == ColorModes.GRAYSCALE:
                # 返回一个标记，表示所有像素都是黑色
                text_color_mask = torch.zeros_like(pixels_gpu[..., 0] if pixels_gpu.ndim == 3 else pixels_gpu, dtype=torch.int32)
                bg_colors_gpu = None

            # 对于彩色模式
            elif color_mode == ColorModes.COLOR:
                if pixels_gpu.ndim == 3 and pixels_gpu.shape[2] >= 3:
                    # 直接在GPU上使用enhance_color函数进行颜色增强
                    # 分离RGB通道
                    r_gpu = pixels_gpu[..., 0]
                    g_gpu = pixels_gpu[..., 1]
                    b_gpu = pixels_gpu[..., 2]
                    
                    # 调用支持GPU的enhance_color函数，传入torch模块以启用GPU计算
                    r_enhanced, g_enhanced, b_enhanced = enhance_color(r_gpu, g_gpu, b_gpu, torch=torch)
                    
                    # 重新组合RGB通道
                    text_color_mask = torch.stack([r_enhanced, g_enhanced, b_enhanced], dim=-1)
                    bg_colors_gpu = None
                else:
                    # 不是RGB图像，返回白色
                    text_color_mask = None

            # 对于背景色模式，我们也需要计算对比度颜色
            elif color_mode == ColorModes.COLOR_BACKGROUND:
                if pixels_gpu.ndim == 3 and pixels_gpu.shape[2] >= 3:
                    # 获取背景色
                    bg_colors_gpu = pixels_gpu.to(torch.uint8)

                    # 计算对比度颜色
                    # 基于亮度计算: Y = 0.299R + 0.587G + 0.114B
                    luminance = 0.299 * pixels_gpu[..., 0] + 0.587 * pixels_gpu[..., 1] + 0.114 * pixels_gpu[..., 2]
                    
                    # 计算亮度因子，与CPU版本保持一致
                    brightness_factor = luminance / 255.0
                    
                    # 基于亮度因子确定文本颜色：
                    # - 亮度因子 > 0.5 使用黑色文本（值为0）
                    # - 亮度因子 <= 0.5 使用白色文本（值为1）
                    text_color_mask = (brightness_factor <= 0.5).to(torch.int32)

                else:
                    # 不是RGB图像，返回白色
                    text_color_mask = torch.ones_like(pixels_gpu[..., 0] if pixels_gpu.ndim == 3 else pixels_gpu,
                                           dtype=torch.int32)
                    bg_colors_gpu = None

            torch.cuda.synchronize()
            # 更新进度条
            pbar.update(1)
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            # 3. 准备用于CPU绘制的数据（只传输必要数据）
            pbar.set_description(f"{image_desc} GPU生成 [准备CPU数据]")
            # 将字符索引传输到CPU
            char_indices = char_indices_gpu.cpu().numpy()

            # 定义需要在后面使用的变量
            color_data = None
            bg_colors = None

            # 根据颜色模式传输颜色数据
            if color_mode == ColorModes.GRAYSCALE:
                # 灰度模式，所有文本都是黑色
                pass
            elif color_mode == ColorModes.COLOR and text_color_mask is not None:
                # 彩色模式，获取增强后的颜色
                color_data = text_color_mask.cpu().numpy()
            elif color_mode == ColorModes.COLOR_BACKGROUND:
                # 背景色模式，获取背景色和文本颜色掩码
                if bg_colors_gpu is not None:
                    bg_colors = bg_colors_gpu.cpu().numpy()

            torch.cuda.synchronize()
            # 更新进度条
            pbar.update(1)
            # 4. 处理单个像素
            pbar.set_description(f"{image_desc} GPU生成 [处理单个像素]")
            # 使用numpy向量化操作预处理坐标
            x_coords = np.arange(width)
            y_coords = np.arange(height)

            # 创建网格以避免在循环中计算
            xx, yy = np.meshgrid(x_coords, y_coords)
            char_xs = (xx * char_width).astype(int)
            char_ys = (yy * char_height).astype(int)

            # 批量绘制字符
            for y_idx in range(height):
                line = ""
                for x_idx in range(width):
                    # 检查是否应该停止
                    if should_stop and should_stop():
                        raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
                    # 获取字符 - 使用预创建的字典
                    char_idx = int(char_indices[y_idx, x_idx])
                    char = char_index_to_char[char_idx]  # 优化: 使用预创建的字典
                    line += char
                    logger.debug(f"{image_desc}-({x_idx},{y_idx}): 生成字符")
                    # 获取预计算的坐标
                    char_x = char_xs[y_idx, x_idx]
                    char_y = char_ys[y_idx, x_idx]

                    # 根据颜色模式设置颜色
                    text_color: Union[str, Tuple[int, int, int]] = 'white'  # 设置默认值，确保变量始终被初始化

                    if color_mode == ColorModes.GRAYSCALE:
                        text_color = 'black'
                    elif color_mode == ColorModes.COLOR:
                        # 直接检查变量而不是使用locals()
                        if color_data is not None and len(color_data.shape) >= 3:
                            # 确保颜色值为整数
                            r = int(color_data[y_idx, x_idx, 0])
                            g = int(color_data[y_idx, x_idx, 1])
                            b = int(color_data[y_idx, x_idx, 2])
                            text_color = (r, g, b)
                        else:
                            text_color = 'white'
                    elif color_mode == ColorModes.COLOR_BACKGROUND:
                        # 直接检查变量而不是使用locals()
                        if bg_colors is not None and len(bg_colors.shape) >= 3:
                            # 确保颜色值为整数
                            bg_r = int(bg_colors[y_idx, x_idx, 0])
                            bg_g = int(bg_colors[y_idx, x_idx, 1])
                            bg_b = int(bg_colors[y_idx, x_idx, 2])
                            bg_color = (bg_r, bg_g, bg_b)
                            # 绘制背景
                            draw.rectangle(
                                [char_x, char_y, char_x + int(char_width), char_y + int(char_height)],
                                fill=bg_color
                            )
                            # 确定文本颜色
                            text_color = get_contrast_color(bg_r, bg_g, bg_b)

                    # 绘制字符
                    draw.text((char_x, char_y), char, font=font, fill=text_color)
                    logger.debug(f"{image_desc}-({x_idx},{y_idx}): 绘制字符")
                    
                    # 更新进度条
                    pbar.update(1)

                char_lines.append(line)
            # 更新进度条
            pbar.update(1)
            # 更新进度条状态为完成
            pbar.set_description(f"{image_desc} GPU生成 [完成]")
        except Exception as e:
            logger.error(ERROR_MESSAGES['gpu_processing_image_failed'].format(image_desc, e), exc_info=True)
            raise
        finally:
            # 清理GPU内存
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

def process_image_cpu(resized_image: Image.Image, draw: Any, char_lines: List[str], width: int, height: int,
                      char_width: int, char_height: int, font: Any, char_set: str, char_count: int, 
                      color_mode: ColorModes, frame_index: Optional[int], position: int = 0,
                      should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    使用CPU处理图像，生成字符画和字符文本
    
    该方法在CPU上执行，将图像转换为字符画图像和字符文本。
    支持灰度、彩色和背景色三种颜色模式，是GPU处理的备选方案。
    
    Args:
        resized_image: Image.Image 调整大小后的图像对象
        draw: Any 用于绘制字符的ImageDraw对象
        char_lines: List[str] 存储生成的字符行的列表
        width: int 图像宽度
        height: int 图像高度
        char_width: int 字符宽度
        char_height: int 字符高度
        font: Any 用于绘制字符的字体对象
        char_set: str 字符集字符串
        char_count: int 字符集的字符数量
        color_mode: ColorModes 颜色模式枚举值
        frame_index: Optional[int] 帧索引（用于动画处理）
        position: int 进度条位置
        should_stop: Optional[Callable[[], bool]] 用于检查是否应停止处理的回调函数
    """

    # 根据是否有帧编号使用不同的描述文本
    image_desc = f"帧 {frame_index}" if frame_index is not None  else "静态图像"

    total_pixels = width * height

    # 创建进度条
    with show_project_status_progress(total=total_pixels, description=f"{image_desc} CPU生成",
                                       position=position) as pbar:
        for y in range(height):
            # 检查是否应该停止
            if should_stop and should_stop():
                raise KeyboardInterrupt(ERROR_MESSAGES['processing_interrupted'])
            line = ""
            for x in range(width):
                pixel = resized_image.getpixel((x, y))
                # 将像素转换为字符
                char = pixel_to_char(pixel,char_set,char_count)
                line += char
                logger.debug(f"{image_desc}-({x},{y}): 生成字符")
                # 计算字符位置
                char_x = x * char_width
                char_y = y * char_height

                # 根据颜色模式设置颜色
                text_color: Union[str, Tuple[int, int, int]] = 'white'  # 设置默认值，确保变量始终被初始化
                if color_mode == ColorModes.GRAYSCALE:
                    text_color =  'black'
                elif color_mode == ColorModes.COLOR:
                    if isinstance(pixel, tuple) and len(pixel) >= 3:
                        text_color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
                    else:
                        text_color = 'white'
                elif color_mode == ColorModes.COLOR_BACKGROUND:  # colorBackground
                    if isinstance(pixel, tuple) and len(pixel) >= 3:
                        bg_color = pixel
                        text_color = get_contrast_color(pixel[0], pixel[1], pixel[2])

                        # 绘制背景（如果提供了绘图对象）
                        if draw is not None:
                            draw.rectangle(
                                [char_x, char_y, char_x + char_width, char_y + char_height],
                                fill=bg_color
                            )
                    else:
                        text_color =  'white'

                # 绘制字符
                draw.text((char_x, char_y), char, font=font, fill=text_color)
                logger.debug(f"{image_desc}-({x},{y}): 绘制字符")
                
                # 更新进度条
                pbar.update(1)

            char_lines.append(line)

