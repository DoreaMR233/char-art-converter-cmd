"""
颜色处理工具模块

该模块提供了一系列用于图像处理和字符画生成的颜色处理功能，
包括颜色转换、对比度增强、颜色增强等核心功能。

主要功能：
- RGB颜色到灰度的转换
- 颜色饱和度和亮度增强
- 对比度颜色计算
- 局部对比度增强
- 图像模式转换确保RGB格式
- 支持CPU和GPU加速处理

依赖：
- PIL (Pillow): 用于图像处理
- numpy: 用于数值计算
- torch: 用于GPU加速（可选）
- colorsys: 用于颜色空间转换（内置模块）
"""
from PIL import Image
from typing import Union, Tuple, Any, Optional

def rgb_to_gray(r: int, g: int, b: int) -> int:
    """
    将RGB颜色转换为灰度值
    
    使用标准的亮度公式：灰度值 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    该公式考虑了人眼对绿色敏感度最高，对蓝色敏感度最低的特性。
    
    Args:
        r : int 红色通道值 (0-255)
        g : int 绿色通道值 (0-255)
        b : int 蓝色通道值 (0-255)
    
    Returns:
        int: 灰度值 (0-255)
    """

    return int(0.2989 * r + 0.5870 * g + 0.1140 * b)

def enhance_color(r: Union[int, float, Any], g: Union[int, float, Any], b: Union[int, float, Any], 
                 saturation_factor: float = 1.5, brightness_factor: float = 1.2, 
                 torch: Optional[Any] = None) -> Union[Tuple[int, int, int], Tuple[Any, Any, Any]]:
    """
    增强RGB颜色的饱和度和亮度
    
    通过HSV颜色空间转换，提高颜色的饱和度和亮度，使颜色在字符画中更加醒目。
    支持单通道值和批量张量处理，同时兼容CPU和GPU环境。
    
    Args:
        r : Union[int, float, Any] 红色通道值 (0-255)
        g : Union[int, float, Any] 绿色通道值 (0-255)
        b : Union[int, float, Any] 蓝色通道值 (0-255)
        saturation_factor : float 饱和度增强因子，值越大饱和度越高
        brightness_factor : float 亮度增强因子，值越大亮度越高
        torch : Optional[Any] PyTorch模块引用，用于GPU加速处理
    
    Returns:
        Union[Tuple[int, int, int], Tuple[Any, Any, Any]]: 增强后的RGB颜色值元组，返回类型与输入类型一致
    """

    # 检查是否为张量输入且torch可用
    if torch is not None and (isinstance(r, torch.Tensor) or isinstance(g, torch.Tensor) or isinstance(b, torch.Tensor)):
        # 确保所有输入都是张量
        r = torch.tensor(r, dtype=torch.float32) if not isinstance(r, torch.Tensor) else r.float()
        g = torch.tensor(g, dtype=torch.float32) if not isinstance(g, torch.Tensor) else g.float()
        b = torch.tensor(b, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b.float()
        
        # 确保在同一设备上
        device = r.device
        g = g.to(device)
        b = b.to(device)
        
        # 归一化到0-1
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # GPU实现：使用与CPU版本完全相同的HSV颜色空间处理逻辑
        # 1. 计算RGB的最大值和最小值
        max_rgb = torch.max(torch.stack([r_norm, g_norm, b_norm], dim=-1), dim=-1).values
        min_rgb = torch.min(torch.stack([r_norm, g_norm, b_norm], dim=-1), dim=-1).values
        delta = max_rgb - min_rgb
        
        # 2. 计算色相(H)
        # 处理delta为0的情况（灰度）
        h = torch.zeros_like(max_rgb)
        
        # 红色占主导
        r_mask = (r_norm == max_rgb) & (delta != 0)
        h[r_mask] = ((g_norm[r_mask] - b_norm[r_mask]) / delta[r_mask]) % 6
        
        # 绿色占主导
        g_mask = (g_norm == max_rgb) & (delta != 0)
        h[g_mask] = ((b_norm[g_mask] - r_norm[g_mask]) / delta[g_mask]) + 2
        
        # 蓝色占主导
        b_mask = (b_norm == max_rgb) & (delta != 0)
        h[b_mask] = ((r_norm[b_mask] - g_norm[b_mask]) / delta[b_mask]) + 4
        
        # 转换为0-1范围
        h = h / 6.0
        
        # 3. 计算饱和度(S)
        s = torch.zeros_like(max_rgb)
        s[max_rgb != 0] = delta[max_rgb != 0] / max_rgb[max_rgb != 0]
        
        # 4. 计算亮度(V)
        v = max_rgb
        
        # 5. 增强饱和度和亮度，与CPU版本完全一致
        s = torch.clamp(s * saturation_factor, 0.0, 1.0)
        v = torch.clamp(v * brightness_factor, 0.3, 0.9)
        
        # 6. HSV转回RGB
        # 计算RGB分量
        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c
        
        # 初始化RGB通道
        r_new = torch.zeros_like(h)
        g_new = torch.zeros_like(h)
        b_new = torch.zeros_like(h)
        
        # 根据色相确定RGB值
        mask1 = (h >= 0) & (h < 1/6)
        r_new[mask1] = c[mask1]
        g_new[mask1] = x[mask1]
        b_new[mask1] = 0
        
        mask2 = (h >= 1/6) & (h < 2/6)
        r_new[mask2] = x[mask2]
        g_new[mask2] = c[mask2]
        b_new[mask2] = 0
        
        mask3 = (h >= 2/6) & (h < 3/6)
        r_new[mask3] = 0
        g_new[mask3] = c[mask3]
        b_new[mask3] = x[mask3]
        
        mask4 = (h >= 3/6) & (h < 4/6)
        r_new[mask4] = 0
        g_new[mask4] = x[mask4]
        b_new[mask4] = c[mask4]
        
        mask5 = (h >= 4/6) & (h < 5/6)
        r_new[mask5] = x[mask5]
        g_new[mask5] = 0
        b_new[mask5] = c[mask5]
        
        mask6 = (h >= 5/6) & (h < 1)
        r_new[mask6] = c[mask6]
        g_new[mask6] = 0
        b_new[mask6] = x[mask6]
        
        # 添加偏移量m
        r_new = r_new + m
        g_new = g_new + m
        b_new = b_new + m
        
        # 转换回0-255范围并转换为整数类型
        r_new = torch.clamp(r_new * 255, 0, 255).to(torch.uint8)
        g_new = torch.clamp(g_new * 255, 0, 255).to(torch.uint8)
        b_new = torch.clamp(b_new * 255, 0, 255).to(torch.uint8)
        
        return r_new, g_new, b_new
    else:
        # 原始数值输入处理逻辑
        import colorsys
        # 转换为HSV
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # 增强饱和度和亮度
        s = min(1.0, s * saturation_factor)
        v = max(0.3, min(0.9, v * brightness_factor))
        
        # 转换回RGB
        r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s, v)
        
        return int(r_new * 255), int(g_new * 255), int(b_new * 255)

def get_contrast_color(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """
    根据背景颜色获取高对比度的前景颜色
    
    使用亮度计算来确定合适的前景色，使文本在背景上清晰可见
    
    Args:
        r : int 背景颜色的红色分量 (0-255)
        g : int 背景颜色的绿色分量 (0-255)
        b : int 背景颜色的蓝色分量 (0-255)
        
    Returns:
        Tuple[int, int, int]: 高对比度前景色的RGB分量
    """
    # 计算亮度
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    
    if brightness > 128:
        # 背景较亮，使用深色前景
        factor = 0.3
    else:
        # 背景较暗，使用浅色前景
        factor = 1.7
    
    new_r = min(255, max(0, int(r * factor)))
    new_g = min(255, max(0, int(g * factor)))
    new_b = min(255, max(0, int(b * factor)))
    
    return new_r, new_g, new_b

def ensure_rgb_mode(image: Image.Image) -> Image.Image:
    """
    确保图像为RGB模式
    
    检查图像模式，如果不是RGB模式则转换
    
    Args:
        image : Image.Image PIL图像对象
        
    Returns:
        Image.Image: RGB模式的图像对象
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image
