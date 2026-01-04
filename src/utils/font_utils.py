
"""
字体工具模块

该模块提供了一系列用于字体管理和处理的工具函数，
包括字体加载、等宽字体检测、字体缓存管理等功能，
是字符画转换过程中字体处理的核心组件。

主要功能：
- 系统字体扫描和获取
- 等宽字体自动检测
- 字体加载和缓存管理
- 字体大小计算
- 跨平台字体支持
- 自动回退字体机制

依赖：
- PIL (Pillow): 用于字体加载和处理
- fontTools: 用于字体分析和等宽检测
- platform: 用于系统类型判断
- pathlib: 用于路径操作

"""
import logging
import platform
from pathlib import Path
from typing import List, Optional, Dict, Union

from PIL import ImageFont
from fontTools.ttLib import TTFont  # type: ignore
from ..configs.message_config import WARNING_MESSAGES

# 初始化日志器
logger = logging.getLogger(__name__)

def _get_font_name(font: 'TTFont') -> Optional[str]:
    """
    从TTFont对象中提取字体名称
    
    尝试从TTFont对象的name表中获取Font Family Name
    
    Args:
        font: TTFont TTFont对象
        
    Returns:
        Optional[str]: 字体名称字符串，如果无法获取则返回None
        
    Raises:
        - KeyError: 当字体文件缺少必要表时
        - AttributeError: 当字体文件结构异常时
        - UnicodeDecodeError: 当字体名称解码失败时
    """
    try:
        name_table = font['name']

        # 尝试获取英文字体名称
        for record in name_table.names:
            if record.nameID == 1:  # Font Family Name
                if record.platformID == 3:  # Microsoft
                    return record.toUnicode()

    except (KeyError, AttributeError, UnicodeDecodeError):
        pass

    return None


def _is_monospace_font(font: 'TTFont') -> bool:
    """
    判断字体是否为等宽字体
    
    通过检查常见字符的宽度是否相同来确定字体是否为等宽字体
    
    Args:
        font: TTFont TTFont对象
        
    Returns:
        bool: 如果是等宽字体则返回True，否则返回False
        
    Raises:
        - KeyError: 当字体文件缺少必要表时
        - AttributeError: 当字体文件结构异常时
        - TypeError: 当字体数据类型不兼容时
    """
    try:
        # 获取字符宽度表
        hmtx = font['hmtx']

        # 检查常见字符的宽度是否相同
        test_chars = ['A', 'i', 'M', 'l', '1', '0']
        widths = []

        for char in test_chars:
            try:
                char_code = ord(char)
                if char_code in font.getGlyphSet():
                    glyph_name = font.getBestCmap()[char_code]
                    width = hmtx[glyph_name][0]
                    widths.append(width)
            except (KeyError, IndexError):
                continue

        # 如果所有字符宽度相同，则为等宽字体
        return len(set(widths)) == 1 and len(widths) > 3

    except (KeyError, AttributeError, TypeError):
        return False


def _find_font_file(font_name: str) -> Optional[Path]:
    """
    根据字体名称查找字体文件
    
    通过预定义的映射关系在系统字体目录中查找指定名称的字体文件
    
    Args:
        font_name: str 字体名称
        
    Returns:
        Optional[Path]: 字体文件路径，如果未找到则返回None
    """
    # 字体名称到文件名的映射
    font_mapping = {
        'Consolas': ['consola.ttf', 'consolab.ttf', 'consolai.ttf', 'consolaz.ttf'],
        'Courier New': ['cour.ttf', 'courbd.ttf', 'couri.ttf', 'courbi.ttf'],
        'Lucida Console': ['lucon.ttf'],
        'Monaco': ['monaco.ttf'],
        'DejaVu Sans Mono': ['DejaVuSansMono.ttf', 'DejaVuSansMono_0.ttf'],
        'Liberation Mono': ['LiberationMono-Regular.ttf'],
        'Ubuntu Mono': ['UbuntuMono-R.ttf'],
        'Source Code Pro': ['SourceCodePro-Regular.ttf'],
        'Fira Code': ['FiraCode-Regular.ttf'],
        'JetBrains Mono': ['JetBrainsMono-Regular.ttf'],
        'Cascadia Code': ['CascadiaCode.ttf']
    }

    if font_name not in font_mapping:
        return None

    # 获取Windows字体目录
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
    ]

    # 搜索字体文件
    for font_dir in font_dirs:
        if font_dir.exists():
            for filename in font_mapping[font_name]:
                font_path = font_dir / filename
                if font_path.exists():
                    return font_path

    return None


class FontManager:
    """
    字体管理器类，负责系统字体的查找、加载和缓存管理
    
    提供字体检测、加载和缓存功能，支持系统字体扫描、等宽字体识别和字体缓存管理。
    该类实现了高效的字体查找和加载机制，确保在字符画生成过程中能够快速获取所需字体。
    
    Args:
        self.logger : logging.Logger 日志记录器，用于记录字体管理过程中的信息和错误
        self._font_cache : Dict[str, ImageFont.FreeTypeFont] 字体缓存字典，键为字体名称和大小的组合，值为加载的字体对象
        self._system_fonts : Optional[List[Path]] 系统字体列表缓存，存储所有发现的字体文件路径，初始为None
        self._monospace_fonts : Optional[List[str]] 等宽字体名称列表缓存，存储所有检测到的等宽字体名称，初始为None
    """
    
    def __init__(self):
        """
        初始化FontManager实例
        
        创建字体管理器，初始化缓存和状态变量
        
        初始化后的对象属性：
            logger: 日志记录器
            _font_cache: 字体缓存字典，键为字体名称和大小的组合
            _system_fonts: 系统字体列表缓存，初始为None
            _monospace_fonts: 等宽字体名称列表缓存，初始为None
        """
        self.logger = logging.getLogger(__name__)
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self._system_fonts: Optional[List[Path]] = None
        self._monospace_fonts: Optional[List[str]] = None

    def get_system_fonts(self) -> List[Path]:
        """
        获取系统中所有可用字体文件的路径列表
        
        根据操作系统类型扫描不同的字体目录，支持Windows、macOS和Linux
        
        Returns:
            List[Path]: 字体文件路径列表
        """
        if self._system_fonts is not None:
            return self._system_fonts
        
        # 初始化字体路径列表
        font_paths: List[Path] = []
        system = platform.system()

        if system == "Windows":
            # Windows字体目录
            font_dirs = [
                Path("C:/Windows/Fonts"),
                Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
            ]
        elif system == "Darwin":  # macOS
            # macOS字体目录
            font_dirs = [
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library/Fonts"
            ]
        else:  # Linux和其他Unix系统
            # Linux字体目录
            font_dirs = [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local/share/fonts"
            ]

        # 扫描字体目录
        for font_dir in font_dirs:
            if font_dir.exists():
                # 支持的字体格式
                font_extensions = {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}
                for ext in font_extensions:
                    font_paths.extend(font_dir.rglob(f'*{ext}'))

        self._system_fonts = font_paths
        self.logger.info(f"发现 {len(font_paths)} 个系统字体文件")
        return font_paths

    def find_monospace_fonts(self) -> List[str]:
        """
        查找系统中的等宽字体
        
        扫描系统字体并识别等宽字体
        
        Returns:
            List[str]: 等宽字体名称列表
        """
        if self._monospace_fonts is not None:
            return self._monospace_fonts

        monospace_fonts = []

        # 常见等宽字体名称
        common_monospace = [
            'Consolas', 'Monaco', 'Menlo', 'DejaVu Sans Mono',
            'Liberation Mono', 'Courier New', 'Courier',
            'Lucida Console', 'Ubuntu Mono', 'Source Code Pro',
            'Fira Code', 'JetBrains Mono', 'Cascadia Code'
        ]

        # 验证字体是否可用
        for font_name in common_monospace:
            try:
                # 尝试加载字体以验证其可用性
                ImageFont.truetype(font_name, 12)
                monospace_fonts.append(font_name)
                self.logger.debug(f"找到等宽字体: {font_name}")
            except (OSError, IOError):
                # 在Windows上，尝试通过文件路径加载
                if platform.system() == "Windows":
                    font_path = _find_font_file(font_name)
                    if font_path:
                        try:
                            ImageFont.truetype(str(font_path), 12)
                            monospace_fonts.append(font_name)
                            self.logger.debug(f"通过路径找到等宽字体: {font_name} -> {font_path}")
                        except (OSError, IOError):
                            continue
                continue

        monospace_fonts.extend(self._detect_monospace_with_fonttools())


        # 去重并排序
        self._monospace_fonts = sorted(list(set(monospace_fonts)))
        self.logger.info(f"发现 {len(self._monospace_fonts)} 个等宽字体")
        return monospace_fonts

    def _detect_monospace_with_fonttools(self) -> List[str]:
        """
        使用fonttools库自动检测系统中的等宽字体
        
        分析系统字体文件，通过检查字符宽度判断是否为等宽字体
        
        Returns:
            List[str]: 检测到的等宽字体名称列表
            
        Notes:
            - 限制检查前50个字体文件以提高性能
            - 使用_is_monospace_font函数判断字体是否为等宽
            - 使用_get_font_name函数获取字体名称
            - 完善的异常处理确保检测过程不会中断
            - 记录详细的调试日志
        """
        monospace_fonts = []

        try:
            system_fonts = self.get_system_fonts()

            for font_path in system_fonts[:50]:  # 限制检查数量以提高性能
                try:
                    font = TTFont(str(font_path))

                    # 检查字体是否为等宽
                    if _is_monospace_font(font):
                        # 获取字体名称
                        font_name = _get_font_name(font)
                        if font_name:
                            monospace_fonts.append(font_name)

                except Exception as e:
                    self.logger.debug(f"无法分析字体 {font_path}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(WARNING_MESSAGES['fonttools_detection_failed'].format(e))

        return monospace_fonts

    def load_font(self, font_name: str, size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """
        加载指定名称和大小的字体
        
        尝试按名称加载字体，如果失败则尝试按文件路径加载，最后使用回退字体
        
        Args:
        font_name: str 字体名称
        size: int 字体大小（像素）
        
        Returns:
            Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]: 加载的字体对象
        """
        cache_key = f"{font_name}_{size}"

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        try:
            # 尝试按名称加载字体
            font = ImageFont.truetype(font_name, size)
            self._font_cache[cache_key] = font
            self.logger.debug(f"成功加载字体: {font_name}, 大小: {size}")
            return font

        except (OSError, IOError) as e:
            self.logger.debug(f"无法通过名称加载字体 {font_name}: {e}")
            
            # 在Windows上，尝试通过文件路径加载
            if platform.system() == "Windows":
                font_path = _find_font_file(font_name)
                if font_path:
                    try:
                        font = ImageFont.truetype(str(font_path), size)
                        self._font_cache[cache_key] = font
                        self.logger.debug(f"通过路径成功加载字体: {font_name} -> {font_path}, 大小: {size}")
                        return font
                    except (OSError, IOError) as path_error:
                        self.logger.debug(f"通过路径加载字体失败 {font_path}: {path_error}")
            
            self.logger.warning(WARNING_MESSAGES['font_load_failed'].format(font_name, e))
            # 回退到默认字体
            return self._load_fallback_font(size)

    def _load_fallback_font(self, size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """
        加载默认的回退字体
        
        当指定的字体无法加载时，尝试加载系统默认字体
        
        Args:
        size: int 字体大小（像素）
        
        Returns:
            Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]: 回退字体对象
        """
        # 尝试加载系统默认等宽字体
        fallback_fonts = ['Consolas', 'Monaco', 'Courier New', 'monospace']

        for font_name in fallback_fonts:
            try:
                font = ImageFont.truetype(font_name, size)
                self.logger.info(f"使用回退字体: {font_name}")
                return font
            except (OSError, IOError):
                continue

        # 最后回退到PIL默认字体
        self.logger.warning(WARNING_MESSAGES['using_pil_default_font'])
        # 返回类型转换为FreeTypeFont兼容
        default_font = ImageFont.load_default()
        # 如果默认字体不是FreeTypeFont类型，尝试创建一个基础字体
        if not hasattr(default_font, 'getbbox'):
            try:
                return ImageFont.truetype('arial.ttf', size)
            except (OSError, IOError):
                # 如果无法加载任何TrueType字体，返回默认字体
                pass
        return default_font
        
    def get_best_monospace_font(self, size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """
        获取系统中最佳的等宽字体
        
        返回系统中可用的最佳等宽字体，按优先级排序
        
        Args:
        size: int 字体大小（像素）
        
        Returns:
            Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]: 加载的最佳等宽字体对象
        """
        monospace_fonts = self.find_monospace_fonts()
        
        if monospace_fonts:
            # 优先选择常见的高质量等宽字体
            preferred_fonts = ['Consolas', 'Monaco', 'Menlo', 'DejaVu Sans Mono']
            
            for preferred in preferred_fonts:
                if preferred in monospace_fonts:
                    return self.load_font(preferred, size)
                    
            # 如果没有首选字体，使用第一个可用的
            return self.load_font(monospace_fonts[0], size)
            
        # 如果没有找到等宽字体，使用回退字体
        return self._load_fallback_font(size)

    def clear_cache(self) -> None:
        """
        清除所有字体缓存
        
        重置字体管理器的所有缓存数据，包括字体对象缓存、系统字体列表和等宽字体列表
        """
        self._font_cache.clear()
        self.logger.debug("字体缓存已清空")


def load_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """
    加载指定大小的字体
    
    独立函数，创建FontManager实例并获取最佳等宽字体
    
    Args:
        font_size: int 字体大小（像素）
        
    Returns:
        Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]: 加载的字体对象
    """
    try:
        # 在方法内部创建FontManager实例
        font_manager = FontManager()
        # 使用FontManager获取最佳等宽字体
        font = font_manager.get_best_monospace_font(font_size)
        if logger:
            logger.info(f"成功加载字体，大小: {font_size}")
        return font
        
    except Exception as e:
        logger.warning(WARNING_MESSAGES['font_manager_load_failed'].format(e))
        # 回退到PIL默认字体
        from PIL import ImageFont
        return ImageFont.load_default()


def calculate_char_size(font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]) -> tuple[int, int]:
    """
    计算指定字体中字符的实际像素大小
    
    Args:
        font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont] 要测量的字体对象
        
    Returns:
        tuple[int, int]: 字符的宽度和高度（像素）
    """
    # 使用字符'M'来测量，因为它通常是最宽的字符
    bbox = font.getbbox('M')
    char_width = int(bbox[2] - bbox[0])
    char_height = int(bbox[3] - bbox[1])
    
    # 确保最小尺寸
    char_width = max(char_width, 6)
    char_height = max(char_height, 8)
    
    return char_width, char_height