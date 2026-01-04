from ..enums.color_modes import ColorModes as ColorModes
from .image_config import SUPPORTED_IMAGE_FORMATS as SUPPORTED_IMAGE_FORMATS
from .video_config import SUPPORTED_VIDEO_FORMATS as SUPPORTED_VIDEO_FORMATS
from _typeshed import Incomplete

CHAR_DENSITY_CONFIG: dict[str, str]
COLOR_MODES: list[str]
ALL_SUPPORTED_FORMATS: list[str]
keys: Incomplete
DEFAULT_DENSITY: str
DEFAULT_COLOR_MODE: ColorModes
DEFAULT_LIMIT_SIZE: None | list[int]
DEFAULT_WITH_TEXT: bool
DEFAULT_WITH_IMAGE: bool
DEFAULT_FONT_SIZE: int
DEFAULT_LOG_LEVEL: str
LOG_FORMAT: str
LOG_DATE_FORMAT: str
ENABLE_GPU: bool
GPU_MEMORY_LIMIT: float
