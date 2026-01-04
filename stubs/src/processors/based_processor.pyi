import threading
from _typeshed import Incomplete
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
# 导入 torch 及其类型
from torch import device as torch_device

from ..enums import SaveModes as SaveModes
from ..enums.color_modes import ColorModes as ColorModes
from ..enums.file_type import FileType as FileType

class DummyFlag:
    value: bool = False

logger: Incomplete

class BasedProcessor:
    should_stop: bool
    file_type: FileType
    density: str
    color_mode: ColorModes
    limit_size: list[int] | None
    enable_multithread: bool
    with_text: bool
    max_workers: Incomplete
    is_debug: bool
    position_list: list[int]
    used_positions: set
    position_lock: threading.Lock
    char_set: str
    char_count: int
    font_size: int
    font: Any
    input_path: Path
    with_image: bool
    output_dir: Path
    text_path: Path | None
    image_path: Path | None
    temp_frame_paths: dict[int, Path]
    torch: Any | None
    torch_cuda_available: bool
    device: Optional[torch_device]
    gpu_memory_limit: int | None
    enable_gpu: bool
    def __init__(self, args: Any, file_type: FileType) -> None: ...
    def signal_handler(self, signum: int, frame: Any) -> None: ...
    def get_progress_position(self, frame_index: int, is_multithread: bool = True) -> int: ...
    def release_progress_position(self, position: int) -> None: ...
    def process_single_frame(self, frame_index: int, image: Image.Image, save_mode: SaveModes, text_frame_path: Path, image_frame_path: Path, duration: int = 0, is_multithread: bool = True) -> Tuple[int, int]: ...
    def process_frames_multithreaded(self, total_frames: int, pbar_description: str, process_frame_func) -> list[int]: ...
    def process_frames_sequentially(self, total_frames: int, pbar_description: str, process_frame_func) -> list[int]: ...
    def load_frames_from_temp(self, total_frames: int, frames_paths: Optional[Dict[int, Path]] = None) -> List[Image.Image]: ...
