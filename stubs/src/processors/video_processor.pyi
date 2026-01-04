from ..configs import ERROR_MESSAGES as ERROR_MESSAGES, SUCCESS_MESSAGES as SUCCESS_MESSAGES, WARNING_MESSAGES as WARNING_MESSAGES
from ..configs.message_config import ASSERT_MESSAGES as ASSERT_MESSAGES
from ..configs.video_config import DEFAULT_FPS as DEFAULT_FPS, DEFAULT_FRAME_EXTENSIONS as DEFAULT_FRAME_EXTENSIONS
from ..enums.file_type import FileType as FileType
from ..enums.save_modes import SaveModes as SaveModes
from ..utils import calculate_resized_dimensions as calculate_resized_dimensions, check_ffmpeg_available as check_ffmpeg_available, cleanup_files as cleanup_files, create_temp_dir as create_temp_dir, extract_audio as extract_audio, format_time as format_time, get_output_path as get_output_path, video_utils as video_utils
from .based_processor import BasedProcessor as BasedProcessor
from _typeshed import Incomplete
from pathlib import Path
from typing import Any

logger: Incomplete

class VideoProcessor(BasedProcessor):
    audio_path: Path | None
    temp_image_dir: Path
    temp_frame_paths: dict[int, Path]
    temp_audio_dir: Path
    ffmpeg_available: bool
    def __init__(self, args: Any) -> None: ...
    def start(self) -> None: ...
    def process_video(self) -> None: ...
