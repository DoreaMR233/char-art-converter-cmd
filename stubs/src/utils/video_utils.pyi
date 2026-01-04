from _typeshed import Incomplete
from pathlib import Path
from typing import Optional, Callable

logger: Incomplete

def get_video_info(video_path: Path | str, show_format: bool = True, show_streams: bool = True, show_packets: bool = False, show_frames: bool = False) -> str: ...
def create_video(input_files: list[Path], output_file: Path,video_info: str, fps: float = 30.0, codec: str | None = None, bitrate: int | None = None,threads_num: int = 1, should_stop: Optional[Callable[[], bool]] = None) -> None: ...

