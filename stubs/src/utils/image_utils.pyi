from ..configs import ANIMATED_FORMATS as ANIMATED_FORMATS
from pathlib import Path
from typing import List, Optional, Callable


def is_animated_image(file_path: Path) -> bool: ...
def create_animated_image(frames_paths: List[Path], output_path: Path, temp_dir: Path,
                           durations: Optional[List[float]] = None,
                            threads_num: int = 1,
                           should_stop: Optional[Callable[[], bool]] = None) -> None: ...
