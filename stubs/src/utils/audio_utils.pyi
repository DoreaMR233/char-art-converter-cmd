from _typeshed import Incomplete
from pathlib import Path
from typing import Optional, Callable

logger: Incomplete

def extract_audio(input_path: Path, temp_dir: Path, video_info: str,should_stop: Optional[Callable[[], bool]] = None) -> Path | None: ...
