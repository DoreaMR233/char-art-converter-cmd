import threading
from .format_utils import format_size as format_size, format_speed as format_speed
from _typeshed import Incomplete
from pathlib import Path
from typing import Any, Callable

logger: Incomplete

def get_tqdm_kwargs(total: int, desc: str, verbose: bool = True, unit: str = 'it', position: int = 0) -> dict[str, Any]: ...
def show_value_file_save_progress(file_path: Path, save_completed: threading.Event, description: str, verbose: bool, position: int = 0, should_stop: Callable[[], bool] | None = None) -> None: ...
def show_project_status_progress(total: int, description: str, verbose: bool = True, unit: str = 'it', position: int = 0, hide_counter: bool = False) -> Any: ...
def no_value_file_save_progress(file_path: Path, save_completed: threading.Event, description: str, verbose: bool, position: int = 0, should_stop: Callable[[], bool] | None = None) -> None: ...


