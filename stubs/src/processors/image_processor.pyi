from _typeshed import Incomplete
from pathlib import Path
from typing import Any

from .based_processor import BasedProcessor as BasedProcessor

logger: Incomplete

class ImageProcessor(BasedProcessor):
    is_animated: bool
    temp_dir: Path | None
    def __init__(self, args: Any) -> None: ...
    def start(self) -> None: ...
    def process_static_image(self) -> None: ...
    def process_animated_image(self) -> None: ...
