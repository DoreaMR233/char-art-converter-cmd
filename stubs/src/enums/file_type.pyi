from ..configs import ERROR_MESSAGES as ERROR_MESSAGES
from enum import Enum
from pathlib import Path

class FileType(Enum):
    IMAGE = 'image'
    VIDEO = 'video'
    TEXT = 'text'
    AUDIO = 'audio'
    @staticmethod
    def from_string(value: str) -> FileType: ...
    @staticmethod
    def from_path(input_path: Path) -> FileType: ...
