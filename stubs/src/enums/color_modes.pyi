from enum import Enum

class ColorModes(Enum):
    GRAYSCALE = 'grayscale'
    COLOR = 'color'
    COLOR_BACKGROUND = 'colorBackground'
    @staticmethod
    def from_string(value: str) -> ColorModes: ...
