from ..configs import ERROR_MESSAGES as ERROR_MESSAGES
from enum import Enum

class SaveModes(Enum):
    TEXT = 'text'
    ANIMATED_TEXT = 'animated_text'
    STATIC_IMAGE = 'static_image'
    ANIMATED_IMAGE = 'animated_image'
    AUDIO = 'audio'
    ANIMATED_IMAGE_TMP_FRAME = 'animated_image_tmp_frame'
    VIDEO_TMP_FRAME = 'video_tmp_frame'
    VIDEO = 'video'
    MERGE_ANIMATE = 'merge_animate'
    @staticmethod
    def from_string(value: str) -> SaveModes: ...
