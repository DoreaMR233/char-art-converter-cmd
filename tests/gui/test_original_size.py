"""输入文件原始尺寸读取：gui\\main_window.py:read_original_size / _probe_original_size。

GUI 自定义尺寸模式的宽/高预填依赖该函数，因此除图片/视频两条常规分支外，
额外覆盖 OpenCV 失败后回退 ffprobe 的分支与彻底读不到时的 None。
"""
import pytest

from gui.main_window import read_original_size


def test_read_image_size(sample_image):
    """图片走 PIL 分支，返回真实像素尺寸。"""
    assert read_original_size(str(sample_image)) == (64, 48)


def test_read_video_size(sample_video):
    """视频走 OpenCV 首帧分支，返回真实像素尺寸。"""
    assert read_original_size(str(sample_video)) == (320, 240)


def test_read_video_size_falls_back_to_ffprobe(sample_video, monkeypatch, ffmpeg_available):
    """OpenCV 读不到时回退 ffprobe，仍能给出原图尺寸（自定义尺寸预填不会落空）。"""
    if not ffmpeg_available:
        pytest.skip("FFmpeg 不可用（文档允许视频用例跳过）")

    import cv2

    class _BrokenCapture:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("模拟 OpenCV 打不开该封装格式")

    monkeypatch.setattr(cv2, "VideoCapture", _BrokenCapture)
    assert read_original_size(str(sample_video)) == (320, 240)


def test_read_unsupported_returns_none(unsupported_file):
    """三种方式都读不到（纯文本文件）时返回 None，且不向调用方抛异常。"""
    assert read_original_size(str(unsupported_file)) is None
