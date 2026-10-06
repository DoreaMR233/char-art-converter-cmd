"""tests/gui 专用 fixtures。

关键决策（供后续维护者参考）：GUI worker 在 QThread 内运行，会经
based_processor.__init__ 调 check_ffmpeg_available() → subprocess.run()。
CPython 3.12 Windows 下，QThread 内创建 subprocess 句柄曾实测触发
"Fatal Python error: Aborted"（subprocess.py:1395 _get_handles）导致整个 pytest
进程崩溃（full_run1 于 test_image_conversion_success 崩溃；当时与预览线程并发，
预览窗格已在 m01135 需求中移除）。
ffmpeg 可用性已由 session 级 ffmpeg_available 夹具在主线程串行验证一次，
GUI 用例聚焦 UI 状态机，因此在此屏蔽 worker 内的重复探测。
"""
import pytest


@pytest.fixture(autouse=True)
def _no_ffmpeg_subprocess_in_worker(monkeypatch):
    import src.processors.based_processor as based_processor

    monkeypatch.setattr(based_processor, "check_ffmpeg_available", lambda: None)


