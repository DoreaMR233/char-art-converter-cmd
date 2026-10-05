"""T-101 / T-102：SIGINT 注册主线程守卫（src\\processors\\based_processor.py:147-150，改造 C1）。"""
import signal
import threading
from argparse import Namespace
from pathlib import Path

import pytest

from src.enums.file_type import FileType
from src.processors import based_processor


def _make_args(tmp_path: Path) -> Namespace:
    image = tmp_path / "input.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return Namespace(
        input=str(image),
        output=None,
        density="medium",
        color_mode="color",
        limit_size=None,
        no_multithread=True,
        with_text=True,
        with_image=False,
        enable_gpu=False,
        gpu_memory_limit=None,
        debug=False,
    )


@pytest.fixture
def stubbed_init(monkeypatch):
    """桩掉 __init__ 里除 SIGINT 守卫外的重依赖（torch/参数校验/ffmpeg 探测）。

    兼容两种调用形态：模块级函数调用与实例方法调用。
    """
    def _maybe_setattr(target, name, value):
        if hasattr(target, name):
            monkeypatch.setattr(target, name, value)

    _maybe_setattr(based_processor, "validate_arguments", lambda *a, **k: None)
    _maybe_setattr(based_processor.BasedProcessor, "validate_arguments", lambda self, *a, **k: None)
    _maybe_setattr(based_processor, "check_ffmpeg_available", lambda *a, **k: None)
    _maybe_setattr(based_processor.BasedProcessor, "check_ffmpeg_available", lambda self, *a, **k: None)
    _maybe_setattr(based_processor, "init_pytorch_and_gpu", lambda: (None, False, None))


def test_signal_handler_registered_in_main_thread(stubbed_init, tmp_path):
    """T-101：主线程构造会注册 SIGINT 处理器；测试结束后必须还原。"""
    previous = signal.getsignal(signal.SIGINT)
    try:
        processor = based_processor.BasedProcessor(_make_args(tmp_path), FileType.IMAGE)
        current = signal.getsignal(signal.SIGINT)
        assert current is not previous, "主线程应注册新的 SIGINT 处理器"
        assert getattr(current, "__self__", None) is processor, "处理器应绑定到本次构造的实例"
    finally:
        signal.signal(signal.SIGINT, previous)


def test_signal_handler_skipped_in_worker_thread(stubbed_init, monkeypatch, tmp_path):
    """T-102：非主线程构造不调用 signal.signal（避免 ValueError 与主线程信号被替换）。"""
    calls = []

    def recording_signal(signum, handler):
        calls.append((signum, handler))
        return handler

    monkeypatch.setattr(based_processor.signal, "signal", recording_signal)

    result = {}

    def _build():
        try:
            processor = based_processor.BasedProcessor(_make_args(tmp_path), FileType.IMAGE)
            result["ok"] = True
            result["handler"] = processor.signal_handler
        except Exception as exc:  # pragma: no cover - 异常经 result 带出用于断言
            result["error"] = repr(exc)

    thread = threading.Thread(target=_build)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "非主线程构造 60s 未完成"
    assert "error" not in result, f"非主线程构造失败: {result['error']}"
    assert calls == [], f"非主线程不应调用 signal.signal，实际调用 {calls}"
