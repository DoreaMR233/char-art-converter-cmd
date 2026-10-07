"""T-213 配套：Windows 控制台控制处理器（Ctrl+Break 优雅中断）单元测试。

背景：Windows 上 `import torch` 会间接载入 Intel Fortran 运行时（MKL），该运行时在
Python 之后注册控制台控制处理器，并抢先处理 CTRL_BREAK（打印 forrtl 报文后进入 CRT
原生终止路径）。当 CUDA 上下文活跃时该路径不会返回，进程永久挂起，基于 signal 的
中断逻辑（`BasedProcessor.signal_handler`）永不执行。为此新增
`src/utils/console_utils.py`：在 torch 载入之后注册更晚（按 LIFO 先执行）的控制台
处理器，把 CTRL_C / CTRL_BREAK 转发为 SIGINT。
"""
import ctypes
import signal
import sys
import threading
from argparse import Namespace
from pathlib import Path

import pytest

from src.enums.file_type import FileType
from src.processors import based_processor
from src.utils import console_utils


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
    """桩掉 __init__ 里除信号/控制台处理器注册外的重依赖（参数校验、ffmpeg 探测）。"""
    def _maybe_setattr(target, name, value):
        if hasattr(target, name):
            monkeypatch.setattr(target, name, value)

    _maybe_setattr(based_processor, "validate_arguments", lambda *a, **k: None)
    _maybe_setattr(based_processor.BasedProcessor, "validate_arguments", lambda self, *a, **k: None)
    _maybe_setattr(based_processor, "check_ffmpeg_available", lambda *a, **k: None)
    _maybe_setattr(based_processor.BasedProcessor, "check_ffmpeg_available", lambda self, *a, **k: None)


def test_handle_console_event_forwards_interrupt(monkeypatch):
    """CTRL_C / CTRL_BREAK 转发为 SIGINT 并返回 True；其他事件不处理。"""
    forwarded = []
    monkeypatch.setattr(console_utils.signal, "raise_signal", lambda signum: forwarded.append(signum))

    assert console_utils._handle_console_event(console_utils.CTRL_C_EVENT) is True
    assert console_utils._handle_console_event(console_utils.CTRL_BREAK_EVENT) is True
    assert forwarded == [signal.SIGINT, signal.SIGINT]

    assert console_utils._handle_console_event(console_utils.CTRL_CLOSE_EVENT) is False
    assert console_utils._handle_console_event(console_utils.CTRL_LOGOFF_EVENT) is False
    assert forwarded == [signal.SIGINT, signal.SIGINT], "非中断事件不应转发 SIGINT"


def test_install_returns_false_off_windows(monkeypatch):
    """非 Windows 平台安全退化，不触碰 kernel32。"""
    monkeypatch.setattr(console_utils.sys, "platform", "linux")

    def _boom(*args, **kwargs):
        raise AssertionError("非 Windows 平台不应调用 WinDLL")

    monkeypatch.setattr(ctypes, "WinDLL", _boom)
    assert console_utils.install_console_interrupt_handler() is False


def test_install_is_idempotent(monkeypatch):
    """已注册时直接返回 True，不重复注册。"""
    monkeypatch.setattr(console_utils, "_console_handler", object())

    def _boom(*args, **kwargs):
        raise AssertionError("重复调用不应再次注册控制台处理器")

    monkeypatch.setattr(ctypes, "WinDLL", _boom)
    assert console_utils.install_console_interrupt_handler() is True


@pytest.mark.skipif(sys.platform != "win32", reason="控制台控制事件仅 Windows 可用")
def test_install_registers_real_handler():
    """真实注册路径：注册成功后幂等，测试结束注销以免影响 pytest 自身。"""
    assert console_utils.install_console_interrupt_handler() is True
    handler = console_utils._console_handler
    assert handler is not None, "注册成功后必须持有回调引用（防止被 GC）"

    try:
        assert console_utils.install_console_interrupt_handler() is True
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(handler, False)
        console_utils._console_handler = None


def test_console_handler_installed_after_torch_init(stubbed_init, monkeypatch, tmp_path):
    """主线程构造时，控制台处理器必须在 torch 初始化之后注册（LIFO 才能抢占 RTL）。"""
    calls = []

    def fake_torch_init():
        calls.append("torch")
        return None, False, None

    monkeypatch.setattr(based_processor, "init_pytorch_and_gpu", fake_torch_init)
    monkeypatch.setattr(based_processor, "install_console_interrupt_handler",
                        lambda: calls.append("console"))

    previous = signal.getsignal(signal.SIGINT)
    try:
        based_processor.BasedProcessor(_make_args(tmp_path), FileType.IMAGE)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert calls == ["torch", "console"], f"注册顺序错误: {calls}"


def test_console_handler_skipped_in_worker_thread(stubbed_init, monkeypatch, tmp_path):
    """非主线程构造不注册控制台处理器（避免子线程劫持整个进程的中断行为）。"""
    calls = []

    monkeypatch.setattr(based_processor, "init_pytorch_and_gpu", lambda: (None, False, None))
    monkeypatch.setattr(based_processor, "install_console_interrupt_handler",
                        lambda: calls.append("console"))

    result = {}

    def _build():
        try:
            based_processor.BasedProcessor(_make_args(tmp_path), FileType.IMAGE)
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - 异常经 result 带出用于断言
            result["error"] = repr(exc)

    thread = threading.Thread(target=_build)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "非主线程构造 60s 未完成"
    assert "error" not in result, f"非主线程构造失败: {result.get('error')}"
    assert calls == [], f"非主线程不应注册控制台处理器，实际调用 {calls}"
