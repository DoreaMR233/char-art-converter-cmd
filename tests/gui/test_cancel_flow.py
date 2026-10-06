"""T-308~T-310：GUI 取消流程（视频中断、取消后重开、closeEvent 拦截）。

已知问题（Lead 批准按 known-issue 处理，交由 leader 复核）：
T-308/T-309 依赖视频转换 worker，在 QT_QPA_PLATFORM=offscreen 下挂起
（>15min 无输出，Lead 强杀；同 test_conversion_flow 的 T-307）。
CLI 视频取消链路由 T-213（slow）覆盖。有显示环境时可在真实窗口平台复跑。
"""
import os
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent

from gui.main_window import AppState


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    reason="已知问题：offscreen 下 GUI 视频 worker 挂起（Lead 批准跳过，交由 leader 复核）",
)
def test_cancel_video_conversion(window, sample_video, qt_wait, ffmpeg_available):
    """T-308：视频转换中取消 → CANCELING→IDLE、产物清理、日志出现“清理”。"""
    if not ffmpeg_available:
        pytest.skip("本机无 ffmpeg，视频用例按文档策略跳过")
    window.file_picker.set_path(str(sample_video))
    window.param_panel.with_text_check.setChecked(True)
    window.param_panel.enable_gpu_check.setChecked(False)
    window._on_start()

    assert qt_wait(lambda: window.progress_bar.maximum() > 0, timeout=300), \
        f"等待视频进入确定进度超时，状态={window._state}，阶段={window.stage_label.text()}"

    window._on_cancel()
    assert window._state is AppState.CANCELING
    assert "正在取消" in window.status_label.text()

    assert qt_wait(lambda: window._state is AppState.IDLE, timeout=300), \
        f"等待取消完成超时，状态={window._state}，文案={window.status_label.text()}"
    assert window.status_label.text() == "转换已取消，可重新开始。"
    assert window.stage_label.text() == "已取消"
    assert window.progress_bar.value() == 0
    assert window.start_btn.isEnabled()
    assert not window.cancel_btn.isVisible()
    assert not window.open_folder_btn.isVisible()

    log_view = window.findChild(type(window.log_drawer.text_edit), "logView")
    assert qt_wait(lambda: "清理" in log_view.toPlainText(), timeout=30), \
        "取消后日志应记录输出目录清理（SUCCESS_MESSAGES['output_folder_cleaned']）"


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    reason="已知问题：offscreen 下 GUI 视频 worker 挂起（Lead 批准跳过，交由 leader 复核）",
)
def test_restart_after_cancel(window, sample_video, sample_image, qt_wait, ffmpeg_available):
    """T-309：取消完成后可再次开始并成功转换。"""
    if not ffmpeg_available:
        pytest.skip("本机无 ffmpeg，视频用例按文档策略跳过")
    window.file_picker.set_path(str(sample_video))
    window.param_panel.enable_gpu_check.setChecked(False)
    window._on_start()
    assert qt_wait(lambda: window.progress_bar.maximum() > 0, timeout=300)
    window._on_cancel()
    assert qt_wait(lambda: window._state is AppState.IDLE, timeout=300)
    assert window.start_btn.isEnabled()

    window.file_picker.set_path(str(sample_image))
    window._on_start()
    assert qt_wait(lambda: window._state is AppState.SUCCESS, timeout=600), \
        f"取消后重开转换失败，状态={window._state}，文案={window.status_label.text()}"
    assert "转换成功" in window.status_label.text()


def test_close_intercepts_while_running(window):
    """T-310：运行中 closeEvent 被拦截（ignore + 触发取消）。"""
    class _FakeWorker:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def isRunning(self):
            return True

    fake = _FakeWorker()
    window._worker = fake
    window._set_state(AppState.RUNNING)

    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()
    assert window._close_requested is True
    assert window._state is AppState.CANCELING
    assert fake.cancelled is True


def test_close_when_idle(window):
    """T-310 附：空闲状态 closeEvent 正常放行。"""
    window._set_state(AppState.IDLE)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert window._close_requested is False
