"""T-313：转换完成信号处理（成功 / 取消 / 失败 / 文件错误）。"""
from gui.main_window import AppState


class _DummyWorker:
    """替代真实 worker 的 deleteLater/isRunning 桩。"""

    def deleteLater(self):
        pass

    def isRunning(self):
        return False


def test_finished_success(window, tmp_path):
    """T-313a：code=0 → SUCCESS、进度 (0,1)=1、打开按钮可见。"""
    window._set_state(AppState.RUNNING)
    window._worker = _DummyWorker()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    window._on_finished(0, {"output_dir": str(output_dir)})

    assert window._state is AppState.SUCCESS
    assert window.progress_bar.maximum() == 1
    assert window.progress_bar.value() == 1
    assert window.stage_label.text() == "转换完成"
    assert f"转换成功。产物目录：{output_dir}" in window.status_label.text()
    assert window.open_folder_btn.isVisible()
    assert window.start_btn.isEnabled()


def test_finished_cancelled(window):
    """T-313b：code=130 → IDLE、进度归零、“已取消”文案。"""
    window._set_state(AppState.CANCELING)
    window._worker = _DummyWorker()

    window._on_finished(130, {})

    assert window._state is AppState.IDLE
    assert window.progress_bar.value() == 0
    assert window.stage_label.text() == "已取消"
    assert window.status_label.text() == "转换已取消，可重新开始。"
    assert not window.open_folder_btn.isVisible()
    assert window.start_btn.isEnabled()


def test_finished_generic_error(window):
    """T-313c：code=1 → ERROR、失败文案含错误信息。"""
    window._set_state(AppState.RUNNING)
    window._worker = _DummyWorker()

    window._on_finished(1, {"error": "模拟失败"})

    assert window._state is AppState.ERROR
    assert window.stage_label.text() == "转换失败"
    assert "转换失败" in window.status_label.text()
    assert "模拟失败" in window.status_label.text()
    assert not window.open_folder_btn.isVisible()
    assert window.start_btn.isEnabled()


def test_finished_file_error_shows_input_error(window):
    """T-313d：code=2 → ERROR 且在输入框标注错误。"""
    window._set_state(AppState.RUNNING)
    window._worker = _DummyWorker()

    window._on_finished(2, {"error": "文件不存在: x.png"})

    assert window._state is AppState.ERROR
    assert window.input_error.isVisible()
    assert "文件不存在" in window.input_error.text()
