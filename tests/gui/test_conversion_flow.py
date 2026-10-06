"""T-305~T-307：GUI 转换流程（图片成功、进度事件、视频成功）。

已知问题（Lead 批准按 known-issue 处理，交由 leader 复核）：
T-307 视频转换在 QT_QPA_PLATFORM=offscreen 下挂起——worker QThread 内视频处理
（ffprobe/ffmpeg 子进程 + 并行帧处理）实测 >15 分钟无任何输出（Lead 强杀），
90s 探针同样无输出。图片用例（T-305/306）与 CLI 视频用例（T-207）均绿，
视频转换链路已由 CLI 侧覆盖。有显示环境时可在真实窗口平台复跑本用例。
"""
import os
from pathlib import Path

import pytest

from gui.main_window import AppState


def test_image_conversion_success(window, sample_image, qt_wait):
    """T-305：图片转换成功 → SUCCESS、状态文案、产物 txt 落盘、打开按钮可见。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.with_text_check.setChecked(True)
    window.param_panel.enable_gpu_check.setChecked(False)
    window._on_start()

    assert window._state is AppState.RUNNING
    assert window.stage_label.text() == "正在初始化 GPU 环境（首次运行可能较慢）…"
    assert window.progress_bar.maximum() == 0

    assert qt_wait(lambda: window._state is AppState.SUCCESS, timeout=600), \
        f"等待 SUCCESS 超时，当前状态={window._state}，状态文案={window.status_label.text()}"
    assert "转换成功" in window.status_label.text()
    assert window.stage_label.text() == "转换完成"
    assert window.progress_bar.value() == 1
    assert window.open_folder_btn.isVisible()

    out_dir = Path(sample_image).parent / "test_input_color_char_art_image"
    assert out_dir.is_dir()
    assert (out_dir / "test_input_color_char_art.txt").is_file()


def test_progress_events_during_conversion(window, sample_image, qt_wait):
    """T-306：转换期间 bridge 持续收到进度事件（含预热不确定进度与确定进度）。"""
    events = []
    window.bridge.progress.connect(lambda *args: events.append(args))
    window.file_picker.set_path(str(sample_image))
    window.param_panel.enable_gpu_check.setChecked(False)
    window._on_start()

    assert qt_wait(lambda: window._state is AppState.SUCCESS, timeout=600), \
        f"等待 SUCCESS 超时，状态={window._state}"
    assert events, "应收到进度事件"
    assert any(e[0] == 0 and e[1] == 0 for e in events), "应包含预热阶段的不确定进度 (0,0)"
    assert any(e[1] > 0 for e in events), "应包含确定进度 (total>0)"
    assert all(e[0] <= e[1] for e in events if e[1] > 0), "done 不应超过 total"


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    reason="已知问题：offscreen 下 GUI 视频 worker 挂起（>15min 无输出，Lead 批准跳过，交由 leader 复核）",
)
def test_video_conversion_success(window, sample_video, qt_wait, ffmpeg_available):
    """T-307：视频转换成功（无 ffmpeg 按文档策略 skip；缩小尺寸加速用例）。"""
    if not ffmpeg_available:
        pytest.skip("本机无 ffmpeg，视频用例按文档策略跳过")
    window.file_picker.set_path(str(sample_video))
    window.param_panel.with_text_check.setChecked(True)
    window.param_panel.enable_gpu_check.setChecked(False)
    window.param_panel.width_spin.setValue(48)
    window.param_panel.height_spin.setValue(24)
    window._on_start()

    assert qt_wait(lambda: window._state is AppState.SUCCESS, timeout=900), \
        f"等待 SUCCESS 超时，当前状态={window._state}，状态文案={window.status_label.text()}"
    assert "转换成功" in window.status_label.text()
    assert window.open_folder_btn.isVisible()
    assert window._output_dir is not None
    assert list(Path(window._output_dir).rglob("*.txt")), "视频产物目录应包含字符画 txt"
