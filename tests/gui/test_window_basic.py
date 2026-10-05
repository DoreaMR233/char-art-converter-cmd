"""T-301~T-303：主窗口基本结构、状态机、进度条显示（gui\\main_window.py）。"""
from gui.main_window import AppState


def test_window_title_and_size(window):
    """T-301：标题、初始尺寸与最小尺寸。"""
    assert window.windowTitle() == "字符画转换器"
    assert window.minimumSize().width() == 800
    assert window.minimumSize().height() == 560
    assert window.width() == 960
    assert window.height() == 640


def test_initial_idle_state(window):
    """T-302 前置：初始状态为 IDLE，控件可见性/可用性与设计一致。"""
    assert window._state is AppState.IDLE
    # 初始 (0,0) 为忙碌指示器，Qt6 下 value() 返回 -1
    assert window.progress_bar.minimum() == 0
    assert window.progress_bar.maximum() == 0
    assert window.progress_bar.value() in (-1, 0)
    assert window.stage_label.text() == "就绪"
    assert window.start_btn.isEnabled()
    assert not window.cancel_btn.isVisible()
    assert not window.open_folder_btn.isVisible()


def test_state_machine_transitions(window):
    """T-302：状态流转时组件启用/可见性同步。"""
    window._set_state(AppState.PREPARING)
    assert not window.start_btn.isEnabled()
    assert window.cancel_btn.isVisible() and window.cancel_btn.isEnabled()
    assert not window.param_panel.density_combo.isEnabled()
    assert not window.file_picker.isEnabled()

    window._set_state(AppState.RUNNING)
    assert not window.start_btn.isEnabled()
    assert window.cancel_btn.isVisible() and window.cancel_btn.isEnabled()

    window._set_state(AppState.CANCELING)
    assert window.cancel_btn.text() == "正在取消…"
    assert not window.cancel_btn.isEnabled()

    window._set_state(AppState.SUCCESS)
    assert window.start_btn.isEnabled()
    assert not window.cancel_btn.isVisible()
    assert window.param_panel.density_combo.isEnabled()
    assert window.file_picker.isEnabled()

    window._set_state(AppState.ERROR)
    assert window.start_btn.isEnabled()
    assert not window.cancel_btn.isVisible()


def test_progress_display(window):
    """T-303：_on_progress 常规进度、不确定进度（total<=0→(0,0)）与阶段文案。"""
    window._on_progress(5, 10, "统计视频帧数", "")
    assert window.progress_bar.maximum() == 10
    assert window.progress_bar.value() == 5
    assert window.stage_label.text() == "统计视频帧数"

    window._on_progress(1, 10, "处理视频帧", "3/10")
    assert window.stage_label.text() == "处理视频帧 3/10"

    window._on_progress(2, 0, "统计视频帧数", "")
    assert window.progress_bar.maximum() == 0
    # (0,0) 忙碌指示器，Qt6 下 value() 为 -1
    assert window.progress_bar.value() in (-1, 0)
    assert window.stage_label.text() == "统计视频帧数"


def test_thread_bars_routing(window):
    """T-303b：GIF/视频任务的每线程进度条——按线程数建条、position 分流、清理。"""
    # 初始无线程条
    assert window._thread_bars == []
    assert not window.thread_bars_container.isVisible()

    # 模拟 worker 报告 3 线程
    window._on_thread_count(3)
    assert len(window._thread_bars) == 3
    assert window.thread_bars_container.isVisible()
    for bar in window._thread_bars:
        assert bar.objectName() == "threadBar"
        assert bar.maximum() == 0   # 初始不确定

    # position>0 → 线程条；主条不受影响
    window._on_progress(5, 10, "帧 1 GPU生成", "", 1)
    assert window._thread_bars[0].maximum() == 10
    assert window._thread_bars[0].value() == 5
    assert window.progress_bar.maximum() == 0
    assert window.stage_label.text() == "就绪"

    # 线程条自身的不确定进度
    window._on_progress(1, 0, "帧 2 GPU生成", "", 2)
    assert window._thread_bars[1].maximum() == 0

    # position 0 → 主条 + 阶段文案
    window._on_progress(5, 10, "统计视频帧数", "", 0)
    assert window.progress_bar.maximum() == 10
    assert window.progress_bar.value() == 5
    assert window.stage_label.text() == "统计视频帧数"

    # 越界 position 不影响任何条
    window._on_progress(1, 10, "帧 9 GPU生成", "", 9)
    for bar in window._thread_bars:
        assert bar.maximum() in (0, 10)
    assert window.progress_bar.maximum() == 10

    # 任务结束清理
    window._clear_thread_bars()
    assert window._thread_bars == []
    assert not window.thread_bars_container.isVisible()
