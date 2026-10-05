"""T-103~T-107：progress sink 增量（src\\utils\\progress_bar_utils.py，改造 C2）。"""
import threading

import pytest

from src.utils import progress_bar_utils as pbu


@pytest.fixture(autouse=True)
def _clean_sink():
    pbu.set_progress_sink(None)
    yield
    pbu.set_progress_sink(None)


def test_set_get_progress_sink():
    """T-103：set/get_progress_sink 往返一致。"""
    assert pbu.get_progress_sink() is None
    marker = object()
    pbu.set_progress_sink(marker)
    assert pbu.get_progress_sink() is marker
    pbu.set_progress_sink(None)
    assert pbu.get_progress_sink() is None


def test_sink_progress_bar_bridges_events():
    """T-104：_SinkProgressBar 将 update/set_postfix_str/set_description 桥接为 (done,total,desc,extra)。"""
    events = []
    pbu.set_progress_sink(lambda *args: events.append(args))

    bar = pbu._SinkProgressBar(10, "处理帧")
    bar.update(3)
    bar.update()
    bar.set_postfix_str("帧 5/10")
    bar.set_description("处理视频帧")
    with bar:
        pass
    bar.close()

    assert events == [
        (3, 10, "处理帧", ""),
        (4, 10, "处理帧", ""),
        (4, 10, "处理帧", "帧 5/10"),
        (4, 10, "处理视频帧", ""),
        (4, 10, "处理视频帧", ""),
    ]


def test_make_progress_returns_sink_or_tqdm():
    """T-105：sink 挂载时 _make_progress 返回替身，未挂载时返回真实 tqdm。"""
    pbu.set_progress_sink(lambda *args: None)
    bar = pbu._make_progress(5, "任务")
    assert isinstance(bar, pbu._SinkProgressBar)
    bar.close()

    pbu.set_progress_sink(None)
    bar = pbu._make_progress(5, "任务")
    assert type(bar).__module__.split(".")[0] == "tqdm"
    bar.close()


def test_save_progress_short_circuits_with_sink():
    """T-106：show_value_file_save_progress 有 sink 时回传 (0,1,desc,'') 并置位 save_completed。"""
    events = []
    pbu.set_progress_sink(lambda *args: events.append(args))
    save_completed = threading.Event()

    pbu.show_value_file_save_progress("dummy.txt", save_completed, "保存", True)

    assert save_completed.is_set()
    assert events == [(0, 1, "保存", "")]


def test_project_status_progress_uses_make_progress():
    """T-107：show_project_status_progress 走 _make_progress（sink 时返回替身并桥接事件）。"""
    events = []
    pbu.set_progress_sink(lambda *args: events.append(args))

    bar = pbu.show_project_status_progress(7, "统计视频帧数")
    assert isinstance(bar, pbu._SinkProgressBar)
    bar.update(1)
    assert events == [(1, 7, "统计视频帧数", "")]
