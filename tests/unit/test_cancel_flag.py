"""T-110 / T-111：GUI 取消路径同时置位双停止标志（gui\\worker.py:73-83，改造 C3）。

cancel 为纯 Python 方法，直接以未绑定方式调用，避免为标志位断言启动 QThread。
"""
from types import SimpleNamespace

from gui.worker import ConversionWorker
from src.processors.based_processor import BasedProcessor, DummyFlag


def test_cancel_sets_should_stop_and_plain_flag():
    """T-110：global_exit_flag 无 .value 属性（布尔形态）时，cancel 置 True。"""
    processor = SimpleNamespace(should_stop=False, global_exit_flag=False)
    fake_self = SimpleNamespace(_processor=processor)

    ConversionWorker.cancel(fake_self)

    assert processor.should_stop is True
    assert processor.global_exit_flag is True


def test_cancel_sets_flag_value_attr():
    """T-111：global_exit_flag 有 .value 属性时，cancel 置 flag.value=True。"""
    flag = SimpleNamespace(value=False)
    processor = SimpleNamespace(should_stop=False, global_exit_flag=flag)
    fake_self = SimpleNamespace(_processor=processor)

    ConversionWorker.cancel(fake_self)

    assert processor.should_stop is True
    assert flag.value is True


def test_dummy_flag_is_falsy_and_toggleable():
    """GUI 无 exit_flag 时的 DummyFlag 默认为假、可置真，布尔语义正确。"""
    flag = DummyFlag()
    assert flag.value is False
    assert bool(flag) is False
    flag.value = True
    assert bool(flag) is True


def test_stop_requested_helper_reads_value_based_flags():
    """_stop_requested：should_stop 优先，其次 flag.value，最后对象布尔值。"""
    def call(fake):
        return BasedProcessor._stop_requested(fake)

    assert call(SimpleNamespace(should_stop=False, global_exit_flag=DummyFlag())) is False
    assert call(SimpleNamespace(should_stop=False, global_exit_flag=SimpleNamespace(value=True))) is True
    assert call(SimpleNamespace(should_stop=False, global_exit_flag=True)) is True
    assert call(SimpleNamespace(should_stop=False, global_exit_flag=False)) is False
    assert call(SimpleNamespace(should_stop=True, global_exit_flag=DummyFlag())) is True
