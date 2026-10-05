"""T-110 / T-111：GUI 取消路径同时置位双停止标志（gui\\worker.py:73-83，改造 C3）。

cancel 为纯 Python 方法，直接以未绑定方式调用，避免为标志位断言启动 QThread。
"""
from types import SimpleNamespace

from gui.worker import ConversionWorker


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
