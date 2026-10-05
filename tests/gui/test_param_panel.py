"""参数面板交互回归：上/下键步进、自定义尺寸原图预填、GPU 滑条映射。"""
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def _click_and_type(spin, digits):
    edit = spin.lineEdit()
    QTest.mouseClick(edit, Qt.LeftButton, pos=edit.rect().center())
    QApplication.processEvents()
    for ch in digits:
        QTest.keyClick(edit, ch)
        QTest.qWait(30)
    QApplication.processEvents()


def test_width_spin_stepping_after_typing(window):
    """输入数字后按上/下键：从键入值步进（曾在焦点全选定时器下被吞）。"""
    panel = window.param_panel
    panel.size_custom_radio.setChecked(True)
    spin = panel.width_spin
    _click_and_type(spin, "320")
    assert spin.lineEdit().hasAcceptableInput()

    QTest.keyClick(spin.lineEdit(), Qt.Key_Up)
    QApplication.processEvents()
    assert spin.value() == 321

    QTest.keyClick(spin.lineEdit(), Qt.Key_Down)
    QApplication.processEvents()
    assert spin.value() == 320


def test_gpu_slider_mapping_and_label(window):
    """GPU 滑条：默认勾选时禁用；取消后 5–95 步长 5，标签同步，映射为占比。"""
    panel = window.param_panel
    assert panel.gpu_use_default_check.isChecked()
    assert not panel.gpu_slider.isEnabled()
    assert panel.values()["gpu_memory_limit"] is None

    panel.gpu_use_default_check.setChecked(False)
    assert panel.gpu_slider.isEnabled()
    assert panel.gpu_slider.minimum() == 5
    assert panel.gpu_slider.maximum() == 95
    assert panel.gpu_slider.singleStep() == 5
    panel.gpu_slider.setValue(50)
    assert panel.gpu_value_label.text() == "50%"
    assert panel.values()["gpu_memory_limit"] == 0.5


def test_custom_size_prefills_original_size(window):
    """自定义尺寸默认值 = 原图尺寸；已输入时不被新尺寸覆盖。"""
    panel = window.param_panel
    panel.set_original_size(640, 480)
    panel.size_custom_radio.setChecked(True)
    assert panel.width_spin.value() == 640
    assert panel.height_spin.value() == 480
    assert panel.values()["limit_size"] == [640, 480]

    panel.width_spin.setValue(320)
    panel.height_spin.setValue(240)
    panel.set_original_size(800, 600)
    assert panel.width_spin.value() == 320
    assert panel.height_spin.value() == 240


def test_original_size_ignored_outside_custom_mode(window):
    """原图/默认大小模式下 set_original_size 不改变宽高框内容。"""
    panel = window.param_panel
    panel.set_original_size(640, 480)
    assert panel.size_original_radio.isChecked()
    assert panel.width_spin.value() == 0
    panel.size_default_radio.setChecked(True)
    assert panel.width_spin.value() == 0
