"""参数面板：密度/颜色模式/尺寸限制/输出选项/GPU/调试 + 输出目录。"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from src.configs.common_config import DEFAULT_COLOR_MODE, DEFAULT_DENSITY

DENSITY_OPTIONS = [
    ("low", "低（10 级）"),
    ("medium", "中（16 级）"),
    ("high", "高（70 级）"),
]
COLOR_MODE_OPTIONS = [
    ("grayscale", "灰度"),
    ("color", "彩色"),
    ("colorBackground", "彩色（深底）"),
]
SIZE_MODE_ORIGINAL = "original"
SIZE_MODE_DEFAULT = "default"
SIZE_MODE_CUSTOM = "custom"
SIZE_HINTS = {
    SIZE_MODE_ORIGINAL: "不限制字符网格，按原图尺寸处理（等价于不传 -l）",
    SIZE_MODE_DEFAULT: "默认大小：网格列数 = 原图宽 ÷ 6、行数 = 原图高 ÷ 6（内置字体大小 12）",
    SIZE_MODE_CUSTOM: "字符网格列数 × 行数（选中时自动填入原图尺寸，即不缩放）；都留空时按默认大小处理",
}


class _SelectAllLineEdit(QLineEdit):
    """点击/聚焦即全选，替代 QAbstractSpinBox 默认“折叠成光标”的行为。

    同步实现（无 QTimer.singleShot 竞态）：用户在“自动”等占位文本上打字时，
    第一个按键就替换全选内容，不会被逐个吞掉；步进前文本状态始终可预期。
    """

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.selectAll()

    def focusInEvent(self, event) -> None:  # noqa: N802
        super().focusInEvent(event)
        self.selectAll()


class ClearableSpinBox(QSpinBox):
    """尺寸输入框：清空文本 = 0（显示“自动”）。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setRange(0, 99999)
        self.setSpecialValueText("自动")
        self.setKeyboardTracking(False)
        self.setLineEdit(_SelectAllLineEdit(self))

    def validate(self, text: str, pos: int):
        """空文本视为合法（含义同 0）。"""
        if not text.strip():
            return QValidator.State.Acceptable, "", pos
        return super().validate(text, pos)

    def valueFromText(self, text: str) -> int:
        if not text.strip():
            return 0
        return super().valueFromText(text)

    def stepBy(self, steps: int) -> None:
        """中间文本非法（如超上限）时先恢复最近合法值，保证上/下键始终可用。"""
        if not self.lineEdit().hasAcceptableInput():
            self.interpretText()
        super().stepBy(steps)


class ParamPanel(QWidget):
    """左侧参数表单。值变化时发出 paramsChanged（主窗口可据此做轻量校验）。"""

    paramsChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._panel_enabled = True
        self._original_size: Optional[tuple] = None

        self.density_combo = QComboBox()
        for value, label in DENSITY_OPTIONS:
            self.density_combo.addItem(label, value)
        self.density_combo.setCurrentIndex(
            next(i for i, (v, _) in enumerate(DENSITY_OPTIONS) if v == DEFAULT_DENSITY))

        self.color_combo = QComboBox()
        for value, label in COLOR_MODE_OPTIONS:
            self.color_combo.addItem(label, value)
        self.color_combo.setCurrentIndex(
            next(i for i, (v, _) in enumerate(COLOR_MODE_OPTIONS) if v == DEFAULT_COLOR_MODE.value))

        self.size_original_radio = QRadioButton("原图尺寸")
        self.size_default_radio = QRadioButton("默认大小")
        self.size_custom_radio = QRadioButton("自定义尺寸")
        self.size_original_radio.setChecked(True)
        self.size_mode_group = QButtonGroup(self)
        for radio in (self.size_original_radio, self.size_default_radio, self.size_custom_radio):
            self.size_mode_group.addButton(radio)
        size_mode_row = QHBoxLayout()
        size_mode_row.setContentsMargins(0, 0, 0, 0)
        size_mode_row.setSpacing(8)
        for radio in (self.size_original_radio, self.size_default_radio, self.size_custom_radio):
            size_mode_row.addWidget(radio)
        size_mode_row.addStretch(1)

        self.width_spin = ClearableSpinBox()
        self.height_spin = ClearableSpinBox()
        self.width_spin.setEnabled(False)
        self.height_spin.setEnabled(False)
        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(8)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        self.size_hint = QLabel(SIZE_HINTS[SIZE_MODE_ORIGINAL])
        self.size_hint.setObjectName("hintLabel")

        self.with_text_check = QCheckBox("同时保存字符画文本 (.txt)")
        self.with_image_check = QCheckBox("同时保存字符画图像")
        self.no_multithread_check = QCheckBox("禁用多线程处理")
        self.enable_gpu_check = QCheckBox("启用 GPU 加速")
        self.enable_gpu_check.setChecked(True)

        self.gpu_use_default_check = QCheckBox("使用默认值（配置文件 80%）")
        self.gpu_use_default_check.setChecked(True)
        self.gpu_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_slider.setRange(5, 95)
        self.gpu_slider.setSingleStep(5)
        self.gpu_slider.setPageStep(10)
        self.gpu_slider.setValue(80)
        self.gpu_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gpu_slider.setTickInterval(5)
        self.gpu_slider.setEnabled(False)
        self.gpu_value_label = QLabel("80%")
        self.gpu_value_label.setObjectName("gpuValueLabel")
        gpu_row = QHBoxLayout()
        gpu_row.setContentsMargins(0, 0, 0, 0)
        gpu_row.setSpacing(8)
        gpu_row.addWidget(self.gpu_use_default_check)
        gpu_row.addWidget(self.gpu_slider, 1)
        gpu_row.addWidget(self.gpu_value_label)
        self.gpu_hint = QLabel("默认值来自配置文件（80%）；取消勾选后可用滑条选择 5%–95%")
        self.gpu_hint.setObjectName("hintLabel")

        self.debug_check = QCheckBox("调试日志（自动展开日志抽屉）")

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("留空 = 输入文件旁自动生成目录")
        self.output_browse = QPushButton("浏览…")
        self.output_browse.setObjectName("browseButton")
        self.output_browse.setCursor(Qt.PointingHandCursor)
        self.output_browse.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(8)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_browse)
        self.output_error = QLabel("")
        self.output_error.setObjectName("fieldError")
        self.output_error.setWordWrap(True)
        self.output_error.setVisible(False)

        form = QFormLayout()
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.addRow("字符密度", self.density_combo)
        form.addRow("颜色模式", self.color_combo)
        form.addRow("尺寸限制", size_mode_row)
        form.addRow("", size_row)
        form.addRow("", self.size_hint)
        form.addRow("GPU 显存", gpu_row)
        form.addRow("", self.gpu_hint)
        form.addRow("输出目录", output_row)
        form.addRow("", self.output_error)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(form)
        layout.addWidget(self.with_text_check)
        layout.addWidget(self.with_image_check)
        layout.addWidget(self.enable_gpu_check)
        layout.addWidget(self.no_multithread_check)
        layout.addWidget(self.debug_check)
        layout.addStretch(1)

        for w in (self.density_combo, self.color_combo):
            w.currentIndexChanged.connect(self.paramsChanged)
        for w in (self.width_spin, self.height_spin):
            w.valueChanged.connect(self.paramsChanged)
        self.gpu_slider.valueChanged.connect(self._on_gpu_slider_changed)
        self.output_edit.textChanged.connect(self.paramsChanged)
        for cb in (self.with_text_check, self.with_image_check,
                   self.no_multithread_check, self.enable_gpu_check, self.debug_check):
            cb.toggled.connect(self.paramsChanged)
        self.size_mode_group.buttonToggled.connect(self._on_size_mode_toggled)
        self.gpu_use_default_check.toggled.connect(self._on_gpu_default_toggled)

    def values(self) -> dict:
        """收集为 worker 所需的 Namespace 字段（§7.1 映射表）。"""
        mode = self.size_mode
        if mode == SIZE_MODE_ORIGINAL:
            limit_size: Optional[list] = None
        elif mode == SIZE_MODE_DEFAULT:
            limit_size = []
        else:
            width = self.width_spin.value()
            height = self.height_spin.value()
            if width and height:
                limit_size = [width, height]
            else:
                limit_size = []  # 只填一维视为无效，交由校验报错
        if self.gpu_use_default_check.isChecked():
            gpu_memory_limit = None
        else:
            gpu_memory_limit = round(self.gpu_slider.value() / 100, 2)
        return {
            "density": self.density_combo.currentData(),
            "color_mode": self.color_combo.currentData(),
            "limit_size": limit_size,
            "with_text": self.with_text_check.isChecked(),
            "with_image": self.with_image_check.isChecked(),
            "no_multithread": self.no_multithread_check.isChecked(),
            "enable_gpu": self.enable_gpu_check.isChecked(),
            "gpu_memory_limit": gpu_memory_limit,
            "debug": self.debug_check.isChecked(),
            "output": self.output_edit.text().strip() or None,
        }

    @property
    def size_mode(self) -> str:
        """当前尺寸限制模式：original / default / custom。"""
        if self.size_default_radio.isChecked():
            return SIZE_MODE_DEFAULT
        if self.size_custom_radio.isChecked():
            return SIZE_MODE_CUSTOM
        return SIZE_MODE_ORIGINAL

    def reset_size_hint(self) -> None:
        """清除校验错误标记，恢复当前模式的提示文案。"""
        self.size_hint.setProperty("invalid", False)
        self.size_hint.setText(SIZE_HINTS[self.size_mode])
        self.size_hint.style().unpolish(self.size_hint)
        self.size_hint.style().polish(self.size_hint)

    def _on_size_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._apply_conditional_enabled()
        self._fill_custom_with_original()
        self.reset_size_hint()
        self.paramsChanged.emit()

    def _on_gpu_default_toggled(self, checked: bool) -> None:
        self._apply_conditional_enabled()
        self.paramsChanged.emit()

    def _on_gpu_slider_changed(self, value: int) -> None:
        self.gpu_value_label.setText(f"{value}%")
        self.paramsChanged.emit()

    def set_original_size(self, width: int, height: int) -> None:
        """记录输入文件的原始尺寸；自定义尺寸模式下空值时自动填入。"""
        if width is None or height is None:
            return
        self._original_size = (int(width), int(height))
        self._fill_custom_with_original()

    def clear_original_size(self) -> None:
        """文件清空后丢弃旧原图尺寸，避免自定义模式误填旧值。"""
        self._original_size = None

    def _fill_custom_with_original(self) -> None:
        if not self._original_size or not self.size_custom_radio.isChecked():
            return
        if self.width_spin.value() == 0 and self.height_spin.value() == 0:
            self.width_spin.setValue(self._original_size[0])
            self.height_spin.setValue(self._original_size[1])

    def _apply_conditional_enabled(self) -> None:
        """按当前选择重设宽高/GPU 滑条可用性（仅在面板整体启用时）。"""
        if not getattr(self, "_panel_enabled", True):
            return
        custom = self.size_custom_radio.isChecked()
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)
        self.gpu_slider.setEnabled(not self.gpu_use_default_check.isChecked())

    def set_enabled(self, enabled: bool) -> None:
        """RUNNING 时整体禁用参数区；恢复时按当前模式重设子项可用性。"""
        self._panel_enabled = enabled
        for w in self.findChildren(QWidget):
            if w not in (self.size_hint, self.gpu_hint):
                w.setEnabled(enabled)
        if enabled:
            self._apply_conditional_enabled()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text() or "")
        if path:
            self.output_edit.setText(path)
