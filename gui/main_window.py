"""主窗口：表单→Namespace 映射、五态状态机、进度显示与日志抽屉集成。"""
from __future__ import annotations

import logging
import os
import subprocess
from argparse import Namespace
from enum import Enum, auto
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from src.configs.message_config import ERROR_MESSAGES
from src.enums.file_type import FileType
from gui.progress_bridge import ProgressBridge, phase_title
from gui.widgets.file_picker import FilePicker
from gui.widgets.log_drawer import LogDrawer
from gui.widgets.param_panel import ParamPanel, SIZE_MODE_CUSTOM
from gui.widgets.preview_pane import PreviewPane
from gui.worker import ConversionWorker, PREWARM_DESCRIPTION

logger = logging.getLogger(__name__)


class AppState(Enum):
    IDLE = auto()
    PREPARING = auto()
    RUNNING = auto()
    CANCELING = auto()
    SUCCESS = auto()
    ERROR = auto()


class MainWindow(QMainWindow):
    """960×640（最小 800×560），8px 间距网格。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("字符画转换器")
        self.resize(960, 640)
        self.setMinimumSize(800, 560)

        self._state = AppState.IDLE
        self._worker = None
        self._close_requested = False

        self.bridge = ProgressBridge(self)
        self.bridge.progress.connect(self._on_progress)

        self.file_picker = FilePicker()
        self.input_error = QLabel("")
        self.input_error.setObjectName("fieldError")
        self.input_error.setWordWrap(True)
        self.input_error.setVisible(False)

        self.param_panel = ParamPanel()
        self.preview_pane = PreviewPane()
        self.preview_pane.size_ready.connect(self.param_panel.set_original_size)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.file_picker)
        left_layout.addWidget(self.input_error)
        left_layout.addWidget(self.param_panel, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 16, 16, 16)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.preview_pane, 1)

        self.stage_label = QLabel("就绪")
        self.stage_label.setObjectName("stageLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # 初始不确定总线
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setObjectName("progressBar")

        # GIF/视频转换时的每线程进度条（position 1..N，与 CLI 主条+线程条布局一致）
        self.thread_bars_container = QWidget()
        self.thread_bars_layout = QVBoxLayout(self.thread_bars_container)
        self.thread_bars_layout.setContentsMargins(0, 0, 0, 0)
        self.thread_bars_layout.setSpacing(2)
        self.thread_bars_container.setVisible(False)
        self._thread_bars: List[QProgressBar] = []
        self._thread_count = 0

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        self.start_btn = QPushButton("开始转换")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)

        self.cancel_btn = QPushButton("取消转换")
        self.cancel_btn.setObjectName("secondaryButton")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setVisible(False)

        self.open_folder_btn = QPushButton("打开输出文件夹")
        self.open_folder_btn.setObjectName("ghostButton")
        self.open_folder_btn.setCursor(Qt.PointingHandCursor)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        self.open_folder_btn.setVisible(False)

        self.log_drawer = LogDrawer()

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(16, 0, 16, 8)
        bottom_layout.setSpacing(8)
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_row.addWidget(self.stage_label)
        progress_row.addWidget(self.progress_bar, 1)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        btn_row.addWidget(self.open_folder_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.start_btn)
        bottom_layout.addLayout(progress_row)
        bottom_layout.addWidget(self.thread_bars_container)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addLayout(btn_row)
        bottom_layout.addWidget(self.log_drawer)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(left, 1)
        body.addWidget(right, 1)
        central_layout.addLayout(body, 1)
        central_layout.addWidget(bottom)
        self.setCentralWidget(central)

        self.file_picker.pathChanged.connect(self._on_path_changed)
        self.param_panel.paramsChanged.connect(self._on_params_changed)
        self._set_state(AppState.IDLE)

    # ---------- 事件连接 ----------

    def _on_path_changed(self, path: str) -> None:
        self.input_error.setVisible(False)
        if path:
            self.preview_pane.set_input(path, self.param_panel.values()["density"])
        else:
            self.preview_pane.clear()
            self.param_panel.clear_original_size()

    def _on_params_changed(self) -> None:
        if self._state in (AppState.IDLE, AppState.SUCCESS, AppState.ERROR):
            self._set_state(AppState.IDLE)
        if self.file_picker.path():
            self.preview_pane.set_density(self.param_panel.values()["density"])

    # ---------- 状态机 ----------

    def _set_state(self, state: AppState) -> None:
        self._state = state
        running = state in (AppState.PREPARING, AppState.RUNNING, AppState.CANCELING)
        self.param_panel.set_enabled(not running)
        self.file_picker.setEnabled(not running)
        self.start_btn.setEnabled(state in (AppState.IDLE, AppState.SUCCESS, AppState.ERROR))
        self.cancel_btn.setVisible(running)
        self.cancel_btn.setEnabled(state in (AppState.PREPARING, AppState.RUNNING))
        self.cancel_btn.setText("正在取消…" if state == AppState.CANCELING else "取消转换")
        if state == AppState.SUCCESS:
            self.status_label.setProperty("kind", "success")
        elif state == AppState.ERROR:
            self.status_label.setProperty("kind", "error")
        elif state == AppState.CANCELING:
            self.status_label.setProperty("kind", "warning")
        else:
            self.status_label.setProperty("kind", "info")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    # ---------- 任务控制 ----------

    def _validate_form(self) -> bool:
        path = self.file_picker.path()
        if not path:
            self._show_input_error("请选择输入文件（图片或视频）")
            return False
        if not Path(path).is_file():
            self._show_input_error(ERROR_MESSAGES['file_not_found'].format(path))
            return False
        try:
            ft = FileType.from_path(Path(path))
        except ValueError as e:
            self._show_input_error(str(e))
            return False
        if ft in (FileType.TEXT, FileType.AUDIO):
            self._show_input_error(
                f"{ERROR_MESSAGES['unsupported_format'].format(Path(path).suffix)} 请选择图片或视频文件")
            return False

        if self.param_panel.size_mode == SIZE_MODE_CUSTOM:
            width = self.param_panel.width_spin.value()
            height = self.param_panel.height_spin.value()
            if bool(width) != bool(height):
                self.param_panel.size_hint.setProperty("invalid", True)
                self.param_panel.size_hint.setText(
                    "自定义尺寸需同时填写宽和高（或都留空），已忽略本次请求")
                self.param_panel.size_hint.style().unpolish(self.param_panel.size_hint)
                self.param_panel.size_hint.style().polish(self.param_panel.size_hint)
                return False
        self.param_panel.reset_size_hint()

        output = self.param_panel.values()["output"]
        self.param_panel.output_error.setVisible(False)
        if output:
            try:
                Path(output).mkdir(parents=True, exist_ok=True)
                if not os.access(output, os.W_OK):
                    raise PermissionError
            except PermissionError:
                self.param_panel.output_error.setText(
                    ERROR_MESSAGES['output_dir_not_writable'].format(output))
                self.param_panel.output_error.setVisible(True)
                return False
            except OSError as e:
                self.param_panel.output_error.setText(f"输出目录无效：{e}")
                self.param_panel.output_error.setVisible(True)
                return False
        return True

    def _show_input_error(self, message: str) -> None:
        self.file_picker.set_error(message)
        self.input_error.setText(message)
        self.input_error.setVisible(True)

    def _build_namespace(self) -> Namespace:
        values = self.param_panel.values()
        values["input"] = self.file_picker.path()
        return Namespace(**values)

    def _on_start(self) -> None:
        if not self._validate_form():
            self._set_state(AppState.ERROR)
            self.status_label.setText("表单校验未通过，请检查标红字段后重试。")
            return
        self._set_state(AppState.PREPARING)
        self.stage_label.setText(PREWARM_DESCRIPTION)
        self.progress_bar.setRange(0, 0)
        self._clear_thread_bars()
        self._thread_phase = False
        self.status_label.setText("正在准备转换任务…")
        self.open_folder_btn.setVisible(False)
        self.input_error.setVisible(False)
        self.param_panel.output_error.setVisible(False)

        self.bridge.attach()
        self._worker = ConversionWorker(self._build_namespace())
        self._worker.progress.connect(self.bridge.progress)
        self._worker.thread_count.connect(self._on_thread_count)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.log_ready.connect(self._on_log_ready)
        self._worker.start()
        self._set_state(AppState.RUNNING)

    def _on_cancel(self) -> None:
        if self._worker is not None and self._state in (AppState.PREPARING, AppState.RUNNING, AppState.CANCELING):
            self._set_state(AppState.CANCELING)
            self.status_label.setText("正在取消，等待当前帧处理完成…")
            self._worker.cancel()

    def _on_log_ready(self) -> None:
        self.log_drawer.mount()
        if self.param_panel.values()["debug"]:
            self.log_drawer.expand(True)

    # 帧处理阶段（CLI 中由各线程进度条表示）独占线程条
    _THREAD_BAR_PHASE = ("处理视频帧", "处理帧")

    def _on_progress(self, done: int, total: int, description: str, extra: str, position: int = 0) -> None:
        if position > 0:
            # 线程条（position 1..N）：与 CLI 每线程一条对齐，tooltip 显示当前活动
            self._thread_phase = True
            index = position - 1
            if 0 <= index < len(self._thread_bars):
                bar = self._thread_bars[index]
                if total <= 0:
                    bar.setRange(0, 0)
                else:
                    bar.setRange(0, total)
                    bar.setValue(min(done, total))
                bar.setToolTip(f"{description} {extra}".strip() if extra else description)
            return
        # 主条（position 0）：总体进度与阶段文案
        if total <= 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
        title = phase_title(description)
        if self._thread_phase and title not in self._THREAD_BAR_PHASE:
            # 帧处理阶段已结束（进入音频提取/合成/保存）：收起不再更新的线程条
            self._clear_thread_bars()
            self._thread_phase = False
        if self.status_label.text() == "正在准备转换任务…" and not (description or "").startswith("正在初始化"):
            # 首个真实工作事件：准备阶段结束
            self.status_label.setText("正在转换…")
        self.stage_label.setText(f"{title} {extra}".strip() if extra else title)

    def _on_thread_count(self, count: int) -> None:
        """GIF/视频任务开始时，按最大线程数重建线程进度条。"""
        self._clear_thread_bars()
        if count <= 0:
            return
        self._thread_count = count
        for _ in range(count):
            bar = QProgressBar()
            bar.setRange(0, 0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setObjectName("threadBar")
            self.thread_bars_layout.addWidget(bar)
            self._thread_bars.append(bar)
        self.thread_bars_container.setVisible(True)

    def _clear_thread_bars(self) -> None:
        """移除全部线程条并隐藏容器（任务开始/结束/取消时调用）。"""
        for bar in self._thread_bars:
            self.thread_bars_layout.removeWidget(bar)
            bar.deleteLater()
        self._thread_bars = []
        self._thread_count = 0
        self.thread_bars_container.setVisible(False)

    def _on_finished(self, code: int, payload: dict) -> None:
        self.bridge.flush()
        self.bridge.detach()
        self._clear_thread_bars()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

        if code == 0:
            self._set_state(AppState.SUCCESS)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.stage_label.setText("转换完成")
            output_dir = payload.get("output_dir")
            self.status_label.setText(
                f"转换成功。产物目录：{output_dir}" if output_dir else "转换成功。")
            self.open_folder_btn.setVisible(bool(output_dir))
            self._output_dir = output_dir
        elif code == 130:
            self._set_state(AppState.IDLE)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.stage_label.setText("已取消")
            self.status_label.setText("转换已取消，可重新开始。")
        else:
            self._set_state(AppState.ERROR)
            self.stage_label.setText("转换失败")
            message = payload.get("error", ERROR_MESSAGES['general_error'].format("未知错误"))
            hint = "请检查输入文件与参数后重试；若反复失败可开启调试日志查看详情。"
            self.status_label.setText(f"转换失败：{message}\n{hint}")
            if code == 2:
                self._show_input_error(message)
        if self._close_requested:
            self.close()

    def _open_output_folder(self) -> None:
        path = getattr(self, "_output_dir", None)
        if path and Path(path).exists():
            subprocess.Popen(["explorer", os.path.normpath(str(path))])

    # ---------- 窗口生命周期 ----------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._close_requested = True
            self._on_cancel()
            event.ignore()
            return
        self.log_drawer.unmount()
        self.bridge.detach()
        self.preview_pane.shutdown()
        super().closeEvent(event)
