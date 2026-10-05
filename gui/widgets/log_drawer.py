"""日志抽屉（C4）：无 ANSI 的 GuiLogHandler + queue.Queue + 主线程 QTimer 批量刷新。

每次任务 log_ready 后重新挂载（旧 handler 置 closed），避免重复挂载导致的
日志翻倍；GUI 线程通过 100ms 定时器批量取队列，避免逐条跨线程更新控件。
"""
from __future__ import annotations

import logging
import queue
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

QT_MSG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class GuiLogHandler(logging.Handler):
    """把 root logger 的记录送入线程安全队列（纯文本，无 ANSI 转义）。"""

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter(QT_MSG_FORMAT))
        self._queue: "queue.Queue[str]" = queue.Queue()
        self.closed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self.closed:
            return
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            pass  # 日志通道绝不抛异常干扰业务

    def close(self) -> None:  # noqa: A003
        self.closed = True
        super().close()

    def drain(self, limit: int = 500) -> list:
        lines = []
        for _ in range(limit):
            try:
                lines.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return lines


class LogDrawer(QWidget):
    """默认收起；展开后显示最近 2000 行日志。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._handler: Optional[GuiLogHandler] = None

        self.toggle_btn = QPushButton("日志 ▸")
        self.toggle_btn.setObjectName("logToggle")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.toggled.connect(self._on_toggled)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(2000)
        self.text_edit.setObjectName("logView")
        font = self.text_edit.document().defaultFont()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.text_edit.document().setDefaultFont(font)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 8, 0, 0)
        content_layout.addWidget(self.text_edit)
        self._content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.addWidget(self.toggle_btn)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self._content)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._pump)

    def mount(self) -> None:
        """log_ready 后调用：关闭旧 handler，向 root logger 挂新 handler。"""
        if self._handler is not None:
            self._handler.close()
            logging.getLogger().removeHandler(self._handler)
        self._handler = GuiLogHandler()
        logging.getLogger().addHandler(self._handler)
        if not self._timer.isActive():
            self._timer.start()

    def unmount(self) -> None:
        if self._handler is not None:
            self._handler.close()
            logging.getLogger().removeHandler(self._handler)
            self._handler = None
        self._timer.stop()

    def expand(self, expanded: bool) -> None:
        if self.toggle_btn.isChecked() != expanded:
            self.toggle_btn.setChecked(expanded)
        else:
            self._content.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self._content.isVisible()

    def clear(self) -> None:
        self.text_edit.clear()

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self.toggle_btn.setText("日志 ▾" if checked else "日志 ▸")

    def _pump(self) -> None:
        if self._handler is None:
            return
        lines = self._handler.drain()
        if not lines:
            return
        self.text_edit.appendPlainText("\n".join(lines))
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
