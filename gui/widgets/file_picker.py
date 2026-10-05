"""文件选择行：只读输入框 + 浏览按钮 + 拖放支持。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from src.configs.image_config import SUPPORTED_IMAGE_FORMATS
from src.configs.video_config import SUPPORTED_VIDEO_FORMATS

IMAGE_FILTER = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_FORMATS)
VIDEO_FILTER = " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_FORMATS)


class FilePicker(QWidget):
    """单选文件输入，接受图片/视频，支持拖放。"""

    pathChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.label = QLabel("输入文件")
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.setPlaceholderText("选择或拖入图片/视频文件")
        self.edit.setProperty("class", "filePath")
        self.browse_btn = QPushButton("浏览…")
        self.browse_btn.setObjectName("browseButton")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._browse)
        self.edit.textChanged.connect(self.pathChanged)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_btn)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str) -> None:
        self.edit.setText(path)

    def set_error(self, message: str) -> None:
        """错误就近显示：输入框红框 + tooltip 文案（颜色非唯一载体）。"""
        self.edit.setProperty("invalid", bool(message))
        self.edit.setToolTip(message if message else "")
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)

    def clear_error(self) -> None:
        self.set_error("")

    def _browse(self) -> None:
        filter_str = (
            f"支持的图片与视频 ({IMAGE_FILTER} {VIDEO_FILTER});;"
            f"图片文件 ({IMAGE_FILTER});;视频文件 ({VIDEO_FILTER})"
        )
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", filter_str)
        if path:
            self.set_path(path)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._dragged_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        path = self._dragged_file(event)
        if path:
            self.set_path(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _dragged_file(event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls() or len(mime.urls()) != 1:
            return None
        path = Path(mime.urls()[0].toLocalFile())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_FORMATS + SUPPORTED_VIDEO_FORMATS:
            return str(path)
        return None
