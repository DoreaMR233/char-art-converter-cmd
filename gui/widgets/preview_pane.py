"""预览窗格：原图缩略图 + 字符画文本预览（后台线程 CPU 生成，不阻塞 UI）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from src.configs.common_config import CHAR_DENSITY_CONFIG, DEFAULT_FONT_SIZE
from src.utils.char_art_utils import image_to_char_text, resize_image_for_chars
from src.utils.font_utils import load_font
from src.utils.image_utils import is_animated_image

logger = logging.getLogger(__name__)

PREVIEW_LIMIT = [120, 100]   # 预览用尺寸上限（宽, 高），保证 CPU 生成即时完成
THUMBNAIL_MAX = 480


class _PreviewWorker(QThread):
    """线程内完成 读取→缩放→CPU 字符映射。"""

    preview_ready = Signal(int, object, str, str)   # 代次, QImage, 文本, 提示
    preview_failed = Signal(int, str)               # 代次, 错误
    size_ready = Signal(int, int, int)              # 代次, 原图宽, 原图高

    def __init__(self, path: str, density: str, generation: int, parent=None):
        super().__init__(parent)
        self._path = path
        self._density = density
        self._generation = generation

    def run(self) -> None:
        try:
            font = load_font(DEFAULT_FONT_SIZE)
            path = Path(self._path)
            img, note = self._load_frame(path)
            self.size_ready.emit(self._generation, img.width, img.height)

            thumb = img.copy()
            thumb.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX))
            qimage = self._to_qimage(thumb)

            char_set = CHAR_DENSITY_CONFIG[self._density]
            resized = resize_image_for_chars(PREVIEW_LIMIT, img, None, True, DEFAULT_FONT_SIZE)
            text = image_to_char_text(
                resized, use_gpu=False, torch=None, device=None,
                char_set=char_set, char_count=len(char_set))
            self.preview_ready.emit(self._generation, qimage, text, note)
        except Exception as e:
            logger.warning("预览生成失败: %s", e)
            self.preview_failed.emit(self._generation, str(e))

    @staticmethod
    def _load_frame(path: Path):
        """读取预览帧：图片直接打开；视频用 cv2 取首帧（提供原图尺寸 + 预览）。"""
        try:
            with Image.open(path) as opened:
                note = "（动图：仅预览首帧）" if is_animated_image(path) else ""
                return opened.convert("RGB"), note
        except Exception:
            import cv2
            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"无法读取视频首帧: {path}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb), "（视频：预览首帧）"
            finally:
                cap.release()

    @staticmethod
    def _to_qimage(pil_img: Image.Image) -> QImage:
        data = pil_img.tobytes("raw", "RGB")
        qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3,
                        QImage.Format_RGB888)
        return qimage.copy()


class PreviewPane(QWidget):
    """右侧预览区，参数/文件变化时自动刷新。"""

    size_ready = Signal(int, int)   # 原图宽, 原图高

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[_PreviewWorker] = None
        self._path: Optional[str] = None
        self._density = "medium"
        self._generation = 0
        self._refresh_pending = False

        self.title = QLabel("预览")
        self.title.setObjectName("sectionTitle")
        self.refresh_btn = QPushButton("刷新预览")
        self.refresh_btn.setObjectName("ghostButton")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.refresh_btn)

        self.image_label = QLabel("选择文件后显示原图缩略图")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("previewImage")
        self.image_label.setMinimumHeight(160)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("字符画文本预览")
        self.text_edit.setObjectName("previewText")
        font = self.text_edit.document().defaultFont()
        font.setFamily("Consolas")
        font.setPointSize(8)
        self.text_edit.document().setDefaultFont(font)

        self.note_label = QLabel("")
        self.note_label.setObjectName("hintLabel")
        self.note_label.setWordWrap(True)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.image_label)
        splitter.addWidget(self.text_edit)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 240])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.note_label)

    def set_input(self, path: str, density: str) -> None:
        self._generation += 1
        self._path = path
        self._density = density
        self.refresh()

    def set_density(self, density: str) -> None:
        if self._density != density:
            self._generation += 1
        self._density = density
        if self._path:
            self.refresh()

    def clear(self) -> None:
        self._generation += 1
        self._path = None
        self.image_label.setText("选择文件后显示原图缩略图")
        self.text_edit.clear()
        self.note_label.setText("")

    def refresh(self) -> None:
        if not self._path or not Path(self._path).is_file():
            self.clear()
            return
        if self._worker is not None and self._worker.isRunning():
            self._refresh_pending = True
            return
        self._refresh_pending = False
        self.note_label.setText("预览生成中…")
        self._worker = _PreviewWorker(self._path, self._density, self._generation, self)
        self._worker.preview_ready.connect(self._on_ready)
        self._worker.preview_failed.connect(self._on_failed)
        self._worker.size_ready.connect(self._on_size_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self) -> None:
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _on_size_ready(self, generation: int, width: int, height: int) -> None:
        if generation != self._generation:
            return
        self.size_ready.emit(width, height)

    def shutdown(self) -> None:
        """窗口关闭前等待预览线程结束，避免 QThread 销毁崩溃。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)

    def _on_ready(self, generation: int, qimage: QImage, text: str, note: str) -> None:
        if generation != self._generation:
            return
        from PySide6.QtGui import QPixmap
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.text_edit.setPlainText(text)
        self.note_label.setText(note if note else "预览就绪")

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.note_label.setText(f"预览失败：{message}")
