"""GUI 应用装配：主题选择/应用、主窗口创建。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMainWindow

from gui.theme import build_qss, self_check


def resource_path(relative: str) -> str:
    """兼容 PyInstaller onefile（_MEIPASS）与源码运行的资源定位。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return str(base / relative)


def apply_theme(app: QApplication, dark: bool = False) -> None:
    theme_name = "dark" if dark else "light"
    if not self_check():
        raise RuntimeError("主题对比度自检未通过（WCAG AA 4.5:1）")
    app.setStyleSheet(build_qss(theme_name))


def create_app(argv: Optional[list] = None) -> tuple:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("char_art_converter_gui")
    app.setOrganizationName("char-art-converter")
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)

    from gui.main_window import MainWindow
    window = MainWindow()
    return app, window
