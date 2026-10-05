"""字符画转换器 GUI 入口。

注意：此文件顶部必须保留 multiprocessing.freeze_support()（§9），
且绝不能 import char_art_converter.py（其模块级 SIGINT 注册会劫持 GUI 进程）。
"""
from __future__ import annotations

import multiprocessing
import sys

multiprocessing.freeze_support()


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from gui.app import apply_theme, create_app

    app, window = create_app()
    apply_theme(app, dark=False)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
