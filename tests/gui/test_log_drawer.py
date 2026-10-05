"""T-311 / T-312：日志抽屉（GuiLogHandler 无 ANSI、mount 幂等、定时泵取）。"""
import logging

from PySide6.QtWidgets import QPlainTextEdit

from gui.widgets.log_drawer import GuiLogHandler


def test_gui_log_handler_plain_format():
    """T-311：GuiLogHandler 输出无 ANSI 颜色码，保留级别与消息。"""
    handler = GuiLogHandler()
    record = logging.LogRecord("test.gui", logging.INFO, __file__, 1, "普通消息", None, None)
    handler.emit(record)
    text = "\n".join(handler.drain())
    assert "\x1b" not in text, "日志抽屉 Formatter 不应携带 ANSI 颜色码"
    assert "[INFO]" in text
    assert "普通消息" in text
    handler.close()


def test_gui_log_handler_ignores_after_close():
    """T-311 附：close 后 emit 为空操作。"""
    handler = GuiLogHandler()
    handler.close()
    handler.emit(logging.LogRecord("test.gui", logging.INFO, __file__, 1, "after-close", None, None))
    assert handler.drain() == []


def test_log_drawer_mount_idempotent(window):
    """T-312：重复 mount 只保留一个 GuiLogHandler，unmount 完全移除。"""
    drawer = window.log_drawer
    drawer.mount()
    drawer.mount()
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, GuiLogHandler)]
    assert len(handlers) == 1

    drawer.unmount()
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, GuiLogHandler)]
    assert handlers == []


def test_log_drawer_pumps_records(window, qt_wait):
    """T-312：挂载后日志经 100ms QTimer 泵入 logView。"""
    window.log_drawer.mount()
    logging.getLogger("test.gui").info("hello-drawer-12345")

    log_view = window.findChild(QPlainTextEdit, "logView")
    assert qt_wait(lambda: "hello-drawer-12345" in log_view.toPlainText(), timeout=10), \
        "日志未在 10s 内泵入 logView"


def test_log_drawer_toggle_text(window):
    """T-312 附：展开/收起时 toggle 按钮文案切换。"""
    drawer = window.log_drawer
    drawer.expand(True)
    assert drawer.toggle_btn.text() == "日志 ▾"
    drawer.expand(False)
    assert drawer.toggle_btn.text() == "日志 ▸"
