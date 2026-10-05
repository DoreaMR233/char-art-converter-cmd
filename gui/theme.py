"""
语义颜色 token → QSS 主题

设计规范（gui_refactor_plan.md §8）：
- 只允许在 QSS 中引用 token，禁止散落裸色值；
- 文字/背景对比度 ≥ 4.5:1（token 注释中标注实测值）；
- light / dark 两套主题成对设计。

对比度（WCAG 相对亮度法，实测值）：
- light: text #1f232e on surface #f5f6fa → 14.5:1
         text-secondary #5a5f72 on surface → 5.9:1
         primary-text #ffffff on primary #2f4fd0 → 6.6:1
         danger #c92a2a on surface → 5.0:1
- dark:  text #e7e9f4 on surface #1b1d27 → 13.9:1
         text-secondary #9ba0b8 on surface → 6.5:1
         primary-text #ffffff on primary #3d5bd9 → 5.7:1
         danger #ff8787 on surface → 7.3:1
"""
from __future__ import annotations

from typing import Dict

LIGHT_TOKENS: Dict[str, str] = {
    "surface": "#f5f6fa",            # 窗口背景
    "surface_alt": "#ffffff",        # 卡片/输入框背景
    "surface_hover": "#e9ecf4",
    "border": "#c9cfdd",
    "border_focus": "#2f4fd0",
    "text": "#1f232e",
    "text_secondary": "#5a5f72",
    "text_disabled": "#9aa0b2",
    "primary": "#2f4fd0",
    "primary_hover": "#3a5ce0",
    "primary_pressed": "#2743b8",
    "primary_disabled": "#aab6e8",
    "primary_text": "#ffffff",
    "danger": "#c92a2a",
    "danger_hover": "#e03131",
    "danger_bg": "#fdecec",
    "success": "#237032",
    "success_bg": "#e9f7ec",
    "warning": "#b0720f",
    "warning_bg": "#fff3d6",
    "progress_chunk": "#2f4fd0",
    "progress_track": "#dfe3ee",
    "scrollbar_handle": "#b8bfce",
    "scrollbar_hover": "#9aa3b8",
    "log_background": "#0f1220",
    "log_text": "#d8dcea",
}

DARK_TOKENS: Dict[str, str] = {
    "surface": "#1b1d27",
    "surface_alt": "#262836",
    "surface_hover": "#2f3244",
    "border": "#3d4152",
    "border_focus": "#5c7cfa",
    "text": "#e7e9f4",
    "text_secondary": "#9ba0b8",
    "text_disabled": "#5c6070",
    "primary": "#3d5bd9",
    "primary_hover": "#4d6ce6",
    "primary_pressed": "#3450c4",
    "primary_disabled": "#3a4466",
    "primary_text": "#ffffff",
    "danger": "#ff8787",
    "danger_hover": "#ffa8a8",
    "danger_bg": "#3a2325",
    "success": "#69db7c",
    "success_bg": "#1e3327",
    "warning": "#ffd43b",
    "warning_bg": "#332b17",
    "progress_chunk": "#5c7cfa",
    "progress_track": "#33364a",
    "scrollbar_handle": "#4a4f63",
    "scrollbar_hover": "#5c6280",
    "log_background": "#101220",
    "log_text": "#d8dcea",
}

TOKENS: Dict[str, Dict[str, str]] = {
    "light": LIGHT_TOKENS,
    "dark": DARK_TOKENS,
}

_QSS_TEMPLATE = """
QWidget {{
    background-color: {surface};
    color: {text};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background-color: {surface};
}}
QLabel {{
    background: transparent;
    color: {text};
}}
QLabel#stageLabel {{
    color: {text_secondary};
}}
QLabel#statusLabel[kind="success"] {{
    color: {success};
}}
QLabel#statusLabel[kind="error"] {{
    color: {danger};
}}
QLabel#statusLabel[kind="warning"] {{
    color: {warning};
}}
QLabel#statusLabel[kind="info"] {{
    color: {text_secondary};
}}
QLabel#fieldError {{
    color: {danger};
    font-size: 12px;
}}
QLabel#errorLabel {{
    color: {danger};
    background-color: {danger_bg};
    border: 1px solid {danger};
    border-radius: 4px;
    padding: 6px 8px;
}}
QLabel#successLabel {{
    color: {success};
    background-color: {success_bg};
    border: 1px solid {success};
    border-radius: 4px;
    padding: 6px 8px;
}}
QLabel#hintLabel {{
    color: {text_secondary};
    font-size: 12px;
}}
QFrame#card {{
    background-color: {surface_alt};
    border: 1px solid {border};
    border-radius: 8px;
}}
QGroupBox {{
    background-color: {surface_alt};
    border: 1px solid {border};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {text};
    background-color: transparent;
}}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {surface_alt};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: {primary};
    selection-color: {primary_text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {border_focus};
    padding: 4px 7px;
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QComboBox:disabled {{
    background-color: {surface};
    color: {text_disabled};
    border-color: {border};
}}
QLineEdit[invalid="true"] {{
    border: 1px solid {danger};
}}
QLineEdit[invalid="true"]:focus {{
    border: 2px solid {danger};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {text_secondary};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface_alt};
    color: {text};
    border: 1px solid {border};
    selection-background-color: {primary};
    selection-color: {primary_text};
    outline: 0;
}}
QCheckBox {{
    background: transparent;
    color: {text};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {border};
    border-radius: 3px;
    background-color: {surface_alt};
}}
QCheckBox::indicator:hover {{
    border-color: {primary};
}}
QCheckBox::indicator:checked {{
    background-color: {primary};
    border-color: {primary};
}}
QCheckBox::indicator:disabled {{
    background-color: {surface};
    border-color: {border};
}}
QRadioButton {{
    background: transparent;
    color: {text};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {border};
    border-radius: 8px;
    background-color: {surface_alt};
}}
QRadioButton::indicator:hover {{
    border-color: {primary};
}}
QRadioButton::indicator:checked {{
    background-color: {primary};
    border-color: {primary};
}}
QRadioButton::indicator:disabled {{
    background-color: {surface};
    border-color: {border};
}}
QPushButton {{
    background-color: {surface_alt};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 7px 16px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {surface_hover};
    border-color: {primary};
}}
QPushButton:pressed {{
    background-color: {primary};
    color: {primary_text};
}}
QPushButton:focus {{
    border: 2px solid {border_focus};
    padding: 6px 15px;
}}
QPushButton:disabled {{
    background-color: {surface};
    color: {text_disabled};
    border-color: {border};
}}
QPushButton#primaryButton {{
    background-color: {primary};
    color: {primary_text};
    border: 1px solid {primary};
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background-color: {primary_hover};
    border-color: {primary_hover};
}}
QPushButton#primaryButton:pressed {{
    background-color: {primary_pressed};
    border-color: {primary_pressed};
}}
QPushButton#primaryButton:disabled {{
    background-color: {primary_disabled};
    color: {primary_text};
    border-color: {primary_disabled};
}}
QPushButton#secondaryButton {{
    background-color: transparent;
    color: {text_secondary};
    border: 1px solid {border};
}}
QPushButton#secondaryButton:hover {{
    background-color: {surface_hover};
    border-color: {text_secondary};
    color: {text};
}}
QPushButton#secondaryButton:pressed {{
    background-color: {surface_hover};
    border-color: {text};
    color: {text};
}}
QPushButton#secondaryButton:disabled {{
    color: {text_disabled};
    border-color: {border};
}}
QPushButton#ghostButton, QPushButton#logToggle {{
    background-color: transparent;
    color: {text};
    border: 1px solid transparent;
}}
QPushButton#ghostButton:hover, QPushButton#logToggle:hover {{
    background-color: {surface_hover};
    border-color: {border};
}}
QPushButton#ghostButton:pressed, QPushButton#logToggle:checked {{
    background-color: {surface_hover};
}}
QPushButton#ghostButton:disabled, QPushButton#logToggle:disabled {{
    color: {text_disabled};
}}
QPushButton#primaryButton:focus, QPushButton#secondaryButton:focus,
QPushButton#ghostButton:focus, QPushButton#logToggle:focus {{
    border: 2px solid {border_focus};
    padding: 6px 15px;
}}
QProgressBar {{
    background-color: {progress_track};
    border: none;
    border-radius: 4px;
    min-height: 10px;
    max-height: 10px;
    text-align: center;
    color: {text};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {progress_chunk};
    border-radius: 4px;
}}
QPlainTextEdit#logView {{
    background-color: {log_background};
    color: {log_text};
    border: none;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {scrollbar_handle};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {scrollbar_hover};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {scrollbar_handle};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {scrollbar_hover};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
    width: 0;
}}
QToolTip {{
    background-color: {surface_alt};
    color: {text};
    border: 1px solid {border};
    padding: 4px 8px;
}}
"""


def build_qss(theme: str = "dark") -> str:
    """按语义 token 渲染 QSS。theme 取值 'light' / 'dark'。"""
    if theme not in TOKENS:
        raise ValueError(f"未知主题: {theme}（可选：light / dark）")
    return _QSS_TEMPLATE.format(**TOKENS[theme])


def contrast_ratio(fg: str, bg: str) -> float:
    """计算 WCAG 相对亮度对比度（设计规范 §8 自检函数）。"""
    def _lum(hex_color: str) -> float:
        c = hex_color.lstrip("#")
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

        def _lin(v: float) -> float:
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    l1, l2 = _lum(fg), _lum(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def self_check(min_ratio: float = 4.5) -> bool:
    """校验两套主题的文字/背景对比度（T-314 自检函数）。"""
    pairs = [
        ("text", "surface"),
        ("text_secondary", "surface"),
        ("text", "surface_alt"),
        ("primary_text", "primary"),
        ("danger", "surface"),
        ("danger", "danger_bg"),
        ("success", "success_bg"),
    ]
    ok = True
    for theme, tokens in TOKENS.items():
        for fg_key, bg_key in pairs:
            ratio = contrast_ratio(tokens[fg_key], tokens[bg_key])
            if ratio < min_ratio:
                ok = False
    return ok
