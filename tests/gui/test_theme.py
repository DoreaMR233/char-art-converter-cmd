"""T-314 / T-315：主题自检与 GUI/CLI 参数映射一致性。"""
import json
import sys

import pytest

from gui.theme import build_qss, contrast_ratio, self_check


def test_build_qss_light_and_dark():
    """T-314a：明暗两套 QSS 均非空且内容不同。"""
    light = build_qss("light")
    dark = build_qss("dark")
    assert light and dark
    assert "QWidget" in light
    assert "QWidget" in dark
    assert light != dark


def test_build_qss_unknown_theme_raises():
    """T-314b：未知主题抛 ValueError。"""
    with pytest.raises(ValueError):
        build_qss("hotdog")


def test_contrast_ratio_wcag():
    """T-314c：对比度计算符合 WCAG 公式。"""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.1)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)


def test_theme_self_check_passes():
    """T-314d：双主题 7 组关键对比对满足 WCAG AA 4.5:1。"""
    assert self_check(min_ratio=4.5) is True


def test_apply_theme_stylesheet(qapp):
    """T-314e：apply_theme 通过自检并设置样式表。"""
    from gui.app import apply_theme

    apply_theme(qapp, dark=False)
    assert qapp.styleSheet()
    apply_theme(qapp, dark=True)
    assert qapp.styleSheet()


def _cli_namespace(project_root, cli_args):
    """进程内解析 CLI 默认值（不启动子进程）。

    原因：本进程已加载 Qt/线程（GUI worker、tqdm monitor），此时再 subprocess
    会命中 CPython 3.12 Windows 的句柄/线程竞态（实测 access violation，
    subprocess.py:1626 _communicate），导致整个 pytest 进程崩溃。
    """
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from char_art_converter import create_parser

    ns = create_parser().parse_args(cli_args)
    data = vars(ns)
    return {k: (v.value if hasattr(v, "value") else v) for k, v in data.items()}


def test_gui_cli_namespace_parity(window, sample_image, project_root):
    """T-315：GUI 默认参数与 CLI create_parser 默认值一致（gpu_memory_limit 均为 None）。"""
    window.file_picker.set_path(str(sample_image))
    gui = window._build_namespace()
    cli = _cli_namespace(project_root, ["dummy.png"])

    assert gui.density == cli["density"]
    assert gui.color_mode == cli["color_mode"]
    assert gui.limit_size == cli["limit_size"]
    assert gui.with_text == cli["with_text"]
    assert gui.with_image == cli["with_image"]
    assert gui.no_multithread == cli["no_multithread"]
    assert gui.enable_gpu == cli["enable_gpu"]
    assert gui.debug == cli["debug"]
    assert gui.output == cli["output"]
    # CLI 未显式传参时 gpu_memory_limit=None（使用配置默认 0.8 占比）；GUI 默认勾选“使用默认值”同为 None
    assert cli["gpu_memory_limit"] is None
    assert gui.gpu_memory_limit is None


def test_gpu_memory_limit_unit_mapping(window, sample_image, project_root):
    """T-315 附：GUI 输入整数百分比（5–95，/100 得占比）；CLI 为整数 MB 语义。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.gpu_use_default_check.setChecked(False)
    window.param_panel.gpu_slider.setValue(35)
    gui = window._build_namespace()
    cli = _cli_namespace(project_root, ["dummy.png", "--gpu-memory-limit", "35"])

    assert round(gui.gpu_memory_limit * 100) == cli["gpu_memory_limit"]


def test_limit_size_mapping(window, sample_image, project_root):
    """T-315 附：宽高 → limit_size 与 CLI --limit-size 一致。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.width_spin.setValue(32)
    window.param_panel.height_spin.setValue(24)
    gui = window._build_namespace()
    cli = _cli_namespace(project_root, ["dummy.png", "--limit-size", "32", "24"])

    assert gui.limit_size == cli["limit_size"] == [32, 24]


def test_size_mode_mapping(window):
    """T-315 附：尺寸单选组三态映射，宽高仅在“自定义尺寸”可编辑。"""
    panel = window.param_panel
    assert panel.size_mode == "original"
    assert panel.values()["limit_size"] is None
    assert not panel.width_spin.isEnabled() and not panel.height_spin.isEnabled()

    panel.size_default_radio.setChecked(True)
    assert panel.values()["limit_size"] == []
    assert not panel.width_spin.isEnabled()

    panel.size_custom_radio.setChecked(True)
    assert panel.width_spin.isEnabled() and panel.height_spin.isEnabled()
    panel.width_spin.setValue(32)
    panel.height_spin.setValue(24)
    assert panel.values()["limit_size"] == [32, 24]


def test_gpu_default_switch_mapping(window):
    """T-315 附：GPU 默认开关——勾选→None（内核用配置 80%），取消勾选→百分比。"""
    panel = window.param_panel
    assert panel.gpu_use_default_check.isChecked()
    assert panel.values()["gpu_memory_limit"] is None
    assert not panel.gpu_slider.isEnabled()

    panel.gpu_use_default_check.setChecked(False)
    assert panel.gpu_slider.isEnabled()
    assert panel.values()["gpu_memory_limit"] == 0.8
    panel.gpu_slider.setValue(35)
    assert panel.values()["gpu_memory_limit"] == 0.35
