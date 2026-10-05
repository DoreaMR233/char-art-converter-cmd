"""T-108 / T-109：GUI 参数 Namespace 装配（docs/gui_refactor_plan.md §7.1）。"""
from argparse import Namespace


def test_gui_namespace_factory_defaults(gui_namespace):
    """T-108：工厂默认值与文档 §7.1 / CLI create_parser 默认一致。"""
    ns = gui_namespace()
    assert isinstance(ns, Namespace)
    assert ns.input is None
    assert ns.output is None
    assert ns.density == "medium"
    assert ns.color_mode == "color"
    assert ns.limit_size is None
    assert ns.with_text is False
    assert ns.with_image is False
    assert ns.no_multithread is False
    assert ns.enable_gpu is True
    assert ns.gpu_memory_limit is None
    assert ns.debug is False


def test_gui_namespace_factory_overrides(gui_namespace):
    """T-108 附：显式覆盖项生效。"""
    ns = gui_namespace(
        "in.png", limit_size=[32, 24], with_text=True, enable_gpu=False, gpu_memory_limit=0.5
    )
    assert ns.input == "in.png"
    assert ns.limit_size == [32, 24]
    assert ns.with_text is True
    assert ns.enable_gpu is False
    assert ns.gpu_memory_limit == 0.5


def test_main_window_build_namespace_defaults(window):
    """T-109：MainWindow._build_namespace 的 input 来自 file_picker，其余来自 ParamPanel 默认值。"""
    window.file_picker.set_path("demo.png")
    ns = window._build_namespace()
    assert ns.input == "demo.png"
    assert ns.output is None
    assert ns.density == "medium"
    assert ns.color_mode == "color"
    assert ns.limit_size is None
    assert ns.with_text is False
    assert ns.with_image is False
    assert ns.no_multithread is False
    assert ns.enable_gpu is True
    assert ns.gpu_memory_limit is None
    assert ns.debug is False
