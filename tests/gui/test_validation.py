"""T-304：表单校验（gui\\main_window.py:_validate_form）。"""
import pytest


def test_missing_input(window):
    """空路径 → 输入错误提示。"""
    assert not window._validate_form()
    assert window.input_error.isVisible()
    assert "请选择输入文件（图片或视频）" in window.input_error.text()


def test_nonexistent_file(window, tmp_path):
    """路径不存在 → file_not_found 提示。"""
    window.file_picker.set_path(str(tmp_path / "missing.png"))
    assert not window._validate_form()
    assert "文件" in window.input_error.text()


def test_unsupported_extension(window, tmp_path):
    """纯 PDF 字节文件 → 不支持格式提示。"""
    pdf = tmp_path / "unsupported.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%PDF content\n%%EOF\n")
    window.file_picker.set_path(str(pdf))
    assert not window._validate_form()
    assert "不支持" in window.input_error.text()


def test_text_type_rejected(window, unsupported_file):
    """.txt 被识别为 TEXT 类型 → 提示仅支持图片或视频。"""
    window.file_picker.set_path(str(unsupported_file))
    assert not window._validate_form()
    assert "请选择图片或视频文件" in window.input_error.text()


def test_limit_size_xor(window, sample_image):
    """自定义尺寸下宽 xor 高单边填写 → 尺寸提示并拒绝。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.width_spin.setValue(32)
    window.param_panel.height_spin.setValue(0)
    assert not window._validate_form()
    assert "需同时填写宽和高" in window.param_panel.size_hint.text()
    assert window.param_panel.size_hint.property("invalid") is True


def test_limit_size_zero_rejected(window, sample_image):
    """自定义尺寸 0×0 → 尺寸提示并拒绝（与 CLI 一致：宽高必须大于 0）。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.clear_original_size()
    assert window.param_panel.width_spin.value() == 0
    assert window.param_panel.height_spin.value() == 0

    assert not window._validate_form()
    hint = window.param_panel.size_hint.text()
    assert "需同时填写宽和高" in hint
    assert "0×0" in hint
    assert window.param_panel.size_hint.property("invalid") is True


def test_limit_size_zero_without_input_file(window):
    """未选择输入文件 + 自定义尺寸 0×0 → 提交时给出尺寸错误提示。"""
    assert window.param_panel.width_spin.text() == "0"
    window.param_panel.size_custom_radio.setChecked(True)
    assert not window._validate_form()
    assert "需同时填写宽和高" in window.param_panel.size_hint.text()
    assert window.param_panel.size_hint.property("invalid") is True


def test_limit_size_restored_after_fix(window, sample_image):
    """自定义尺寸下补全宽高后提示复位、校验通过。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.width_spin.setValue(32)
    window.param_panel.height_spin.setValue(24)
    assert window._validate_form()
    assert window.param_panel.size_hint.property("invalid") is False


def test_non_custom_size_mode_skips_size_check(window, sample_image):
    """原图尺寸/默认大小模式下宽高不参与校验（宽高禁用，残留值被忽略）。"""
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.width_spin.setValue(32)
    window.param_panel.size_original_radio.setChecked(True)
    assert window._validate_form()
    assert window.param_panel.size_hint.property("invalid") is False
    assert window._build_namespace().limit_size is None


def test_output_path_is_existing_file(window, sample_image, tmp_path):
    """输出路径为已存在文件 → mkdir 失败提示“输出目录无效”。"""
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("x", encoding="utf-8")
    window.file_picker.set_path(str(sample_image))
    window.param_panel.output_edit.setText(str(blocker))
    assert not window._validate_form()
    assert window.param_panel.output_error.isVisible()
    assert "输出目录无效" in window.param_panel.output_error.text()


def test_valid_form(window, sample_image, tmp_path):
    """合法输入 + 自定义尺寸宽高 + 输出目录 → 校验通过并创建目录。"""
    out_dir = tmp_path / "outdir"
    window.file_picker.set_path(str(sample_image))
    window.param_panel.size_custom_radio.setChecked(True)
    window.param_panel.width_spin.setValue(32)
    window.param_panel.height_spin.setValue(24)
    window.param_panel.output_edit.setText(str(out_dir))
    assert window._validate_form()
    assert out_dir.is_dir()
