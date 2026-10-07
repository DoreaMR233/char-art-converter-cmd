"""T-208~T-212：CLI 错误处理用例（docs/gui_test_plan.md §5）。

退出码约定：1=参数/校验错误（ValueError），2=输入文件不存在（FileNotFoundError）。
"""
import pytest


def _detail(res):
    return f"stdout:\n{res.stdout[-1500:]}\nstderr:\n{res.stderr[-1500:]}"


def _combined(res):
    return res.stdout + res.stderr


def test_missing_input_file(run_cli, tmp_path):
    """T-208：输入文件不存在 → rc2，输出含“文件”。"""
    res = run_cli([str(tmp_path / "no_such_file.png")])
    assert res.returncode == 2, _detail(res)
    assert "文件" in _combined(res)


def test_unsupported_format(run_cli, tmp_path):
    """T-209：扩展名不受支持的文件 → rc1，输出含“不支持”。

    说明：文档示例写 .txt，但 .txt 是 FileType.TEXT（src/enums/file_type.py:97），
    CLI main() 对非 IMAGE/VIDEO 类型静默返回 0（§7.1 亦确认该行为）。
    为使用例真正命中“不支持格式”路径，本用例改用 puremagic 可识别为 PDF 的文件；
    该文档/实现差异将写入对 Lead 的汇报。
    """
    unsupported = tmp_path / "unsupported.pdf"
    unsupported.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    res = run_cli([str(unsupported)])
    assert res.returncode == 1, _detail(res)
    combined = _combined(res)
    assert "不支持" in combined
    assert ".pdf" in combined


@pytest.mark.parametrize(
    "limit_args,expected_fragment",
    [
        (["32"], "--limit-size"),
        (["0", "10"], "宽度和高度必须大于0"),
        (["0", "0"], "宽度和高度必须大于0"),
    ],
)
def test_invalid_limit_size(run_cli, sample_image, limit_args, expected_fragment):
    """T-210：非法的 -l（单值 / 非正值）→ rc1，输出含 limit-size 提示。"""
    res = run_cli([str(sample_image), "-l", *limit_args])
    assert res.returncode == 1, _detail(res)
    assert expected_fragment in _combined(res)


def test_output_path_not_directory(run_cli, sample_image, tmp_path):
    """T-211：-o 指向已存在文件 → rc1，输出含“目录”提示。

    说明：文档建议“盘符根/只读目录”，但 Windows 下 os.access(W_OK) 对管理员/ACL
    场景不可靠；改用“-o 指向已存在文件”确定性触发输出目录校验失败，
    验证的是同一契约（非法输出路径 → rc1 + 目录提示）。
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("占位文件", encoding="utf-8")

    res = run_cli([str(sample_image), "-o", str(blocker)])
    assert res.returncode == 1, _detail(res)
    assert "目录" in _combined(res)


def test_debug_logging(run_cli, sample_image, tmp_path):
    """T-212：成功用例 + --debug → rc0，输出含 DEBUG 与 INFO 日志行。"""
    out = tmp_path / "out"
    res = run_cli([str(sample_image), "-t", "-o", str(out), "--debug"])
    assert res.returncode == 0, _detail(res)
    combined = _combined(res)
    assert "DEBUG" in combined, "缺少 DEBUG 日志行"
    assert "INFO" in combined, "缺少 INFO 日志行"
