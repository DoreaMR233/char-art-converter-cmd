"""T-401 / T-402：构建产物冒烟（标记 slow，发版前执行）。"""
import subprocess
import sys

import pytest

PROJECT_ROOT = None
DIST_DIR = None


@pytest.fixture(scope="module")
def roots(project_root):
    global PROJECT_ROOT, DIST_DIR
    PROJECT_ROOT = project_root
    DIST_DIR = project_root / "dist"
    return project_root, DIST_DIR


@pytest.mark.slow
def test_cli_exe_build_artifact(roots):
    """T-401：dist\\char_art_converter.exe 存在且 --version 返回 1.0.0。"""
    exe = DIST_DIR / "char_art_converter.exe"
    if not exe.is_file():
        pytest.fail(
            f"未找到构建产物 {exe}（需先执行 pyinstaller char_art_converter.spec）"
        )
    result = subprocess.run(
        [str(exe), "--version"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, f"--version 退出码 {result.returncode}: {result.stderr}"
    assert "1.0.0" in result.stdout


@pytest.mark.slow
def test_gui_build_artifacts(roots):
    """T-402：char_art_converter_gui.spec 存在；GUI exe 未构建时跳过并说明。"""
    spec = PROJECT_ROOT / "char_art_converter_gui.spec"
    assert spec.is_file(), "缺少 char_art_converter_gui.spec（打包配置）"
    gui_exe = DIST_DIR / "char_art_converter_gui.exe"
    if not gui_exe.is_file():
        pytest.skip(
            "char_art_converter_gui.exe 尚未构建（发版前执行 pyinstaller char_art_converter_gui.spec）"
        )
    result = subprocess.run(
        [str(gui_exe), "--version"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, f"GUI exe --version 退出码 {result.returncode}: {result.stderr}"
