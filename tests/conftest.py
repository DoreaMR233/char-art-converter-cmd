"""tests/conftest.py — 共享夹具（docs/gui_test_plan.md §3/§5.2）。

约定：
- CLI 用例通过 subprocess 真实调用仓库根 char_art_converter.py（§5.2）；
- 素材夹具优先复用 prepare 成员生成的 tests\\fixtures\\ 产物，缺失时按同规格现场生成；
- GUI 测试统一使用 QT_QPA_PLATFORM=offscreen（§6）。
"""
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

# GUI 测试必须在无显示环境下运行（文档 §6）；尊重用户已显式设置的值
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"

# 让测试进程能以与 CLI 相同的方式导入 src 包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ 路径与进程
@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录（仓库根，含 char_art_converter.py）。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def venv_python() -> str:
    """当前 .venv 的 python 解释器路径。"""
    return sys.executable


@pytest.fixture(scope="session")
def run_cli(venv_python, project_root):
    """CLI 运行器：subprocess.run([venv_python, char_art_converter.py, ...], cwd=项目根)（§5.2）。"""
    def _run(args, *, timeout=1800):
        return subprocess.run(
            [venv_python, str(project_root / "char_art_converter.py"), *[str(arg) for arg in args]],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    return _run


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    """FFmpeg 探测：与 src/utils/ffmpeg_utils.py 的 check_ffmpeg_available() 语义一致（跑 `ffmpeg -version`）。

    此处直接探测而不导入 src 包，避免测试进程为此支付 torch/PIL/cv2 的导入成本。
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ------------------------------------------------------------------ 素材夹具
def _materialize_fixture(tmp_path: Path, name: str) -> Path:
    """复制 tests\\fixtures\\<name> 到 tmp_path；缺失时按同规格现场生成。"""
    source = FIXTURE_DIR / name
    target = tmp_path / name
    if source.is_file():
        shutil.copyfile(source, target)
        return target
    _generate_fallback(name, target)
    return target


@pytest.fixture
def sample_image(tmp_path):
    """64×48 RGB 渐变 PNG（tests\\fixtures\\test_input.png 副本）。"""
    return _materialize_fixture(tmp_path, "test_input.png")


@pytest.fixture
def sample_gif(tmp_path):
    """3 帧 64×48、duration=100ms、loop=0 的 GIF（tests\\fixtures\\test_input.gif 副本）。"""
    return _materialize_fixture(tmp_path, "test_input.gif")


@pytest.fixture
def sample_video(tmp_path):
    """320×240、30fps、约60帧的 AVI（tests\\fixtures\\test_input.avi 副本）。"""
    return _materialize_fixture(tmp_path, "test_input.avi")


@pytest.fixture
def unsupported_file(tmp_path):
    """纯文本文件（tests\\fixtures\\unsupported.txt 副本），用于不支持格式相关用例。"""
    return _materialize_fixture(tmp_path, "unsupported.txt")


def _gradient_frame(frame_index: int = 0):
    """64×48 RGB 渐变 + 随帧移动的白色方块（与 tests\\fixtures\\generate.py 的 make_frame 同思路）。"""
    from PIL import Image

    width, height = 64, 48
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 4) % 256, (x * 2 + y * 2) % 256, (255 - x * 4) % 256)
    block_x = (frame_index * 16) % width
    for y in range(8, 24):
        for x in range(block_x, min(block_x + 16, width)):
            pixels[x, y] = (255, 255, 255)
    return image


def _generate_fallback(name: str, target: Path) -> None:
    from PIL import Image

    if name == "test_input.png":
        _gradient_frame(0).save(target)
    elif name == "test_input.gif":
        frames = [_gradient_frame(i) for i in range(3)]
        frames[0].save(
            target, save_all=True, append_images=frames[1:], duration=100, loop=0
        )
    elif name == "test_input.avi":
        _write_test_avi(target)
    elif name == "unsupported.txt":
        target.write_text("这是一个用于“不支持格式”用例的测试文本文件。\n", encoding="utf-8")
    else:
        raise FileNotFoundError(f"未知夹具: {name}")


def _write_test_avi(target: Path) -> None:
    import cv2
    import numpy as np

    for fourcc_code in ("MJPG", "mp4v"):
        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*fourcc_code), 30, (320, 240)
        )
        if not writer.isOpened():
            continue
        for i in range(60):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 4) % 256
            frame[:, :, 1] = (i * 3) % 256
            frame[:, :, 2] = (255 - i * 4) % 256
            writer.write(frame)
        writer.release()
        if target.is_file() and target.stat().st_size > 0:
            return
    raise RuntimeError("无法生成测试视频（MJPG/mp4v 均不可用）")


# ------------------------------------------------------------------ GUI 相关
@pytest.fixture
def gui_namespace():
    """Namespace 工厂，按文档 §7.1 参数表装配（默认值与 create_parser 默认一致）。"""
    def _build(input_path=None, **overrides):
        values = dict(
            input=input_path,
            output=None,
            density="medium",
            color_mode="color",
            limit_size=None,
            with_text=False,
            with_image=False,
            no_multithread=False,
            enable_gpu=True,
            gpu_memory_limit=None,
            debug=False,
        )
        values.update(overrides)
        return Namespace(**values)
    return _build


# ------------------------------------------------------------------ GUI 运行环境（pytest-qt 不可用，直接驱动 PySide6）
@pytest.fixture(scope="session")
def qapp():
    """session 级 QApplication（offscreen，见文件顶部环境设置）。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["pytest"])
    yield app
    # 不显式退出：session 内复用，进程结束自然回收


@pytest.fixture
def window(qapp):
    """新建 MainWindow 实例；测试结束后取消残留 worker 并清理日志挂载。

    兼容假 worker（T-310 用带 cancel/isRunning 的桩，无 QThread.wait）。
    """
    from gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    yield win
    worker = getattr(win, "_worker", None)
    if worker is not None and getattr(worker, "isRunning", lambda: False)():
        if hasattr(worker, "cancel"):
            worker.cancel()
        wait = getattr(worker, "wait", None)
        if callable(wait):
            wait(10000)
    win.log_drawer.unmount()
    win.bridge.detach()
    win.close()


@pytest.fixture
def qt_wait():
    """轮询条件直到满足或超时：持续 processEvents 驱动 Qt 事件循环。"""
    from PySide6.QtWidgets import QApplication

    def _wait(predicate, *, timeout=10.0, interval=0.05):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if predicate():
                return True
            time.sleep(interval)
        QApplication.processEvents()
        return predicate()

    return _wait
