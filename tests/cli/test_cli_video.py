"""T-207 / T-213：视频类 CLI 用例（docs/gui_test_plan.md §5）。

T-207 无 FFmpeg 时允许 skip；T-213 仅 Windows 且能投递 CTRL_BREAK_EVENT 时执行（slow）。
"""
import os
import queue
import signal
import subprocess
import threading
import time

import pytest


def _detail(res):
    return f"stdout:\n{res.stdout[-1500:]}\nstderr:\n{res.stderr[-1500:]}"


def test_video_convert(run_cli, sample_video, ffmpeg_available):
    """T-207：视频 + -t：rc0，生成文本目录与视频产物，stdout 含“统计视频帧数”。"""
    if not ffmpeg_available:
        pytest.skip("FFmpeg 不可用（文档允许视频用例跳过）")

    res = run_cli([str(sample_video), "-t"])
    assert res.returncode == 0, _detail(res)

    out_dir = sample_video.parent / "test_input_color_char_art_video"
    assert (out_dir / "test_input_color_char_art_text").is_dir(), "缺少逐帧文本目录"
    assert (out_dir / "test_input_color_char_art.avi").is_file(), "缺少视频产物"
    assert "统计视频帧数" in res.stdout


@pytest.mark.slow
def test_interrupt_video_returns_130(project_root, venv_python, sample_video, ffmpeg_available):
    """T-213：视频处理中发送 CTRL_BREAK_EVENT → rc130，且输出目录被清理。

    仅在 Windows 且能投递 CTRL_BREAK_EVENT 时执行（文档允许不支持信号的环境 skip）。
    """
    if not ffmpeg_available:
        pytest.skip("FFmpeg 不可用（文档允许视频用例跳过）")
    if os.name != "nt" or not hasattr(signal, "CTRL_BREAK_EVENT"):
        pytest.skip("CTRL_BREAK_EVENT 仅 Windows 支持（文档允许不支持信号的环境 skip）")

    proc = subprocess.Popen(
        [venv_python, str(project_root / "char_art_converter.py"), str(sample_video), "-t"],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    output_chunks = queue.Queue()

    def _reader():
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            output_chunks.put(chunk)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    output = b""
    marker = "统计视频帧数".encode("utf-8")
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline and marker not in output:
            if proc.poll() is not None:
                break
            try:
                output += output_chunks.get(timeout=0.2)
            except queue.Empty:
                pass

        if marker not in output:
            proc.kill()
            pytest.fail(
                f"300s 内未观察到视频处理进度。输出片段:\n"
                f"{output.decode('utf-8', 'replace')[-2000:]}"
            )

        time.sleep(3)  # 确保进入帧处理循环

        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        except OSError as exc:
            proc.kill()
            pytest.skip(f"当前会话无法投递 CTRL_BREAK_EVENT: {exc}")

        try:
            returncode = proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(
                f"发送中断后进程未在 240s 内退出。输出片段:\n"
                f"{output.decode('utf-8', 'replace')[-2000:]}"
            )

        assert returncode == 130, (
            f"期望退出码 130，实际 {returncode}。输出片段:\n"
            f"{output.decode('utf-8', 'replace')[-2000:]}"
        )

        out_dir = sample_video.parent / "test_input_color_char_art_video"
        assert not out_dir.exists(), "中断后默认输出目录未清理"
    finally:
        if proc.poll() is None:
            proc.kill()
