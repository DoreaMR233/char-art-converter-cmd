"""生成 tests\\baseline\\cli_baseline.txt（P0 基线，T-214 用）。

对每个 CLI 场景记录：命令、退出码、确定性关键输出标记（进度存在性、100% 完成、
DEBUG 日志、错误文案）。只记布尔标记，不含耗时/速度等易变内容，保证两次运行的
diff 稳定。

用法：.venv\\Scripts\\python tests\\baseline\\generate_cli_baseline.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "char_art_converter.py"
PYTHON = sys.executable
BASELINE_DIR = PROJECT_ROOT / "tests" / "baseline"
WORK_DIR = PROJECT_ROOT / "build" / "baseline_work"
INPUT_IMAGE = WORK_DIR / "test_input.png"
INPUT_GIF = WORK_DIR / "test_input.gif"
INPUT_VIDEO = WORK_DIR / "test_input.avi"
INPUT_PDF = WORK_DIR / "unsupported.pdf"
BLOCKER_FILE = WORK_DIR / "blocker.txt"


def run(args, timeout=600):
    result = subprocess.run(
        [PYTHON, str(CONVERTER), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def make_inputs():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    if not INPUT_IMAGE.is_file():
        w, h = 64, 48
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(h):
            for x in range(w):
                px[x, y] = (
                    int(255 * x / (w - 1)),
                    int(255 * y / (h - 1)),
                    128,
                )
        img.save(INPUT_IMAGE)

    if not INPUT_GIF.is_file():
        frames = []
        for i in range(3):
            f = Image.new("RGB", (64, 48), (30 * i, 60, 90))
            frames.append(f)
        frames[0].save(
            INPUT_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

    if not INPUT_VIDEO.is_file():
        import cv2

        writer = cv2.VideoWriter(
            str(INPUT_VIDEO), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        for i in range(30):
            import numpy as np

            writer.write(np.full((48, 64, 3), 255 * (i % 2) * 0.3, dtype=np.uint8))
        writer.release()

    if not INPUT_PDF.is_file():
        INPUT_PDF.write_bytes(b"%PDF-1.4\n% content\n%%EOF\n")
    if not BLOCKER_FILE.is_file():
        BLOCKER_FILE.write_text("blocker", encoding="utf-8")


def sections():
    """返回 (标题, 命令) 列表——与 tests\\cli\\* 的断言场景一一对应。

    density×color 组合与 tests\\cli\\test_cli_image.py:43-49 一致
    （合法色值：grayscale / color / colorBackground）。
    """
    img = str(INPUT_IMAGE)
    gif = str(INPUT_GIF)
    video = str(INPUT_VIDEO)
    pdf = str(INPUT_PDF)
    blocker = str(BLOCKER_FILE)
    return [
        ("help", ["--help"]),
        ("version", ["--version"]),
        ("image_default", [img]),
        ("image_density_low_grayscale", [img, "-d", "low", "-c", "grayscale", "-t"]),
        ("image_density_low_color_background", [img, "-d", "low", "-c", "colorBackground", "-t"]),
        ("image_density_medium_color", [img, "-d", "medium", "-c", "color", "-t"]),
        ("image_density_high_grayscale", [img, "-d", "high", "-c", "grayscale", "-t"]),
        ("image_density_high_color_background", [img, "-d", "high", "-c", "colorBackground", "-t"]),
        ("image_no_multithread", [img, "--no-multithread"]),
        ("image_limit_size_32x24", [img, "-l", "32", "24"]),
        ("gif_animated", [gif]),
        ("video_convert", [video]),
        ("video_with_text", [video, "-t"]),
        ("error_nonexistent", [str(WORK_DIR / "missing.png")]),
        ("error_unsupported_pdf", [pdf]),
        ("error_limit_size_single", [img, "-l", "32"]),
        ("error_limit_size_zero", [img, "-l", "0", "10"]),
        ("error_output_is_file", [img, "-o", blocker]),
        ("image_debug", [img, "--debug"]),
    ]


def key_markers(rc, stdout, stderr, title):
    text = stdout + "\n" + stderr
    markers = [f"rc={rc}"]
    if title in ("help",):
        markers.append("usage_present=" + str("usage:" in text.lower()))
    elif title == "version":
        markers.append("version_1.0.0_present=" + str("1.0.0" in text))
    elif title.startswith("image_") or title.startswith("gif") or title.startswith("video"):
        has_progress = any("%" in line for line in text.splitlines())
        has_100 = any("100%" in line for line in text.splitlines())
        markers.append(f"progress_present={has_progress}")
        markers.append(f"progress_100_present={has_100}")
        if title.startswith("video"):
            markers.append("frame_count_msg_present=" + str("统计视频帧数" in text))
        if "debug" in title:
            markers.append("DEBUG_present=" + str("DEBUG" in text))
            markers.append("INFO_present=" + str("INFO" in text))
        else:
            markers.append("DEBUG_absent=" + str("DEBUG" not in text))
    elif title.startswith("error_"):
        probes = {
            "文件": "文件",
            "不支持": "不支持",
            ".pdf": ".pdf",
            "--limit-size": "--limit-size",
            "宽度和高度必须大于0": "宽度和高度必须大于0",
            "目录": "目录",
        }
        for label, probe in probes.items():
            if probe in text:
                markers.append(f"error_text_present={label}")
    return markers


def main():
    make_inputs()
    lines = ["char-art-converter CLI baseline (P0)", f"python={PYTHON}", ""]
    for title, args in sections():
        rc, stdout, stderr = run(args)
        lines.append(f"[{title}]")
        lines.append("cmd=" + " ".join(args))
        lines.extend(key_markers(rc, stdout, stderr, title))
        lines.append("")
    baseline = BASELINE_DIR / "cli_baseline.txt"
    baseline.write_text("\n".join(lines), encoding="utf-8")
    print(f"baseline written: {baseline}")


if __name__ == "__main__":
    main()
