"""生成 tests/fixtures 测试夹具（幂等：重复运行直接覆盖，不报错）。

用法（项目根目录执行）:
    & ".venv\\Scripts\\python.exe" tests\\fixtures\\generate.py --verify
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FIXTURE_DIR = Path(__file__).resolve().parent

PNG_PATH = FIXTURE_DIR / "test_input.png"
GIF_PATH = FIXTURE_DIR / "test_input.gif"
AVI_PATH = FIXTURE_DIR / "test_input.avi"
TXT_PATH = FIXTURE_DIR / "unsupported.txt"

PNG_SIZE = (64, 48)
GIF_FRAME_COUNT = 3
GIF_DURATION_MS = 100
VIDEO_SIZE = (320, 240)
VIDEO_FPS = 30
VIDEO_FRAME_COUNT = 60
VIDEO_FOURCC_PRIORITY = ("MJPG", "mp4v")


def make_frame(width, height, block_x):
    """RGB 渐变底图 + 白色移动色块。"""
    xs = np.linspace(0, 255, width, dtype=np.uint16)
    ys = np.linspace(0, 255, height, dtype=np.uint16)
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[..., 0] = xs[None, :]
    arr[..., 1] = ys[:, None]
    arr[..., 2] = (xs[None, :] + 2 * ys[:, None]) // 3
    block_w, block_h = width // 8, height // 4
    x0 = int(block_x) % width
    y0 = height // 2 - block_h // 2
    arr[y0:y0 + block_h, x0:x0 + block_w] = (255, 255, 255)
    return Image.fromarray(arr, "RGB")


def generate_png():
    make_frame(*PNG_SIZE, PNG_SIZE[0] // 2).save(PNG_PATH)


def generate_gif():
    frames = [make_frame(*PNG_SIZE, i * 16) for i in range(GIF_FRAME_COUNT)]
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )


def generate_avi():
    width, height = VIDEO_SIZE
    writer = None
    fourcc_used = None
    for fourcc_name in VIDEO_FOURCC_PRIORITY:
        writer = cv2.VideoWriter(
            str(AVI_PATH),
            cv2.VideoWriter_fourcc(*fourcc_name),
            VIDEO_FPS,
            (width, height),
        )
        if writer.isOpened():
            fourcc_used = fourcc_name
            break
        writer.release()
    if fourcc_used is None:
        raise RuntimeError(f"无法创建 VideoWriter（依次尝试过 {VIDEO_FOURCC_PRIORITY}）")
    try:
        for i in range(VIDEO_FRAME_COUNT):
            frame = np.asarray(make_frame(width, height, i * 4 % (width - width // 8)))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return fourcc_used


def generate_txt():
    TXT_PATH.write_text("这是一个用于\"不支持格式\"用例的测试文本文件。\n", encoding="utf-8")


def generate_all():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    generate_png()
    generate_gif()
    fourcc_used = generate_avi()
    generate_txt()
    print(f"夹具已生成到 {FIXTURE_DIR}（AVI fourcc={fourcc_used}）")


def verify_all():
    rows = []
    ok_all = True

    def record(name, ok, detail):
        nonlocal ok_all
        ok_all = ok_all and ok
        rows.append((name, "PASS" if ok else "FAIL", detail))

    for path in (PNG_PATH, GIF_PATH, AVI_PATH, TXT_PATH):
        size = path.stat().st_size if path.is_file() else "-"
        record(f"存在 {path.name}", path.is_file(), f"size={size} B")

    with Image.open(PNG_PATH) as img:
        record("PNG 尺寸=64×48 且 RGB", img.size == PNG_SIZE and img.mode == "RGB",
               f"size={img.size} mode={img.mode}")

    with Image.open(GIF_PATH) as img:
        duration = img.info.get("duration")
        record("GIF 帧数=3 且 duration=100ms",
               img.n_frames == GIF_FRAME_COUNT and duration == GIF_DURATION_MS,
               f"frames={img.n_frames} duration={duration}ms")

    cap = cv2.VideoCapture(str(AVI_PATH))
    if not cap.isOpened():
        record("AVI 可被 VideoCapture 打开", False, "open failed")
    else:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        record("AVI 帧数≈60", 55 <= count <= 65,
               f"frames={count} size={w}×{h} fps={fps:.1f}")
        record("AVI 尺寸=320×240 且 fps=30", (w, h) == VIDEO_SIZE and abs(fps - VIDEO_FPS) < 1.0,
               f"size={w}×{h} fps={fps:.1f}")
    cap.release()

    for name, status, detail in rows:
        print(f"{status}  {name}  [{detail}]")
    return ok_all


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成测试夹具（幂等）")
    parser.add_argument("--verify", action="store_true", help="生成后逐项验证并输出结果")
    args = parser.parse_args(argv)
    generate_all()
    if args.verify:
        sys.exit(0 if verify_all() else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
