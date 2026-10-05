"""T-201~T-206：图片类 CLI 集成测试（docs/gui_test_plan.md §5）。

所有用例通过 subprocess 真实调用仓库根的 char_art_converter.py（§5.2）。
退出码约定：0=成功，1=参数/校验错误，2=输入文件不存在，130=用户中断。
"""
import pytest
from PIL import Image

from src.configs.common_config import CHAR_DENSITY_CONFIG


def _detail(res):
    return f"stdout:\n{res.stdout[-1500:]}\nstderr:\n{res.stderr[-1500:]}"


def test_help_and_version(run_cli):
    """T-201：--help 展示全部参数；--version 含 1.0.0；退出码均为 0。"""
    help_res = run_cli(["--help"])
    assert help_res.returncode == 0, _detail(help_res)
    for flag in ("-o", "--output", "-d", "--density", "-c", "--color-mode",
                 "-l", "--limit-size", "-t", "--with-text", "-i", "--with-image",
                 "--no-multithread", "--disable-gpu", "--debug", "--gpu-memory-limit"):
        assert flag in help_res.stdout, f"--help 输出缺少参数 {flag}"

    version_res = run_cli(["--version"])
    assert version_res.returncode == 0, _detail(version_res)
    assert "1.0.0" in version_res.stdout


def test_static_image_default(run_cli, sample_image):
    """T-202：静图默认参数 + -t -i：rc0，输出目录出现 .txt 与图片产物，stdout 含进度特征。"""
    res = run_cli([str(sample_image), "-t", "-i"])
    assert res.returncode == 0, _detail(res)

    out_dir = sample_image.parent / "test_input_color_char_art_image"
    assert out_dir.is_dir(), "未生成默认输出目录"
    assert (out_dir / "test_input_color_char_art.txt").is_file(), "缺少文本产物"
    assert (out_dir / "test_input_color_char_art.png").is_file(), "缺少图片产物"

    assert "%" in res.stdout, "stdout 缺少 tqdm 进度特征（%）"


_DENSITY_COLOR_COMBOS = [
    ("low", "grayscale"),
    ("low", "colorBackground"),
    ("medium", "color"),
    ("high", "grayscale"),
    ("high", "colorBackground"),
]


@pytest.mark.parametrize("density,color_mode", _DENSITY_COLOR_COMBOS)
def test_density_color_smoke(run_cli, sample_image, density, color_mode):
    """T-203：-d × -c 组合冒烟：全部 rc0，txt 非空且所有字符属于所选档位字符集。

    文档用例表未写 -t，但预期检查“txt 产物”，因此命令补上 -t（与 §5 预期一致）。
    """
    res = run_cli([str(sample_image), "-d", density, "-c", color_mode, "-t"])
    assert res.returncode == 0, _detail(res)

    out_dir = sample_image.parent / f"test_input_{color_mode}_char_art_image"
    txt = out_dir / f"test_input_{color_mode}_char_art.txt"
    assert txt.is_file(), f"缺少文本产物 {txt}"
    content = txt.read_text(encoding="utf-8").replace("\r", "").replace("\n", "")
    assert content, "文本产物为空"

    allowed = set(CHAR_DENSITY_CONFIG[density])
    stray = sorted({ch for ch in content if ch not in allowed})
    assert not stray, f"出现档位外字符: {''.join(stray[:20])}"


def test_no_multithread_identical_output(run_cli, sample_image, tmp_path):
    """T-204：--no-multithread 与默认多线程输出的 txt 字节级一致。"""
    out_default = tmp_path / "out_default"
    out_single = tmp_path / "out_single"

    res_default = run_cli([str(sample_image), "-t", "-o", str(out_default)])
    res_single = run_cli(
        [str(sample_image), "-t", "-o", str(out_single), "--no-multithread"]
    )
    assert res_default.returncode == 0, _detail(res_default)
    assert res_single.returncode == 0, _detail(res_single)

    txt_default = out_default / "test_input_color_char_art.txt"
    txt_single = out_single / "test_input_color_char_art.txt"
    assert txt_default.is_file() and txt_single.is_file()
    assert txt_default.read_bytes() == txt_single.read_bytes(), "多线程与单线程文本产物不一致"


def test_limit_size_32x24(run_cli, sample_image, tmp_path):
    """T-205：-l 32 24 后字符网格恰为 32×24，图片像素 ≈ 32*cw × 24*ch（cw/ch 为产品同款字符尺寸）。"""
    from src.utils.font_utils import calculate_char_size, load_font

    out = tmp_path / "out"
    res = run_cli([str(sample_image), "-t", "-l", "32", "24", "-o", str(out)])
    assert res.returncode == 0, _detail(res)

    txt = out / "test_input_color_char_art.txt"
    lines = txt.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 24, f"期望 24 行，实际 {len(lines)}"
    assert all(len(line) == 32 for line in lines), "存在宽度不为 32 的字符行"

    char_width, char_height = calculate_char_size(load_font(12))
    png = out / "test_input_color_char_art.png"
    with Image.open(png) as image:
        assert abs(image.width - 32 * char_width) <= char_width, (
            f"输出宽度 {image.width} 偏离 32*{char_width}={32 * char_width}"
        )
        assert abs(image.height - 24 * char_height) <= char_height, (
            f"输出高度 {image.height} 偏离 24*{char_height}={24 * char_height}"
        )


def test_animated_gif(run_cli, sample_gif):
    """T-206：动图 GIF + -t -i：rc0，生成动图产物与逐帧 txt。"""
    res = run_cli([str(sample_gif), "-t", "-i"])
    assert res.returncode == 0, _detail(res)

    out_dir = sample_gif.parent / "test_input_color_char_art_image"
    gif_out = out_dir / "test_input_color_char_art.gif"
    assert gif_out.is_file(), "缺少动图产物"

    text_dir = out_dir / "test_input_color_char_art_text"
    assert text_dir.is_dir(), "缺少逐帧文本目录"
    frame_txts = sorted(text_dir.glob("test_input_color_char_art_frame*.txt"))
    assert len(frame_txts) == 3, f"期望 3 个帧文本，实际 {len(frame_txts)}"
    for frame_txt in frame_txts:
        assert frame_txt.read_text(encoding="utf-8").strip(), f"{frame_txt.name} 为空"

    with Image.open(gif_out) as image:
        assert getattr(image, "n_frames", 1) == 3, "产物帧数不为 3"
