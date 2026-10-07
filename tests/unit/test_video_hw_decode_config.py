"""硬件解码友好编码：编码器/像素格式/编码档次映射的单元测试（硬解改造回归）。

覆盖：
- 所有受支持的视频容器都有编码器与编码档次映射，且映射到的都是硬件解码器可解码的编码格式；
- 自动生成的编码参数固定包含 8bit 4:2:0 像素格式与宽高取偶滤镜；
- mp4/mov 追加 faststart；WebM 追加 VP9 速度参数；
- 显式指定编码器时的兼容行为（档次仅在与容器默认编码器一致时才套用）。
"""
from pathlib import Path

import pytest

from src.configs import (
    EXTENSION_TO_CODEC,
    HW_DECODE_FASTSTART_EXTENSIONS,
    HW_DECODE_PIX_FMT,
    HW_DECODE_UNSUPPORTED_EXTENSIONS,
    SUPPORTED_VIDEO_FORMATS,
    WARNING_MESSAGES,
)
from src.utils import get_hw_decode_encode_config, get_hw_decode_video_args

# 主流硬件解码器（NVDEC/DXVA2/D3D11VA/QuickSync/VAAPI）可解码的编码格式对应的 ffmpeg 编码器
HW_DECODABLE_ENCODERS = {'libx264', 'libx265', 'libvpx-vp9', 'mpeg2video'}


def _args_for(ext: str, encoder=None) -> list[str]:
    return get_hw_decode_video_args(Path(f'output{ext}'), encoder=encoder)


@pytest.mark.unit
def test_supported_video_formats_have_hw_decode_mapping():
    """所有受支持的视频容器都应映射到硬件解码器可解码的编码格式。"""
    assert SUPPORTED_VIDEO_FORMATS, "受支持的视频格式列表不应为空"
    for ext in SUPPORTED_VIDEO_FORMATS:
        encoder, _profile = get_hw_decode_encode_config(Path(f'output{ext}'))
        assert encoder == EXTENSION_TO_CODEC[ext]
        if ext in HW_DECODE_UNSUPPORTED_EXTENSIONS:
            continue
        assert encoder in HW_DECODABLE_ENCODERS, f"{ext} 映射到不可硬解的编码器 {encoder}"


@pytest.mark.unit
@pytest.mark.parametrize('ext', SUPPORTED_VIDEO_FORMATS)
def test_args_always_enforce_hw_decode_constraints(ext):
    """每个容器的编码参数都必须包含 8bit 4:2:0 像素格式与宽高取偶滤镜。"""
    args = _args_for(ext)
    assert args[args.index('-c:v') + 1] == EXTENSION_TO_CODEC[ext]
    assert args[args.index('-pix_fmt') + 1] == HW_DECODE_PIX_FMT == 'yuv420p'
    assert 'scale=trunc(iw/2)*2:trunc(ih/2)*2' == args[args.index('-vf') + 1]
    has_faststart = '-movflags' in args
    assert has_faststart is (ext in HW_DECODE_FASTSTART_EXTENSIONS)


@pytest.mark.unit
def test_h264_container_args_use_high_profile():
    """MP4/AVI/MKV/MOV/FLV 使用 H.264 high 档次（硬件解码器普遍支持的档次）。"""
    for ext in ('.mp4', '.avi', '.mkv', '.mov', '.flv'):
        args = _args_for(ext)
        assert args[args.index('-c:v') + 1] == 'libx264'
        assert args[args.index('-profile:v') + 1] == 'high'


@pytest.mark.unit
def test_webm_args_use_vp9_speed_options():
    """WebM 使用 VP9 并附带速度参数，避免默认编码速度过慢。"""
    args = _args_for('.webm')
    assert args[args.index('-c:v') + 1] == 'libvpx-vp9'
    assert args[args.index('-profile:v') + 1] == '0'
    for option in ('-deadline', '-cpu-used', '-row-mt'):
        assert option in args


@pytest.mark.unit
def test_mpeg_container_args_are_valid_for_mpeg2():
    """MPG/MPEG 使用 MPEG-2 的 main 档次（high 档次对 mpeg2video 非法）。"""
    for ext in ('.mpg', '.mpeg'):
        args = _args_for(ext)
        assert args[args.index('-c:v') + 1] == 'mpeg2video'
        assert args[args.index('-profile:v') + 1] == 'main'


@pytest.mark.unit
def test_wmv_is_declared_unsupported_with_warning_message():
    """WMV 没有可用的硬解编码格式，必须被声明为不支持并配有用户可见的警告模板。"""
    assert '.wmv' in HW_DECODE_UNSUPPORTED_EXTENSIONS
    assert 'hw_decode_unsupported_container' in WARNING_MESSAGES
    text = WARNING_MESSAGES['hw_decode_unsupported_container'].format('.wmv', 'wmv2')
    assert '.wmv' in text and 'wmv2' in text


@pytest.mark.unit
def test_explicit_encoder_override_skips_unsupported_profile():
    """显式指定与容器默认不同的编码器时不得套用容器档次（如 mpeg4 不支持 high）。"""
    args = _args_for('.mp4', encoder='mpeg4')
    assert args[args.index('-c:v') + 1] == 'mpeg4'
    assert '-profile:v' not in args
    # 像素格式与宽高取偶约束仍然保留
    assert args[args.index('-pix_fmt') + 1] == 'yuv420p'
    assert '-vf' in args


@pytest.mark.unit
def test_explicit_encoder_override_keeps_container_profile_when_matching():
    """显式指定的编码器与容器默认编码器一致时，仍套用该容器的编码档次。"""
    args = _args_for('.mp4', encoder='libx264')
    assert args[args.index('-c:v') + 1] == 'libx264'
    assert args[args.index('-profile:v') + 1] == 'high'


@pytest.mark.unit
def test_unknown_extension_falls_back_to_default_encoder():
    """未知扩展名回退到默认编码器，并仍受硬解约束（不抛异常）。"""
    args = _args_for('.unknown')
    assert args[args.index('-c:v') + 1]
    assert args[args.index('-pix_fmt') + 1] == 'yuv420p'
