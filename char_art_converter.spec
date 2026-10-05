# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['char_art_converter.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Core dependencies
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageFont',
        'PIL.ImageDraw',
        'PIL.ImageFilter',
        # Font tools
        'fontTools',
        'fontTools.ttLib',
        # File detection
        'puremagic',
        # JSON
        'jsonpath',
        'jsonpath_ng',
        # FFmpeg
        'ffmpeg_python',
        'python_ffmpeg',
        # Progress/UI
        'tqdm',
        'colorama',
        # System
        'psutil',
        # Torch related
        'scipy',
        'sympy',
        'networkx',
        'jinja2',
        'markupsafe',
        # Subprocess related
        'multiprocessing',
        'concurrent.futures',
        'threading',
        'subprocess',
        # Other
        'better_ffmpeg_progress',
        'pillow_heif',
        'ply',
        'pyee',
        'librt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='char_art_converter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
