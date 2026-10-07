# -*- mode: python ; coding: utf-8 -*-
"""字符画转换器 GUI 打包配置（照抄 CLI spec 的 hiddenimports/excludes）。"""

a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=[],
    datas=[('gui/resources', 'gui/resources')],
    hiddenimports=[
        'torch', 'torchvision', 'cv2', 'numpy', 'PIL', 'PIL.Image',
        'PIL.ImageFont', 'PIL.ImageDraw', 'PIL.ImageFilter',
        'fontTools', 'fontTools.ttLib', 'puremagic', 'jsonpath', 'jsonpath_ng',
        'ffmpeg_python', 'python_ffmpeg', 'tqdm', 'colorama', 'psutil',
        'scipy', 'sympy', 'networkx', 'jinja2', 'markupsafe',
        'multiprocessing', 'concurrent.futures', 'threading', 'subprocess', 'ctypes',
        'better_ffmpeg_progress', 'pillow_heif', 'ply', 'pyee', 'librt',
        'PySide6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pandas', 'IPython', 'jupyter'],
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
    name='char_art_converter_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
