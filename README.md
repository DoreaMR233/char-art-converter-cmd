# 字符画转换器 (Char Art Converter)

## 项目说明

字符画转换器用于将图像或视频转换为精美的字符画，提供命令行（CLI）与图形界面（GUI）两种使用方式。支持多种配置选项，包括字符密度、颜色模式、输出大小限制等，并提供GPU加速功能以提高处理效率。

主要功能：

- 支持多种图像和视频格式输入
- 三种字符密度级别可选（低/中/高）
- 支持灰度/彩色/彩色背景字符画输出
- 尺寸限制三种模式：原图尺寸（不限制）/ 默认大小 / 自定义字符网格宽高
- 支持GPU加速处理（默认启用，GPU不可用时自动回退CPU）
- 支持多线程处理动图帧/视频帧
- 可同时输出图像/文本格式的字符画
- 输出的字符画视频采用硬件解码友好编码（H.264/H.265/VP9/MPEG-2 等主流编码 + `yuv420p` 4:2:0 8bit 像素格式 + 主流编码档次 + 偶数宽高），可被 NVDEC、DXVA2、D3D11VA、QuickSync、VAAPI 等硬件解码器直接解码
- 图形界面：文件选择、参数面板（滚动布局）、进度显示、取消转换、日志抽屉

## 文件目录

```
char-art-converter-cmd/
├── char_art_converter.py          # 命令行入口
├── gui_app.py                     # 图形界面入口
├── char_art_converter.spec        # CLI 打包配置（PyInstaller）
├── char_art_converter_gui.spec    # GUI 打包配置（PyInstaller，窗口程序）
├── requirements.txt               # 依赖清单（运行/GUI/开发合并为一个文件）
├── pytest.ini                     # pytest 配置（unit/cli/gui/slow 标记）
├── LICENSE                        # 许可证文件
├── README.md                      # 项目说明文档
├── src/                           # 内核模块（CLI 与 GUI 共用）
│   ├── configs/                   # 配置模块
│   │   ├── audio_config.py        # 音频格式配置
│   │   ├── common_config.py       # 密度/颜色模式/默认参数/GPU 配置
│   │   ├── image_config.py        # 图像格式配置
│   │   ├── message_config.py      # 提示消息配置
│   │   └── video_config.py        # 视频格式/编解码器配置
│   ├── enums/                     # 枚举类型模块
│   │   ├── color_modes.py         # 颜色模式枚举
│   │   ├── file_type.py           # 文件类型枚举
│   │   └── save_modes.py          # 保存模式枚举
│   ├── processors/                # 处理器模块
│   │   ├── based_processor.py     # 处理器基类（参数初始化/校验/多线程/进度/停止检查）
│   │   ├── image_processor.py     # 图像处理
│   │   └── video_processor.py     # 视频处理
│   └── utils/                     # 工具函数模块
│       ├── audio_utils.py         # 音频工具
│       ├── char_art_utils.py      # 字符画生成与尺寸计算
│       ├── color_utils.py         # 颜色处理工具
│       ├── ffmpeg_utils.py        # FFmpeg 可用性检查
│       ├── file_utils.py          # 输出路径/文件操作工具
│       ├── font_utils.py          # 字体处理工具
│       ├── format_utils.py        # 格式化工具
│       ├── gpu_utils.py           # GPU 加速工具
│       ├── image_utils.py         # 图像处理工具
│       ├── logging_utils.py       # 日志工具
│       ├── multi_processing_utils.py # 多进程工具
│       ├── progress_bar_utils.py  # 进度条工具
│       ├── save_uitls.py          # 保存工具
│       ├── validate_utils.py      # 参数校验工具
│       └── video_utils.py         # 视频处理工具
├── gui/                           # 图形界面（PySide6）
│   ├── app.py                     # 应用装配与主题
│   ├── main_window.py             # 主窗口（状态机/表单校验/任务控制）
│   ├── worker.py                  # 转换线程（与 CLI 相同退出码语义）
│   ├── progress_bridge.py         # 处理器→GUI 进度桥
│   ├── theme.py                   # 主题令牌与 QSS 构建（含对比度自检）
│   ├── resources/                 # 资源文件
│   │   ├── icons/                 # SVG 图标
│   │   ├── theme_light.qss        # 浅色主题样式
│   │   └── theme_dark.qss         # 深色主题样式
│   └── widgets/                   # 界面组件
│       ├── file_picker.py         # 文件选择
│       ├── log_drawer.py          # 日志抽屉
│       └── param_panel.py         # 参数面板
└── tests/                         # 测试目录
    ├── baseline/                  # CLI 回归基线
    ├── cli/                       # CLI 回归测试
    ├── fixtures/                  # 测试素材
    ├── gui/                       # GUI 测试
    ├── smoke/                     # 冒烟测试
    ├── unit/                      # 单元测试
    └── conftest.py                # 共享 fixture
```

## 项目结构

项目采用模块化设计，主要分为以下几个核心模块：

1. **配置模块（src/configs）**：管理应用程序的各种配置参数，包括字符密度、颜色模式、默认设置、GPU 配置等。
2. **枚举模块（src/enums）**：定义颜色模式、文件类型、保存模式等枚举类型。
3. **处理器模块（src/processors）**：实现核心的图像处理和视频处理逻辑，CLI 与 GUI 共用。
4. **工具函数模块（src/utils）**：提供字符画生成、颜色处理、文件操作、GPU 加速等辅助功能。
5. **图形界面模块（gui）**：基于 PySide6 的桌面界面，通过 `ConversionWorker` 线程调用与 CLI 相同的处理内核。

## 环境要求

- Python 3.10+
- Windows 10/11
- FFmpeg（必需，`ffmpeg` 命令需加入系统 PATH，视频转换依赖）
- 支持CUDA的NVIDIA GPU（可选，用于GPU加速）

## 安装依赖

### 1. 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 激活虚拟环境（Linux/macOS）
source .venv/bin/activate
```

### 2. 安装依赖

`requirements.txt` 已合并运行依赖、GUI 依赖（PySide6）与开发/打包依赖（pytest、pyinstaller）：

```bash
pip install -r requirements.txt
```

### 3. 安装PyTorch（GPU版本，可选）

`requirements.txt` 中的 `torch` 为 PyPI 默认版本。如需 GPU 加速，请根据您的系统和 CUDA 版本安装对应的 PyTorch 构建（覆盖默认安装）。请访问[PyTorch官方网站](https://pytorch.org/get-started/locally/)获取最新的安装命令。

示例（Windows + CUDA 11.8）：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 图形界面（GUI）

```bash
python gui_app.py
```

主窗口为上下布局（960×640，最小 800×560）：上方是可滚动的配置区，下方是进度与操作区（窗口变矮或日志抽屉展开时，配置区改为滚动，不会被压缩或遮挡）。

- **上方配置区**：输入文件选择（图片或视频）、参数面板
  - 字符密度：低（10 级）/ 中（16 级）/ 高（70 级）
  - 颜色模式：灰度 / 彩色 / 彩色（深底）
  - 尺寸限制（单选按钮组）：
    - **原图尺寸**：不限制字符网格，按原图尺寸处理（等价于不传 `-l`）
    - **默认大小**：网格列数 = 原图宽 ÷ 6、行数 = 原图高 ÷ 6（内置字体大小 12）
    - **自定义尺寸**：宽/高输入框启用，填写字符网格的列数与行数（选中该模式时会自动填入原图尺寸，即不缩放）；都留空时按默认大小处理，只填一维时启动会提示错误
  - GPU 显存：默认勾选"使用默认值（配置文件 80%）"，滑条灰显不可拖动；取消勾选后用滑条选择 5%–95%
  - 输出选项：同时保存文本 (.txt)、同时保存字符画图像、启用 GPU 加速、禁用多线程处理、调试日志（自动展开日志抽屉）
  - 输出目录：留空则在输入文件旁自动生成目录
- **下方操作区**：阶段标签 + 进度条（GIF/视频转换时额外显示与线程数一致的每线程进度条，和命令行模式一致）、状态信息、开始转换/取消转换/打开输出文件夹按钮、日志抽屉

转换期间参数面板与文件选择会被锁定，可点击"取消转换"中止；转换成功后可直接打开输出目录。

GUI 参数与 CLI 参数对应关系：

| GUI 控件 | CLI 参数 | 说明 |
|---------|---------|------|
| 字符密度 | `-d, --density` | low / medium / high，默认 medium |
| 颜色模式 | `-c, --color-mode` | 灰度 / 彩色 / 彩色（深底），默认彩色 |
| 尺寸限制·原图尺寸 | （缺省） | 不传 `-l`，不限制字符网格，按原图尺寸处理 |
| 尺寸限制·默认大小 | `-l`（不带参数） | 字符网格 = 原图宽 ÷ 6 × 原图高 ÷ 6 |
| 尺寸限制·自定义尺寸 | `-l 宽 高` | 自定义字符网格列数 × 行数（上限为原图尺寸，超出则夹取） |
| GPU 显存·使用默认值 | （缺省） | 使用配置默认 80% |
| GPU 显存·自定义 | `--gpu-memory-limit` | GUI 用滑条选择 5%–95%，内核按占比计算 |
| 启用 GPU 加速 | 默认启用（`--disable-gpu` 关闭） | GPU 不可用自动回退 CPU |
| 同时保存文本 | `-t, --with-text` | 输出 .txt 字符画文本 |
| 同时保存字符画图像 | `-i, --with-image` | 仅视频/动图有效 |
| 禁用多线程处理 | `--no-multithread` | 动画帧/视频帧处理 |
| 调试日志 | `--debug` | GUI 中自动展开日志抽屉 |
| 输出目录 | `-o, --output` | 留空 = 输入文件旁自动生成 |

### 命令行（CLI）

#### 基本用法

```bash
python char_art_converter.py input.jpg
```

#### 指定输出目录

```bash
python char_art_converter.py input.gif -o output_dir
```

#### 转换为灰度字符画并指定尺寸

```bash
python char_art_converter.py input.png --color-mode grayscale --limit-size 120 60
```

#### 同时输出图像和文本文件

```bash
python char_art_converter.py input.png --with-text
```

#### 视频转换（GPU 默认启用）

```bash
python char_art_converter.py input.mp4
```

如需关闭 GPU 加速：

```bash
python char_art_converter.py input.mp4 --disable-gpu
```

## 命令行参数

### 必需参数

| 参数    | 描述       |
|-------|----------|
| input | 输入图像或视频文件路径 |

### 可选参数

| 参数                      | 描述                                                                                                       |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `-o, --output`          | 设置输出目录。如果不指定，将在输入文件同目录创建 `{文件名}_{颜色模式}_char_art_{image\|video}` 目录                                    |
| `-d, --density`         | 字符密度级别 (默认: medium)<br>选项: low, medium, high                                                             |
| `-c, --color-mode`      | 颜色模式 (默认: color)<br>选项: grayscale, color, colorBackground                                                |
| `-l, --limit-size`      | 调整输入图片尺寸，限制的是**字符网格的列数 × 行数**（最终图像像素 = 网格 × 单字符块大小，内置字体 12 时为 7×8）<br>不指定该参数：不限制，按原图尺寸处理<br>不带参数：使用默认大小（宽与高均为原图对应尺寸 ÷ 6，内置字体大小 12）<br>带两个参数：指定字符网格的宽度和高度（正整数，超过原图尺寸会被夹到原图尺寸并给出警告）<br>格式: `[LIMIT_WIDTH, LIMIT_HEIGHT]` |
| `-t, --with-text`       | 同时输出字符画图像和文本文件 (.txt)                                                                                     |
| `-i, --with-image`      | 同时输出字符画视频/动图和字符画图像（仅当输入文件为视频或动图时有效）                                                                      |
| `--no-multithread`      | 禁用多线程处理动画帧/视频帧（默认启用多线程）                                                                                  |
| `--disable-gpu`         | 禁用GPU并行计算加速，使用CPU处理（默认启用GPU，GPU不可用时自动回退CPU）                                                              |
| `--debug`               | 启用DEBUG级别日志输出                                                                                            |
| `--gpu-memory-limit MB` | 设置GPU内存限制（MB，正整数），默认使用配置文件中的值（80%）                                                                       |
| `--version`             | 显示版本信息（当前 1.0.0）                                                                                        |

## 退出码

CLI 与 GUI 采用相同的退出码语义：

| 退出码 | 含义 |
|-------|------|
| 0 | 成功 |
| 1 | 一般错误（参数无效、处理异常等） |
| 2 | 文件未找到 |
| 130 | 用户中断（CLI 按 Ctrl+C / GUI 点击取消） |

## 开发与测试

### 运行测试

```powershell
# GUI 测试需要在 offscreen 平台运行
$env:QT_QPA_PLATFORM = 'offscreen'

# 全量快速回归（跳过慢速端到端测试）
pytest -m "not slow"

# 按分类运行
pytest tests\unit
pytest tests\cli
pytest tests\gui
```

pytest 标记（见 pytest.ini）：`unit`（单元测试）、`cli`（CLI 回归测试）、`gui`（GUI 测试，需 `QT_QPA_PLATFORM=offscreen`）、`slow`（慢速端到端测试，默认跳过）。

### 打包

```bash
# CLI 可执行文件（控制台程序）
pyinstaller char_art_converter.spec
# 生成 dist\char_art_converter\char_art_converter.exe

# GUI 可执行文件（窗口程序）
pyinstaller char_art_converter_gui.spec
# 生成 dist\char_art_converter_gui\char_art_converter_gui.exe
```

## 常见问题

### 1. 为什么我的GPU加速没有生效？

- 请确保已安装与 CUDA 版本匹配的 PyTorch GPU 版本（见"安装依赖"）
- 请检查您的 GPU 是否支持 CUDA
- 程序默认启用 GPU 加速，GPU 不可用时会自动回退到 CPU 处理
- 可使用 `--disable-gpu` 参数（CLI）或取消勾选"启用 GPU 加速"（GUI）强制使用 CPU

### 2. 支持哪些文件格式？

- 图像格式：JPG、JPEG、PNG、BMP、GIF、WebP、TIFF、TIF、HEIF、HEIC、AVIF、APNG
- 视频格式：MP4、AVI、MOV、MKV、WebM、FLV、MPG、MPEG、WMV
- 动图（按多帧处理）：GIF、WebP、APNG

### 3. 输出的字符画视频能否被硬件解码（硬解）？

可以。输出视频的编码器按**输出容器的扩展名**选择，而不是沿用输入视频的编码器，并统一附带硬件解码器要求的编码约束：

| 输出容器 | 视频编码器 | 编码档次 | 像素格式 |
| --- | --- | --- | --- |
| MP4 / AVI / MKV / MOV / FLV | H.264（`libx264`） | `high` | `yuv420p` |
| WebM | VP9（`libvpx-vp9`） | `0` | `yuv420p` |
| MPG / MPEG | MPEG-2（`mpeg2video`） | `main` | `yuv420p` |
| WMV | WMV2（`wmv2`） | 不指定 | `yuv420p` |

- 像素格式强制为 8bit 4:2:0（`-pix_fmt yuv420p`），宽高自动取偶（`scale=trunc(iw/2)*2:trunc(ih/2)*2`），MP4/MOV 额外加 `-movflags +faststart` 便于边下边播；
- WMV 容器没有硬件解码器支持的编码格式可用，因此输出 WMV2，该格式通常无法硬解，转换时会输出相应警告；
- 需要重新编码为其他格式时，可在自己的流程中调用 `src.utils.create_video(..., codec='编码器名')` 显式指定编码器（此时不再套用容器的编码档次，避免档次不被该编码器支持而报错）。

### 4. 视频转换失败或提示找不到 ffmpeg？

字符画视频的帧合成依赖 FFmpeg。请安装 FFmpeg 并将其加入系统 PATH（`ffmpeg -version` 能正常输出即表示可用）。

## 许可证

本项目采用MIT许可证，详细信息请查看[LICENSE](LICENSE)文件。
