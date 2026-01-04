# 字符画转换器 (Char Art Converter)

## 项目说明

字符画转换器是一个功能强大的命令行工具，用于将图像或视频转换为精美的字符画。该工具支持多种配置选项，包括字符密度、颜色模式、输出大小限制等，并提供GPU加速功能以提高处理效率。

主要功能：
- 支持多种图像和视频格式输入
- 提供多种字符密度级别可选
- 支持彩色和黑白字符画输出
- 可配置输出大小限制
- 支持GPU加速处理
- 支持多线程处理视频帧
- 可同时输出图像和文本格式的字符画

## 文件目录

```
char-art-converter-cmd/
├── char_art_converter.py    # 命令行工具入口
├── LICENSE                  # 许可证文件
├── README.md                # 项目说明文档
├── src/                     # 源代码目录
│   ├── configs/             # 配置模块
│   │   ├── audio_config.py  # 音频相关配置
│   │   ├── common_config.py # 通用配置
│   │   ├── image_config.py  # 图像相关配置
│   │   ├── message_config.py # 消息配置
│   │   └── video_config.py  # 视频相关配置
│   ├── enums/               # 枚举类型模块
│   │   ├── color_modes.py   # 颜色模式枚举
│   │   ├── file_type.py     # 文件类型枚举
│   │   └── save_modes.py    # 保存模式枚举
│   ├── processors/          # 处理器模块
│   │   ├── based_processor.py # 基础处理器
│   │   ├── image_processor.py # 图像处理
│   │   └── video_processor.py # 视频处理
│   └── utils/               # 工具函数模块
│       ├── audio_utils.py   # 音频工具
│       ├── char_art_utils.py # 字符画生成工具
│       ├── color_utils.py   # 颜色处理工具
│       ├── ffmpeg_utils.py  # FFmpeg工具
│       ├── file_utils.py    # 文件操作工具
│       ├── font_utils.py    # 字体处理工具
│       ├── format_utils.py  # 格式化工具
│       ├── gpu_utils.py     # GPU加速工具
│       ├── image_utils.py   # 图像处理工具
│       ├── logging_utils.py # 日志工具
│       ├── progress_bar_utils.py # 进度条工具
│       ├── save_uitls.py    # 保存工具
│       └── video_utils.py   # 视频处理工具
```

## 项目结构

项目采用模块化设计，主要分为以下几个核心模块：

1. **配置模块**：管理应用程序的各种配置参数，包括字符密度、颜色模式、默认设置等。
2. **枚举模块**：定义应用程序中使用的各种枚举类型，如颜色模式、文件类型、保存模式等。
3. **处理器模块**：实现核心的图像处理和视频处理逻辑。
4. **工具函数模块**：提供各种辅助功能，如字符画生成、颜色处理、文件操作、GPU加速等。

## 环境要求

- Python 3.8+
- Windows 10/11
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

### 2. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 3. 安装PyTorch（GPU版本）

PyTorch的GPU版本需要根据您的系统和CUDA版本选择合适的安装命令。请访问[PyTorch官方网站](https://pytorch.org/get-started/locally/)获取最新的安装命令。

示例（Windows + CUDA 11.8）：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 基本用法

```bash
python char_art_converter.py input.jpg
```

### 转换为彩色字符画

```bash
python char_art_converter.py input.gif --color-mode color
```

### 同时输出图像和文本文件

```bash
python char_art_converter.py input.png --with-text
```

### 使用GPU加速

```bash
python char_art_converter.py input.mp4 --enable-gpu
```

## 命令行参数

### 必需参数

| 参数    | 描述       |
|-------|----------|
| input | 输入图像文件路径 |

### 可选参数

| 参数                      | 描述                                                                                                       |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `-o, --output`          | 设置输出目录（如果不指定，将在输入文件同目录生成）                                                                                |
| `-d, --density`         | 字符密度级别 (默认: medium)<br>选项: low, medium, high                                                             |
| `-c, --color-mode`      | 颜色模式 (默认: color)<br>选项: grayscale, color, colorBackground                                                |
| `-l, --limit-size`      | 调整输入图片尺寸。不带参数时使用默认大小(若指定了字体大小则为其1/2，否则原图宽度的1/4和原图高度的1/6)，带两个参数时指定宽度和高度<br>格式: [LIMIT_WIDTH,LIMIT_HEIGHT] |
| `-t, --with-text`       | 同时输出字符画图像和文本文件                                                                                           |
| `-i, --with-image`      | 同时输出字符画视频/动图和字符画图像（仅当输入文件为视频或动图时有效）                                                                      |
| `--no-multithread`      | 禁用多线程处理动画帧（默认启用多线程）                                                                                      |
| `--disable-gpu`         | 禁用GPU并行计算加速，使用CPU处理（默认启用GPU）                                                                             |
| `--debug`               | 启用DEBUG级别日志输出                                                                                            |
| `--gpu-memory-limit MB` | 设置GPU内存限制（MB），默认使用配置文件中的值                                                                                |
| `--version`             | 显示版本信息                                                                                                   |

## 常见问题

### 1. 为什么我的GPU加速没有生效？

- 请确保已正确安装PyTorch的GPU版本
- 请检查您的GPU是否支持CUDA
- 可以使用`--disable-gpu`参数强制使用CPU处理

### 2. 支持哪些文件格式？

- 图像格式：JPG、JPEG、PNG、BMP、GIF、WebP、TIFF、TIF、HEIF、HEIC、AVIF、APNG
- 视频格式：MP4、AVI、MOV、MKV、WebM、FLV、MPG、MPEG、WMV

## 许可证

本项目采用MIT许可证，详细信息请查看[LICENSE](LICENSE)文件。