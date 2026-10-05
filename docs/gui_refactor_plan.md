# 字符画转换器 GUI 改造文档

> 版本：v1.0（方案稿）
> 状态：仅文档，尚未实施。本文所有代码片段均为**参考实现**，不直接修改任何源码。
> 适用范围：`E:\Personal Documents\Codes\char-art-converter-cmd`

---

## 1. 目标与范围

在**完整保留现有命令行模式**的前提下，为项目增加一套图形界面（GUI）：

- **CLI 零退化**：`char_art_converter.py` 的入口、参数、日志、进度条、退出码、打包产物全部不变。
- **GUI 增量叠加**：新增独立入口 `gui_app.py` 与 `gui/` 包，复用现有转换内核（processors/utils），不复制、不重写转换逻辑。
- **依赖隔离**：所有新增 pip 包安装到项目虚拟环境 `E:\Personal Documents\Codes\char-art-converter-cmd\.venv`；不改动现有 `requirements.txt`。

明确**不做**的事：

1. 不修改转换算法（字符映射、颜色、GPU/CPU 逻辑）。
2. 不修改 CLI 参数集与默认值（density/color-mode/limit-size 等语义不变）。
3. 不把 GUI 变成多任务并行系统（单任务模型，与 CLI 一致）。
4. 不动 `requirements.txt` 与 `char_art_converter.spec`。

---

## 2. 硬性约束（来自需求方）

| 编号 | 约束 | 落实方式 |
|---|---|---|
| H1 | 保留原命令行模式 | `char_art_converter.py` 与 `char_art_converter.spec` **零改动**（见 §5 双入口方案）；全部兼容层改动只发生在 `src/` 内部且对 CLI 行为不可见（见 §6 每处改动的"CLI 兼容性"证明） |
| H2 | 新 pip 包装在 `.venv` | 所有安装命令统一使用 `E:\Personal Documents\Codes\char-art-converter-cmd\.venv\Scripts\python.exe -m pip install ...`（见 §4） |
| H3 | 先文档后代码 | 本文档 + `docs/gui_test_plan.md` 经评审后再实施 |

---

## 3. 现状架构（改造基线）

```
char_art_converter.py            CLI 唯一入口
  ├─ create_parser()  :96        参数定义
  ├─ main(args=None)  :202       返回 int（0 成功 / 1 一般错误 / 2 文件不存在 / 130 用户中断）
  │    ├─ setup_logging(verbose=args.debug)   :242
  │    ├─ parsed_args.exit_flag = exit_flag   :245（模块级 bool 拷贝）
  │    ├─ FileType.from_path(input_path)      :240（IMAGE / VIDEO / TEXT / AUDIO）
  │    └─ ImageProcessor / VideoProcessor .start()   :253-258
  └─ if __name__ == '__main__': sys.exit(main())    :280-281

src/processors/
  ├─ based_processor.py         基类：校验、FFmpeg 检查、exit_flag 捕获、
  │                              torch 初始化(:210)、帧级进度循环(:368/:457/:501)
  ├─ image_processor.py         静图/动图处理，start() 阻塞 (:127)
  └─ video_processor.py         视频处理，start() 阻塞 (:129)；ffprobe 取视频信息；
                                帧计数进度条 (:199)；create_video (:342)

src/utils/
  ├─ progress_bar_utils.py      全部进度输出的汇聚点
  │    ├─ get_tqdm_kwargs()                  :44
  │    ├─ show_value_file_save_progress()    :88
  │    ├─ show_project_status_progress()     :269   ← 帧级进度唯一入口
  │    └─ no_value_file_save_progress()      :308（动画式，被 file_utils 起线程调用）
  ├─ validate_utils.py          validate_arguments(args)（文件存在/格式/输出目录/limit_size）
  ├─ logging_utils.py           setup_logging() :64 清空 root handler 后挂 ColoredFormatter→stdout
  ├─ file_utils.py              get_output_path() :45、save_file() :215
  └─ char_art_utils.py          字符画生成（内含 4 处像素级 tqdm：:296/:351/:489/:703）

char_art_converter.spec         PyInstaller onedir、console=True、excludes 含 'tkinter'
requirements.txt                CLI 运行依赖（本次不改）
```

对 GUI 而言的三个关键事实：

1. **处理器必须运行在工作线程**：`ImageProcessor.start()` / `VideoProcessor.start()` 为阻塞调用。
2. **`BasedProcessor.__init__` 在非主线程会崩**：`src/processors/based_processor.py:147` 直接调用 `signal.signal(signal.SIGINT, self.signal_handler)`，Python 规定 signal 注册只能在主线程执行。
3. **进度全部经 tqdm 写 stdout**：GUI 无法消费 tqdm 输出，需要一条可插拔的"进度汇"通路。

---

## 4. 技术选型与依赖管理

### 4.1 选型：PySide6（Qt for Python）

| 候选 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **PySide6** | 现代深色主题体系；原生文件对话框（含格式过滤）；QProgressBar/QThread/信号槽天然适配进度与跨线程上报；Windows 打包成熟（PyInstaller 官方 hook） | 安装包增大约 100–150MB（相对现有 torch 体量占比小） | **选用** |
| CustomTkinter | 轻量（约 +3MB） | 需自行维护主题；现 spec `excludes` 已排除 tkinter 需另改；控件覆盖少 | 备选 |
| pywebview | Web 前端自由度最高 | 引入 JS/Python 桥接复杂度，打包更繁琐 | 备选 |

版本：`PySide6>=6.7`（Python 3.12 支持需要 6.6+；`.venv` 当前为 **Python 3.12.1**）。

### 4.2 安装命令（全部在 `.venv` 内执行）

```powershell
# GUI 运行依赖
& "E:\Personal Documents\Codes\char-art-converter-cmd\.venv\Scripts\python.exe" -m pip install "PySide6>=6.7"

# 开发/测试依赖（见 docs/gui_test_plan.md）
& "E:\Personal Documents\Codes\char-art-converter-cmd\.venv\Scripts\python.exe" -m pip install pytest pytest-qt

# 验证
& "E:\Personal Documents\Codes\char-art-converter-cmd\.venv\Scripts\python.exe" -c "import PySide6; print(PySide6.__version__)"
```

### 4.3 依赖清单文件（新增，不触碰 requirements.txt）

| 文件 | 内容 | 用途 |
|---|---|---|
| `requirements-gui.txt` | `PySide6>=6.7` | GUI 运行依赖 |
| `requirements-dev.txt` | `pytest`、`pytest-qt`、`pyinstaller` | 测试与打包（建议用 `-r requirements.txt` 与 `-r requirements-gui.txt` 引用既有清单，避免重复） |

---

## 5. 总体改造架构：双入口并存

```
                 ┌──────────────────────────────────────────────┐
CLI 入口（不动）   │  char_art_converter.py  →  main(namespace)  │
                 └───────────────┬──────────────────────────────┘
                                  │ 复用
                                  ▼
        ┌─────────────────────────────────────────────┐
        │  转换内核（processors / utils / configs）     │  ← 仅两处兼容层小改：
        │                                             │     ① based_processor.py:147 信号守卫
        │                                             │     ② progress_bar_utils.py 进度汇
        └───────────────▲─────────────────────────────┘
                        │ 复用
GUI 入口（新增）  ┌──────┴──────────────────────────────────────┐
                 │  gui_app.py → gui/app.py → MainWindow        │
                 │  gui/worker.py  QThread 封装转换任务          │
                 │  gui/progress_bridge.py  进度汇 → Qt 信号     │
                 └──────────────────────────────────────────────┘
```

### 5.1 双入口对"保留 CLI"的落实

| 产物 | CLI（现状） | GUI（新增） |
|---|---|---|
| 源码入口 | `char_art_converter.py`（**零改动**） | `gui_app.py`（新增） |
| 启动命令 | `.venv\Scripts\python.exe char_art_converter.py <input> [参数]` | `.venv\Scripts\python.exe gui_app.py` |
| 打包 spec | `char_art_converter.spec`（**不改**，console=True） | `char_art_converter_gui.spec`（新增，console=False） |
| 参数语义 | `create_parser()` 为准 | GUI 表单字段一一映射到等价的 `argparse.Namespace` 属性 |
| 退出码 | 0/1/2/130 | worker 线程复刻同一套 0/1/2/130 语义（见 §6.3） |
| 进度输出 | tqdm → stdout（不变） | 进度汇 → Qt 信号（tqdm 静默） |

### 5.2 GUI 如何复用内核而不改 CLI 入口

关键决策：**GUI 不调用 `char_art_converter.main()`，而是在 worker 线程中按同样的顺序直接构造处理器**（方案 B，见 §6.3 对比）。理由：

- `main()` 内部的处理器实例不外露，GUI 无法拿到取消句柄；
- 复刻的只有 ~20 行"错误码映射"胶水，转换逻辑 100% 复用 `ImageProcessor` / `VideoProcessor`；
- `char_art_converter.py` 因此**一行都不用改**，CLI 保留约束（H1）得到最强保证；
- 同时避免了 `char_art_converter.py:79` 模块导入即注册 SIGINT 的问题——GUI 进程根本不需要导入该模块，不会劫持 GUI 进程的 Ctrl+C。

---

## 6. 详细改造点

### 6.1 C1 — `BasedProcessor` 信号注册主线程守卫（必须）

- **文件/位置**：`src/processors/based_processor.py:147`
- **现状**：`signal.signal(signal.SIGINT, self.signal_handler)`
- **问题**：`signal.signal()` 仅允许主线程调用。GUI 在 QThread 中构造处理器 → 直接抛 `ValueError: signal only works in main thread`。
- **改法**（参考实现）：

```python
# 替换 based_processor.py:147 的单行调用
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, self.signal_handler)
else:
    logger.debug("非主线程运行，跳过SIGINT注册（由上层负责取消）")
```

- **CLI 兼容性证明**：CLI 在 `__main__` 主线程构造处理器，条件恒真，行为与现状逐字节一致；仅 GUI worker 线程进入 else 分支。模块顶部已 `import threading`（`:182` 处有使用），无新增依赖。
- **对应测试**：`T-101`（见测试文档）。

### 6.2 C2 — 进度汇（Progress Sink）机制（必须）

- **文件/位置**：`src/utils/progress_bar_utils.py`
- **现状**：`show_project_status_progress`(:269)、`show_value_file_save_progress`(:88)、`no_value_file_save_progress`(:308) 是全部进度输出汇聚点，直接创建 tqdm 写 `sys.stdout`。调用方遍布 based_processor（:389/:477/:536）、video_processor(:199)、char_art_utils(:296/:351/:489/:703)、file_utils(:250/:272/:302)。
- **目标**：GUI 模式下让这些调用方**零改动**地把进度转成 Qt 信号；CLI 模式行为不变。
- **改法**（参考实现，纯增量）：

```python
# progress_bar_utils.py 模块级新增
from typing import Callable, Optional

# 回调签名：(done:int, total:int, description:str, extra:str) -> None
_progress_sink: Optional[Callable[[int, int, str, str], None]] = None

def set_progress_sink(sink: Optional[Callable[[int, int, str, str], None]]) -> None:
    """设置进度汇。None 恢复默认 tqdm 行为（CLI）。"""
    global _progress_sink
    _progress_sink = sink

def get_progress_sink():
    return _progress_sink


class _SinkProgressBar:
    """tqdm 的鸭子类型替身，仅实现调用方实际使用的接口。"""

    def __init__(self, total: int, description: str, unit: str = 'it'):
        self.total = total
        self.n = 0
        self.description = description
        self._last_emit = 0.0

    def update(self, n: int = 1) -> None:
        self.n += n
        self._emit('')

    def set_postfix_str(self, extra: str = '') -> None:
        self._emit(extra)

    def _emit(self, extra: str) -> None:
        if _progress_sink is not None:
            _progress_sink(self.n, self.total, self.description, extra)

    def refresh(self) -> None:      # 防御性兼容
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        self._emit('')
        return self

    def __exit__(self, *exc) -> None:
        pass


def _make_progress(total: int, description: str, unit: str = 'it', **kwargs):
    """有 sink 时返回替身对象，否则返回原 tqdm 对象（CLI 行为不变）。"""
    if _progress_sink is not None:
        return _SinkProgressBar(total, description, unit)
    from tqdm import tqdm as tqdm_func
    tqdm_kwargs = get_tqdm_kwargs(total, description, True, unit, kwargs.get('position', 0))
    if kwargs.get('hide_counter'):
        tqdm_kwargs['bar_format'] = '{desc} {percentage:3.0f}%|{bar}| {postfix}'
    return tqdm_func(**tqdm_kwargs)
```

`show_project_status_progress`(:269) 的返回语句改为 `return _make_progress(total, description, unit, **kwargs)`。

`show_value_file_save_progress`(:88) 与 `no_value_file_save_progress`(:308)：函数入口处加

```python
if _progress_sink is not None:
    _progress_sink(0, 1, description, '')   # 阶段事件
    save_completed.set()                    # 有值时直接置完成；无值时跳过动画
    return
```

- **节流**：进度事件在 `gui/progress_bridge.py` 侧节流（≥50ms 发一次），而不是在 sink 内部——保证 CLI 路径零感知。
- **CLI 兼容性证明**：`_progress_sink` 默认 `None`，`_make_progress` 走原 tqdm 分支；三个函数在 sink 为空时的控制流与改造前完全一致。CLI 回归用例 `T-201~T-206` 专门守护这一点。
- **对应测试**：`T-102`、`T-103`、`T-201`。

### 6.3 C3 — 取消机制（必须）

- **现状**：处理器把 `args.exit_flag`（模块级 bool 拷贝，`char_art_converter.py:245`）捕获为 `self.global_exit_flag`；停止检查点为 `src/processors/based_processor.py` 的 :381/:397/:404/:423/:470/:482/:531/:540，形式均为 `if self.should_stop or (hasattr(self,'global_exit_flag') and self.global_exit_flag): self.should_stop = True`。停止后由调用链抛 `KeyboardInterrupt`（char_art_utils.py 多处、video_processor.py:334、image_processor.py:286 等）→ CLI 的 `main()` 捕获 → 返回 130。
- **GUI 需求**：取消按钮 → 置位处理器两个标志 → 任务在帧循环边界停止 → worker 收到 KeyboardInterrupt → 映射为 130。

**方案对比与推荐**

| | 方案 A：`char_art_converter.py` 增加处理器注册表 | 方案 B：worker 自行构造处理器（**推荐**） |
|---|---|---|
| 做法 | `main()` 里 `_active_processor = proc`，新增 `request_cancel()` 供 GUI 调用 | `gui/worker.py` 复刻 main() 的 try/except 骨架，直接持有处理器实例 |
| char_art_converter.py | 需小改（约 10 行） | **零改动** |
| CLI 风险 | 极低但不为零 | 无 |
| 维护成本 | 退出码映射单一来源 | 映射逻辑重复 ~20 行 |
| 选择理由 | — | 与 H1 约束最契合；`main()` 每次运行还会执行 `parsed_args.exit_flag = exit_flag`(:245) 覆写传入值，方案 A 还需配套处理，复杂度反而更高 |

**方案 B 参考实现**（`gui/worker.py` 核心骨架，节选）：

```python
from PySide6.QtCore import QThread, Signal
from argparse import Namespace
import logging

from src.utils.logging_utils import setup_logging
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.enums.file_type import FileType

class ConversionWorker(QThread):
    progress = Signal(int, int, str, str)   # done, total, description, extra
    finished_ok = Signal(int, dict)         # exit_code, payload(产物路径等)
    log_ready = Signal()                    # setup_logging 完成后通知主线程挂 GUI Handler

    def __init__(self, ns: Namespace, parent=None):
        super().__init__(parent)
        self.ns = ns
        self._processor = None

    def run(self) -> None:
        setup_logging(verbose=self.ns.debug)
        self.log_ready.emit()
        try:
            file_type = FileType.from_path(Path(self.ns.input))
            if file_type == FileType.IMAGE:
                self._processor = ImageProcessor(self.ns)
            elif file_type == FileType.VIDEO:
                self._processor = VideoProcessor(self.ns)
            else:
                raise ValueError(ERROR_MESSAGES['unsupported_format'])  # TEXT/AUDIO 拒绝
            self._processor.start()
            self.finished_ok.emit(0, self._collect_outputs())
        except KeyboardInterrupt:
            self.finished_ok.emit(130, {})       # 用户取消
        except FileNotFoundError as e:
            self.finished_ok.emit(2, {"error": str(e)})
        except Exception as e:
            self.finished_ok.emit(1, {"error": str(e)})

    def cancel(self) -> None:
        p = self._processor
        if p is not None:
            p.should_stop = True
            flag = getattr(p, 'global_exit_flag', None)
            if flag is not None and hasattr(flag, 'value'):
                flag.value = True               # DummyFlag 场景
            else:
                p.global_exit_flag = True       # 布尔标志场景（args.exit_flag 路径）
```

- **要点**：
  1. `should_stop` 与 `global_exit_flag` **必须同时置位**（停止检查点两处都读）。
  2. `global_exit_flag` 可能是 bool（args 传入）或 DummyFlag 对象（builtins 未设置时的 :129 兜底），取消代码两种形态都要处理。
  3. GUI 不导入 `char_art_converter.py`，因此该模块的 `exit_flag`/SIGINT 状态完全不会影响 GUI；GUI 也**无需**在多次任务间重置任何模块级状态。
  4. `with_text` / `with_image` 任一为 True 才允许开始（CLI 的处理器同样要求二者至少其一？——以 `validate_arguments` 与处理器实际行为为准，GUI 侧做同等前置校验，报错文案与 CLI 一致）。
- **对应测试**：`T-104`（取消置位）、`T-301`（GUI 取消全流程）。

### 6.4 C4 — 日志桥接（必须）

- **现状**：`setup_logging()`（`src/utils/logging_utils.py:64`）清空 root 全部 handler，再挂单个 `ColoredFormatter` StreamHandler 到 stdout。
- **GUI 策略**：
  1. worker 的 `run()` 第一行调用 `setup_logging(verbose=debug)`（与 CLI 顺序一致），随后发 `log_ready` 信号；
  2. 主线程收到 `log_ready` 后，向 root logger 追加 `gui/widgets/log_drawer.py::GuiLogHandler`——**必须追加在 setup_logging 之后**，否则会被它清掉；
  3. `GuiLogHandler` 使用**无 ANSI 的普通 Formatter**（ANSI 色码只在 StreamHandler 的 ColoredFormatter 里产生，不会进入 GUI handler），内部写入 `queue.Queue`，由主线程 QTimer（如 100ms）批量取出追加到 QPlainTextEdit；
  4. 每次新任务都会重新 `setup_logging` → 主线程须在每次 `log_ready` 后重新挂 handler（旧实例置 `closed=True` 防重复输出）。
- **CLI 兼容性**：本改造只新增 GUI 侧类，`logging_utils.py` 零改动。
- **对应测试**：`T-302`。

### 6.5 C5 — torch/GPU 首次加载提示（建议，P3 阶段）

- **现状**：`based_processor.py:210` `init_pytorch_and_gpu()` 首次执行需 import torch，冷启动数秒。
- **改法**：GUI 不做源码改动，而是在 worker `run()` 中于构造处理器**前**发一条 progress 阶段事件（`total=0, done=0, description="正在初始化 GPU 环境（首次运行可能较慢）…"`），并可选地在 `gui/app.py` 启动时起一个低优先级预热线程 `import torch`。
- **CLI 兼容性**：不涉及 src 改动。

### 6.6 不改清单（防蔓延）

`char_art_converter.py`、`requirements.txt`、`char_art_converter.spec`、`src/configs/*`、`src/enums/*`、`src/processors/image_processor.py`、`src/processors/video_processor.py`、`src/utils/char_art_utils.py`、`src/utils/file_utils.py`、`src/utils/validate_utils.py`、`src/utils/logging_utils.py` —— 均不修改。

---

## 7. 新增 GUI 模块设计

```
gui_app.py                      # GUI 入口：freeze_support() → QApplication → MainWindow
gui/
  __init__.py
  app.py                        # 应用工厂：高DPI策略、主题加载、全局样式、资源注册
  theme.py                      # 语义颜色 token（light/dark 两套）→ 生成 QSS
  main_window.py                # 主窗口：布局、表单收集 → Namespace、状态机、信号连接
  worker.py                     # ConversionWorker(QThread)：§6.3 方案 B 骨架
  progress_bridge.py            # set_progress_sink() 对接：描述→阶段映射 + 50ms 节流 → worker.progress
  widgets/
    file_picker.py              # 输入文件选择：QFileDialog 过滤 + 拖放（dragEnter/dropEvent）
    param_panel.py              # 密度/颜色模式/限制尺寸/输出选项/高级折叠区
    preview_pane.py             # 原图缩略图 + 字符画文本预览（复用 load_font(DEFAULT_FONT_SIZE)）
    log_drawer.py               # 可折叠日志区 + GuiLogHandler(queue.Queue → QTimer 刷新)
  resources/
    icons/*.svg                 # 统一线性图标族（同一来源、同一线宽）
    theme_light.qss
    theme_dark.qss
```

### 7.1 参数装配（GUI 表单 → Namespace）

`main_window.py::build_namespace()` 生成与 CLI 等价的 `argparse.Namespace`，属性名与 `create_parser()` 的 dest 一一对应：

| GUI 控件 | Namespace 属性 | CLI 参数 | 默认值 |
|---|---|---|---|
| 输入文件选择/拖放 | `input` | positional | 必填 |
| 输出目录选择 | `output` | `-o/--output` | None（走内核默认） |
| 密度下拉 | `density` | `-d/--density` | `'medium'` |
| 颜色模式下拉 | `color_mode` | `-c/--color-mode` | `'color'` |
| 限制尺寸 宽×高 | `limit_size` | `-l/--limit-size` | `None`；空→`[]`，两值→`[w,h]` |
| 保存文本 复选框 | `with_text` | `-t/--with-text` | `False` |
| 保存图片 复选框 | `with_image` | `-i/--with-image` | `False` |
| 高级：单线程 | `no_multithread` | `--no-multithread` | `False` |
| 高级：禁用GPU | `enable_gpu` | `--disable-gpu`(dest 取反) | `True` |
| 高级：显存限额(MB) | `gpu_memory_limit` | `--gpu-memory-limit` | `0.8` |
| 高级：调试日志 | `debug` | `--debug` | `False` |

文件格式过滤由 `src/configs` 的 `ALL_SUPPORTED_FORMATS` 生成（图像 + 视频两类过滤串）；`FileType.TEXT/AUDIO` 被 GUI 显式拒绝并给出错误提示（CLI 的 `main()` 对这两种类型会静默返回 0，GUI 不能照搬这个静默行为）。

### 7.2 状态机（单任务模型）

```
IDLE ──[参数合法 + 点击开始]──▶ PREPARING ──[worker 构造处理器完成/进度首事件]──▶ RUNNING
 RUNNING ──[progress 事件]──▶ 更新进度条/阶段文案（参数区整体禁用）
 RUNNING ──[取消]──▶ CANCELING（按钮变"正在取消…"禁用）──[finished(130)]──▶ IDLE + 提示
 RUNNING ──[finished(0)]──▶ SUCCESS（展示产物路径 + "打开文件夹"）
 RUNNING ──[finished(1|2|异常)]──▶ ERROR（错误就近显示 + 恢复路径）
 任何状态 ──[窗口关闭请求且任务运行中]──▶ 先请求 cancel 再等待退出（closeEvent 拦截）
```

### 7.3 进度信号协议

`sink(done:int, total:int, description:str, extra:str)` 中 `description` 为现有中文描述（"获取视频信息："、"GPU生成字符文本"、"从临时文件夹加载帧图片" 等），`progress_bridge` 维护"描述→阶段标题"映射表，未命中时原样显示。`total==0` 视为不确定进度（QProgressBar range 0,0 总线型）。

---

## 8. 界面设计规范（依据 ui-ux-pro-max 检查表）

| 规则 | Qt 落地方式 |
|---|---|
| 语义化颜色 token、明暗主题成对设计 | `gui/theme.py` 定义 `surface/surface-alt/text/text-secondary/primary/primary-hover/danger/…` 双主题字典 → 生成 `theme_light.qss` / `theme_dark.qss`；QSS 中只允许引用 token 变量，禁止散落裸色值 |
| 文字对比度 ≥ 4.5:1 | 主题内每对前景/背景色附对比度校验（token 字典中注释实测值） |
| 单一主 CTA | 仅"开始转换"使用 `primary` 实心样式；"取消"为描边/次要样式，且仅 RUNNING 状态可见 |
| 错误就近显示 + 恢复路径 | 输入框下方红色文字+图标（红字配合文字说明，颜色不作唯一载体）；错误文案给出下一步动作（"请重新选择受支持的文件"） |
| 进度/加载/取消全程可见 | §7.2 状态机；不确定阶段用总线进度条+阶段文案 |
| 8px 间距节奏 | 全部布局 margin/spacing 取 8 的倍数（8/16/24/32） |
| 图标统一 | 一套 SVG 线性图标（同来源、同线宽 1.5px、同圆角风格）；禁用 emoji |
| 状态清晰 | 所有按钮/输入控件定义 hover/pressed/disabled/focus 四态 QSS；键盘焦点可见（focus 描边） |
| 动效 150–300ms 且尊重 reduced-motion | 主题切换/折叠动画 200ms；检测系统"减少动态效果"（QStyleHints 或 Windows 注册表）时直接瞬时切换 |
| 颜色不作唯一信息载体 | 状态提示一律 图标+文字；进度条配百分比文本 |

窗口建议：初始 960×640，最小 800×560；参数区为两列表单；日志抽屉默认收起，`debug=True` 时自动展开。

---

## 9. 打包方案（双 spec 并存）

### 9.1 新增 `char_art_converter_gui.spec`（参考关键差异）

```python
# 与现有 char_art_converter.spec 的差异点（其余 Analysis 参数照抄）
a = Analysis(['gui_app.py'], ...)
exe = EXE(pyz, a.scripts, [],
          name='char_art_converter_gui',
          console=False,          # 窗口应用，无控制台
          upx=True,
          ...)
# datas 增加 gui/resources 下的 qss/svg
# hiddenimports 保留现有清单；PySide6 由官方 hook 自动收集（无需手写）
```

- `gui_app.py` 顶部必须调用 `multiprocessing.freeze_support()`（Qt + PyInstaller 冻结环境要求）。
- `console=False` 时 tqdm 不再有 stdout 句柄——但 GUI 路径下 sink 生效、tqdm 根本不创建，双重保险。
- 保留原 `char_art_converter.spec` 不动，构建脚本生成**两个产物**：CLI exe（现状行为）与 GUI exe。

### 9.2 构建验证命令（.venv 内）

```powershell
& ".venv\Scripts\pyinstaller.exe" char_art_converter.spec          # CLI（回归）
& ".venv\Scripts\pyinstaller.exe" char_art_converter_gui.spec      # GUI
& "dist\char_art_converter\char_art_converter.exe" --version       # 应输出 1.0.0
& "dist\char_art_converter_gui\char_art_converter_gui.exe"         # 应出现窗口
```

---

## 10. 实施顺序与验收清单

| 阶段 | 内容 | 验收标准（全部由 `docs/gui_test_plan.md` 用例覆盖） |
|---|---|---|
| P0 基线 | 先跑通 CLI 回归测试，固化基线 | `T-2xx` 全绿 |
| P1 兼容层 | C1 信号守卫 + C2 进度汇 + 单元测试 | `T-101/102/103` 绿；CLI 回归仍全绿 |
| P2 最小 GUI | `gui_app.py` + 主窗口 + worker + 静图转换 | `T-301~306` 绿；CLI 回归仍全绿 |
| P3 体验完善 | 视频/动图、拖放、预览、日志抽屉、错误提示、torch 预热 | `T-307~313` 绿 |
| P4 主题打磨 | 双主题、图标、动效、reduced-motion、对比度检查 | 设计规范 §8 逐项核对 + `T-314` |
| P5 打包 | GUI spec + 双产物构建 | `T-401/402` 绿 |

---

## 11. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| torch 首次加载慢 | 用户误以为卡死 | §6.5 阶段提示 + 预热线程 |
| 大视频内存峰值 | 低配机卡顿 | GUI 默认建议 limit-size 1920×1080；暴露 gpu_memory_limit 高级项 |
| FFmpeg 缺失 | 视频转换失败 | 复用内核 `check_ffmpeg_available()`；GUI 捕获后给出"下载引导"文案而非裸堆栈 |
| 取消延迟 | 体验差 | 帧循环边界即可停止（现有检查点密集）；取消按钮立即进入 CANCELING 禁用态 |
| 进度事件洪水 | UI 卡顿 | progress_bridge 50ms 节流（§6.2） |
| 打包体积膨胀 | 分发成本 | PySide6 官方 hook 自动裁剪；可选 `--exclude-module QtWebEngine` 等剔除未用模块（P5 验证） |
| 主题对比度不达标 | 可用性 | token 字典内注释实测对比度，P4 用工具校验 |

---

## 12. 附录：改动文件总表

| 文件 | 动作 | 说明 |
|---|---|---|
| `src/processors/based_processor.py` | 修改（1 处） | :147 信号注册加主线程守卫（C1） |
| `src/utils/progress_bar_utils.py` | 修改（增量） | 新增 sink 机制与鸭子进度条（C2） |
| `gui_app.py`、`gui/**` | 新增 | GUI 入口与全部界面 |
| `char_art_converter_gui.spec` | 新增 | GUI 打包 |
| `requirements-gui.txt`、`requirements-dev.txt` | 新增 | GUI/测试依赖声明 |
| `docs/gui_refactor_plan.md`、`docs/gui_test_plan.md` | 新增 | 本文档与测试文档 |
| `char_art_converter.py`、`char_art_converter.spec`、`requirements.txt` | **不动** | CLI 保留（H1） |
