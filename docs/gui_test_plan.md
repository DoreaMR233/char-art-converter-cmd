# 字符画转换器 GUI 测试文档

> 版本：v1.0（方案稿）
> 配套文档：`docs/gui_refactor_plan.md`。本测试文档覆盖**GUI 新增功能**与**CLI 回归保障**两部分。
> 状态：仅文档，尚未实施。当前仓库不存在 `tests/` 目录，本方案为从零搭建。

---

## 1. 测试目标

| 目标 | 说明 |
|---|---|
| G1 CLI 零退化 | 证明 GUI 改造后，`char_art_converter.py` 的参数、输出、进度、退出码与改造前完全一致（约束 H1 的可验证形式） |
| G2 兼容层正确 | C1 信号守卫、C2 进度汇在两种模式下都行为正确 |
| G3 GUI 功能完整 | 文件选择/参数装配/转换/进度/取消/错误提示/日志全链路可用 |
| G4 可发布 | 双 spec 打包产物可启动、可工作 |

**测试原则**

1. 所有测试运行于 `.venv`（Python 3.12.1），不污染系统环境。
2. GUI 测试必须支持无人值守：设置 `QT_QPA_PLATFORM=offscreen`。
3. 转换类测试使用**微小固定夹具**（64×48 图片、2 秒短视频），控制单用例耗时；GPU 依赖的环境自动跳过。
4. CLI 回归测试通过 `subprocess` 调用真实入口文件，不 mock 内核——这是"CLI 保留"承诺的证据。

---

## 2. 测试环境准备

```powershell
cd "E:\Personal Documents\Codes\char-art-converter-cmd"

# 测试依赖（PySide6 已按改造文档 §4.2 安装）
& ".venv\Scripts\python.exe" -m pip install pytest pytest-qt

# 无头运行 GUI 测试
$env:QT_QPA_PLATFORM = "offscreen"

# 运行前自检
& ".venv\Scripts\python.exe" -m pytest --version
& ".venv\Scripts\python.exe" -c "import PySide6, pytestqt; print('ok')"
```

`pytest.ini`（新增，建议内容）：

```ini
[pytest]
testpaths = tests
markers =
    unit: 单元测试（兼容层/参数装配）
    cli:  CLI 端到端回归（真实子进程）
    gui:  GUI 功能测试（pytest-qt，offscreen）
    slow: 长耗时/打包冒烟
addopts = -ra
```

---

## 3. 测试分层与目录结构

```
tests/
  conftest.py                 # 全局夹具：生成图片/GIF/视频；ffmpeg 探测；Namespace 工厂
  unit/
    test_signal_guard.py      # C1 主线程守卫
    test_progress_sink.py     # C2 进度汇（sink 开/关双模式）
    test_namespace_builder.py # GUI 表单 → Namespace 映射（纯逻辑，不依赖 Qt）
  cli/
    test_cli_image.py         # CLI 静图/动图回归（T-2xx）
    test_cli_video.py         # CLI 视频回归
    test_cli_errors.py        # CLI 错误路径与退出码
  gui/
    test_window_basic.py      # 窗口/控件/默认值/文件过滤（pytest-qt）
    test_validation.py        # 输入校验与错误就近显示
    test_conversion_flow.py   # 静图/视频转换 + 进度信号 + 成功态
    test_cancel_flow.py       # 取消流程（130 语义）
    test_log_drawer.py        # 日志桥接
    test_theme.py             # 双主题与设计规范抽查
  smoke/
    test_build_artifacts.py   # 双 spec 打包冒烟（T-4xx，mark slow）
```

---

## 4. 测试夹具设计（tests/conftest.py）

| 夹具 | 生成方式 | 用途 |
|---|---|---|
| `sample_image` | PIL 新建 64×48 RGB 渐变图 → `tmp_path/test_input.png` | 静图转换 |
| `sample_gif` | PIL 保存 3 帧 GIF（每帧 64×48，duration=100ms） | 动图转换 |
| `sample_video` | `cv2.VideoWriter`（fourcc `MJPG` 或 `mp4v`）生成 320×240、30fps、约 60 帧的 `.avi` | 视频转换（避免夹具生成依赖外部 ffmpeg） |
| `unsupported_file` | 任意 `.txt` 文本文件 | 格式拒绝路径 |
| `ffmpeg_available` | 探测 `check_ffmpeg_available()`（复用 `src` 内函数，无副作用形式） | 视频用例缺 ffmpeg 时 `pytest.skip` |
| `gui_namespace` | `argparse.Namespace` 工厂，属性全集对齐改造文档 §7.1 | 参数装配单测 |

约束：夹具全部在 `tmp_path` 内生成，不写死绝对路径；视频用例在无 ffmpeg 环境自动 skip 并给出原因（`better-ffmpeg-progress` 首次运行会自动下载 ffmpeg，CI 机需注意首次网络开销）。

---

## 5. 测试用例清单

### 5.1 单元测试（unit，mark=unit）

| ID | 用例 | 前置 | 步骤 | 预期 |
|---|---|---|---|---|
| T-101 | C1：主线程注册信号 | — | 在主线程构造 `BasedProcessor` 子类实例（monkeypatch `init_pytorch_and_gpu` 与 `signal.signal` 记录调用） | `signal.signal` 被调用一次 |
| T-102 | C1：工作线程跳过信号注册 | — | 在 `threading.Thread` 内构造同款实例并 join | 不抛 `ValueError`；`signal.signal` 未被调用；构造成功 |
| T-103 | C2：无 sink 时行为与改造前一致 | `set_progress_sink(None)` | 调 `show_project_status_progress(10, "t")`，断言返回对象类型 | 返回 `tqdm` 实例；`with` 块内 `update(1)` 正常；stdout 捕获到进度输出 |
| T-104 | C2：有 sink 时返回鸭子进度条 | `set_progress_sink(cb)` 记录事件 | `with show_project_status_progress(10, "阶段A", unit='帧') as p: p.update(1); p.set_postfix_str("已提交 1/10")` | 返回对象具备 `update/set_postfix_str/__enter__/__exit__`；cb 收到 `(1,10,"阶段A",…)` 与 postfix 事件；stdout **无任何 tqdm 输出** |
| T-105 | C2：`show_value_file_save_progress` sink 模式短路 | 有 sink | 调用该函数（事件已置位） | 函数快速返回；sink 收到一次 `(0,1,desc,'')`；不创建 tqdm 线程 |
| T-106 | C2：`no_value_file_save_progress` sink 模式不启动动画线程 | 有 sink | 调用该函数 | 立即返回；无残留后台线程 |
| T-107 | C2：sink 清除后恢复 CLI 行为 | 先 set 再 `set_progress_sink(None)` | 重复 T-103 | 与 T-103 一致（可重复设置/清除） |
| T-108 | 参数装配：GUI 表单 → Namespace 全字段映射 | `gui_namespace` 工厂 | 对 §7.1 表格每一行断言属性名、类型、默认值 | 全部一致；`limit_size` 空输入→`[]`，两值→`[w,h]`；`enable_gpu` 复选框取反逻辑正确 |
| T-109 | 参数装配：TEXT/AUDIO 输入被显式拒绝 | `.txt` 文件 | 调 worker 前置校验 | 抛 ValueError，错误文案含"不支持"；**不得**静默返回 0（对照 CLI `main()` 的既有静默行为，GUI 刻意收紧） |
| T-110 | 取消置位：bool 型 global_exit_flag | 构造 Namespace（`exit_flag=False`）+ 处理器实例（monkeypatch 重初始化） | 调 `worker.cancel()` 逻辑（或等价函数） | `should_stop is True` 且 `global_exit_flag is True` |
| T-111 | 取消置位：DummyFlag(.value) 型 | 模拟 builtins 未设置路径 | 同上 | `should_stop is True` 且 `global_exit_flag.value is True` |

### 5.2 CLI 回归测试（cli，mark=cli）——G1 的证据链

统一通过 `subprocess.run([venv_python, "char_art_converter.py", ...])`，`cwd=项目根`，捕获 stdout/stderr/exit code。

| ID | 用例 | 命令要点 | 预期 |
|---|---|---|---|
| T-201 | `--help` / `--version` | `-h` 与 `--version` | 退出码 0；help 含全部参数；version 输出 `1.0.0` |
| T-202 | 静图转换成功（默认参数） | `sample_image`，加 `-t -i` | 退出码 0；输出目录出现 `.txt` 与图片产物；stdout 出现 tqdm 进度特征（`%`/进度条或帧描述） |
| T-203 | 三档密度 × 三色模式冒烟 | `-d low/medium/high -c grayscale/color/colorBackground` 组合抽样（≥5 组） | 全部退出码 0；txt 产物非空且字符集符合所选档位（low 产物只含 `" .:-=+*#%@"` 字符集子集） |
| T-204 | `--no-multithread` 路径 | 静图 + `--no-multithread` | 退出码 0；产物与多线程模式一致（字节级比较 txt） |
| T-205 | `-l/--limit-size` 尺寸生效 | `-l 32 24` | 退出码 0；产物图片尺寸 ≤ 32×24（按像素宽高校验） |
| T-206 | 动图（GIF）转换 | `sample_gif` + `-t -i` | 退出码 0；生成动图产物与 txt |
| T-207 | 视频转换 | `sample_video` + `-t`；无 ffmpeg 则 skip | 退出码 0；生成 txt 与视频产物；stdout 含"统计视频帧数"/帧进度描述 |
| T-208 | 输入文件不存在 | 随机不存在路径 | 退出码 2；stderr/stdout 含"文件"错误信息 |
| T-209 | 不支持格式 | `unsupported_file(.txt)` | 退出码 1；错误信息含"不支持"与支持格式列表 |
| T-210 | 非法 `-l` 参数 | `-l 32`、`-l 0 10` | 退出码 1；错误信息含 limit-size 提示 |
| T-211 | 输出目录不可写 | `-o` 指向无法创建/不可写路径（如盘符根/只读目录） | 退出码 1；错误信息含目录相关提示 |
| T-212 | `--debug` 日志 | 任一成功用例加 `--debug` | 退出码 0；stdout 含 `DEBUG`/`INFO` 日志行 |
| T-213 | Ctrl+C 中断 → 130 | 视频大循环中向子进程发送 `CTRL_BREAK_EVENT`（Windows） | 退出码 130；无损坏产物残留（临时目录已清理）；标记 `slow`，CI 环境不支持信号时 skip |
| T-214 | GUI 改造后回归锚点 | P1 兼容层合入后，将 T-201~T-212 全量重跑 | 结果与 P0 基线逐项一致（保留 P0 的 stdout/退出码快照做 diff） |

> T-214 是"CLI 保留"的验收门禁：改造合入前先跑一遍存基线，合入后逐项比对。

### 5.3 GUI 功能测试（gui，mark=gui，pytest-qt + offscreen）

| ID | 用例 | 步骤 | 预期 |
|---|---|---|---|
| T-301 | 窗口启动与默认值 | `qtbot` 启动 MainWindow | 标题正确；密度=medium、颜色模式=color、with_text/with_image=False；"开始转换"可用、"取消"不可见 |
| T-302 | 文件选择过滤 | 打开 FilePicker 的对话框逻辑（直接调其过滤串构建函数） | 图像过滤含 JPG/PNG/GIF/WebP/TIFF/HEIF/AVIF/APNG；视频过滤含 MP4/AVI/MOV/MKV/WebM/FLV/MPG/MPEG/WMV；`.txt` 不在过滤内 |
| T-303 | 拖放接受 | 对 FilePicker 派发含 `file:///…/test_input.png` 的 dropEvent | 输入框更新为该路径 |
| T-304 | 输入校验错误就近显示 | 填入 `.txt` 路径后点开始（或触发校验） | 不启动任务；输入框下方出现红色错误文案（图标+文字，颜色非唯一载体） |
| T-305 | 静图转换全流程 | 选 `sample_image` + `-t -i` → 点开始 | progress 信号按序到达；最终 finished(0)；成功态显示产物路径与"打开文件夹"；参数区恢复可用 |
| T-306 | 进度信号协议 | 记录 worker.progress 事件序列 | 每条含 (done,total,description,extra)；done ≤ total；total=0 事件被 UI 视为不确定进度；相邻事件间隔 ≥50ms（节流生效） |
| T-307 | 视频转换 + 阶段映射 | 选 `sample_video` → 开始 | 阶段文案出现"获取视频信息/统计视频帧数"等中文描述；finished(0) |
| T-308 | 取消流程（130 语义） | 视频转换中点击取消 | 按钮立即变"正在取消…"禁用；数秒内 finished(130)；UI 回到 IDLE；临时目录被清理；日志含"中断"记录 |
| T-309 | 取消后再转换 | T-308 结束后直接再跑一次静图 | 第二次转换成功 finished(0)——验证无残留停止标志污染 |
| T-310 | 运行中参数区禁用 | 转换进行中 | 密度/颜色/输入等控件 disabled；窗口关闭触发 closeEvent 拦截（先取消后退出，无崩溃） |
| T-311 | 错误路径 → 1/2 语义 | 输入不存在文件 / 输出目录非法 | finished(2)/(1)；错误框就近显示且含恢复路径文案 |
| T-312 | 日志抽屉 | debug=True 运行任一转换 | `log_ready` 后 GuiLogHandler 挂载；抽屉收到不含 ANSI 码的日志行；连续两次任务无重复挂载输出 |
| T-313 | torch 预热提示 | 首次转换前 worker 发预热阶段事件 | UI 显示"正在初始化 GPU 环境"类文案；转换完成后消失 |
| T-314 | 主题与规范抽查 | 切换 light/dark；遍历主要控件 | 主题生效；焦点态可见；抽查 2 组文字/背景色对比度 ≥4.5:1（theme.py 内自检函数）；图标为 SVG 资源非 emoji |
| T-315 | Namespace→内核一致性 | 用 T-305 相同表单设置生成 Namespace，与 `create_parser().parse_args([...等价CLI参数...])` 结果对比 | 所有字段值相等 |

### 5.4 打包冒烟（smoke，mark=slow）

| ID | 用例 | 步骤 | 预期 |
|---|---|---|---|
| T-401 | CLI exe 回归 | 构建 `char_art_converter.spec` → 运行 `dist\char_art_converter\char_art_converter.exe --version` 与一次静图转换 | 输出 `1.0.0`；转换成功退出码 0 |
| T-402 | GUI exe 启动 | 构建 `char_art_converter_gui.spec` → 启动 exe（`QT_QPA_PLATFORM=offscreen` 或人工点检） | 进程存活、无控制台窗口、无缺失 DLL 报错；人工点检项：窗口出现、选文件→转换→成功 |

---

## 6. 运行方式

```powershell
cd "E:\Personal Documents\Codes\char-art-converter-cmd"
$env:QT_QPA_PLATFORM = "offscreen"

# 全部（含 slow 打包冒烟，耗时最长）
& ".venv\Scripts\python.exe" -m pytest -v

# 日常快速回归（单元 + CLI + GUI，不含打包）
& ".venv\Scripts\python.exe" -m pytest -v -m "not slow"

# 仅 CLI 回归（P0 基线 / P1 合入后门禁）
& ".venv\Scripts\python.exe" -m pytest tests/cli -v

# 仅 GUI
& ".venv\Scripts\python.exe" -m pytest tests/gui -v

# 单用例
& ".venv\Scripts\python.exe" -m pytest tests/cli/test_cli_image.py::test_static_image_default -v
```

---

## 7. 通过标准（缺陷判定）

1. **门禁（必须全绿）**：T-101~T-111（兼容层）、T-201~T-212（CLI 回归）、T-301~T-312、T-314、T-315。
2. **条件通过**：视频类（T-207/213/307/308）在无 ffmpeg 环境允许 skip，但必须在至少一台有 ffmpeg 的 Windows 机器上全绿并留记录。
3. **慢速项**：T-401/402 在每次发版前执行；CI 每日执行。
4. **失败即缺陷**：任何 CLI 用例（T-2xx）在 GUI 改造后从绿变红，一律视为 H1 违约，禁止合入，无论 GUI 功能是否正常。

---

## 8. 回归策略

| 时机 | 执行范围 |
|---|---|
| P0 基线 | T-201~T-212 全量 + 保存基线快照（stdout/退出码） |
| P1 兼容层合入 | T-101~T-111 + T-2xx 全量（与基线 diff，即 T-214） |
| P2/P3 每个功能点 | 对应 T-3xx + T-2xx 快速集（T-202/203/208/209） |
| P4 主题 | T-314 + 全部 T-3xx |
| P5 发版 | 全量 + T-401/402 |

每次回归的 CLI 快照 diff 结果随提交说明留存（截图或文本均可）。

---

## 9. 附录：与改造文档的追溯矩阵

| 改造点（gui_refactor_plan.md） | 对应测试 |
|---|---|
| C1 信号守卫（based_processor.py:147） | T-101、T-102 |
| C2 进度汇（progress_bar_utils.py） | T-103~T-107 |
| C3 取消机制（worker.cancel 双置位） | T-110、T-111、T-308、T-309 |
| C4 日志桥接（GuiLogHandler） | T-312 |
| C5 torch 预热提示 | T-313 |
| §7.1 参数装配 | T-108、T-109、T-315 |
| §7.2 状态机 | T-305、T-308、T-310 |
| §7.3 进度信号协议/节流 | T-306 |
| §8 设计规范 | T-314 |
| §9 打包双 spec | T-401、T-402 |
| H1 CLI 保留 | T-2xx 全套 + T-214 门禁 |
