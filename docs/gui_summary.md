# GUI 重构总结检查报告（task-4）

检查人：leader（task-4 owner）
检查日期：2026-10-05
检查范围：docs\gui_refactor_plan.md（计划）、docs\gui_test_plan.md（测试计划）、test 成员执行报告（m00296）、leader 独立复跑证据。

## 一、总结论

**GUI 重构验收通过（满足 CLI 保留约束 H1）。**

- 约束核查：仅 2 个计划允许修改的 tracked 文件被改动；3 个受保护文件（char_art_converter.py / char_art_converter.spec / requirements.txt）SHA256 与基线完全一致。
- 实现核查：C1–C5 全部落实；gui 包结构、GUI spec、依赖文件均符合计划 §7/§9。
- 测试核查：leader 独立复跑门禁 `pytest -m "not slow"` 得 **62 passed / 3 skipped / 3 deselected（EXIT=0，254.10s）**，与 test 成员报告一致；CLI 17/17、unit 11/11、GUI 全绿；T-214 基线 diff=0 经哈希独立证实。
- 2 个 slow 失败（T-213、T-401）复核后均判定为**测试设计/环境问题，非 CLI 回归、非实现缺陷**（详见 §四、§五）。

## 二、改动清单

| 类别 | 文件 | 状态 |
|------|------|------|
| 修改（计划允许） | src\processors\based_processor.py（+5 行，C1） | ✓ |
| 修改（计划允许） | src\utils\progress_bar_utils.py（+82/-13 行，C2） | ✓ |
| 新增（计划允许） | gui_app.py、gui\**（app/theme/main_window/worker/progress_bridge + widgets\{file_picker,param_panel,preview_pane,log_drawer} + resources\{icons 8×svg, theme_light.qss, theme_dark.qss}）、char_art_converter_gui.spec、requirements-gui.txt、requirements-dev.txt、pytest.ini | ✓ |
| 新增（计划允许） | docs\gui_refactor_plan.md、docs\gui_test_plan.md、docs\gui_summary.md、tests\**（fixtures/unit/gui/cli/smoke/baseline） | ✓ |
| 受保护文件 | char_art_converter.py、char_art_converter.spec、requirements.txt | 未改动 ✓ |
| 受保护目录 | src\configs、src\enums、src\processors\image_processor.py、video_processor.py、src\utils\{char_art_utils,file_utils,validate_utils,logging_utils}.py | 未改动 ✓ |

受保护文件基线 SHA256（2026-10-05 复核，全部一致）：

- char_art_converter.py = 25ABC4CFD137446DA584D8A56DF5BF9840DEB20F7596A0C48AD6AFB74A9498E7
- char_art_converter.spec = 46971B8CD8659F92B227A12DE38CCC099A2B327FB76EFDD185439C04747038F5
- requirements.txt = 08D44FA98B9C22C9160904237964EE40D4FB75488D836E536010E69AF6006B1B

## 三、实现核查表

| 检查项 | 证据 | 结论 |
|--------|------|------|
| C1 信号注册主线程守卫 | src\processors\based_processor.py:147：主线程才 `signal.signal(signal.SIGINT, self.signal_handler)`，else 分支 `logger.debug("非主线程运行，跳过SIGINT注册（由上层负责取消）")` | PASS |
| C2 进度 sink 增量改造 | src\utils\progress_bar_utils.py：`_progress_sink` + set/get_progress_sink + _SinkProgressBar + _make_progress；show_value_file_save_progress 与 no_value_file_save_progress 均 sink 短路 `_progress_sink(0,1,description,'')` + save_completed.set() + return；show_project_status_progress 走 _make_progress | PASS |
| C3 worker 无 CLI 依赖 + 退出码 | gui\worker.py：仅导入 src.configs / src.enums / src.processors.{image,video}_processor / src.utils.logging_utils，全 gui\ 包 grep 无 char_art_converter 导入；退出码 0/1/2/130 映射正确；cancel() 同时置 should_stop 与 global_exit_flag（.value 与 bool 两形态） | PASS |
| C4 GUI 日志管道 | gui\widgets\log_drawer.py：纯文本格式（无 ANSI）+ queue.Queue + QTimer(100ms) 泵取 + QPlainTextEdit(maxBlockCount=2000)；mount() 先关闭并移除旧 handler 再挂新 handler（log_ready 后重挂） | PASS |
| C5 预热进度事件 | gui\worker.py run()：构造处理器前 emit progress(0,0,"正在初始化 GPU 环境（首次运行可能较慢）…","") | PASS |
| gui 包结构 | 与计划 §7 逐文件一致（6 顶层模块 + 4 widgets + 8 svg + 2 qss） | PASS |
| GUI spec | char_art_converter_gui.spec：Analysis(['gui_app.py'])、datas=gui/resources、hiddenimports 含 torch/PySide6/cv2 等、excludes tkinter 等、console=False | PASS |
| GUI 入口 | gui_app.py：顶部 multiprocessing.freeze_support()，main()=create_app→apply_theme(dark=False)→show→exec | PASS |
| 进度桥 | gui\progress_bridge.py：THROTTLE_MS=50、15 条阶段规则、attach/detach/flush；main_window.py:254 attach、:269 mount、:283-284 flush+detach、:329-330 unmount+detach | PASS |
| 依赖 | requirements-gui.txt=PySide6>=6.7；requirements-dev.txt 含 -r requirements.txt/-r requirements-gui.txt/pytest>=8.0/pytest-qt>=4.4/pyinstaller>=6.0 | PASS |
| pytest 配置 | pytest.ini：testpaths=tests、addopts=-ra、markers=unit/cli/gui/slow | PASS |

## 四、测试结果

### 4.1 leader 独立复跑（权威门禁）

命令：`$env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONIOENCODING="utf-8"; .venv\Scripts\python.exe -m pytest -m "not slow" -q`

**结果：62 passed, 3 skipped, 3 deselected in 254.10s（EXIT=0）**

- CLI：17/17 全绿（含 T-214 基线对比）
- unit：11/11 全绿（cancel_flag / namespace_builder / progress_sink / signal_guard）
- GUI：全部通过（validation / window_basic / error_handling / theme / conversion_flow 图片路径等）
- skip 3 项：T-307/308/309（offscreen 下 GUI 视频 worker 挂起，已知问题，Lead 批准跳过，CLI T-207 已覆盖视频链路）

注：首次复跑（未设 PYTHONIOENCODING）出现 5 个 CLI 中文断言失败，经确认为子进程 GBK 输出被测试按 utf-8+replace 解码所致的环境伪失败；设置 PYTHONIOENCODING=utf-8 后全部通过，与 test 成员报告完全一致。

### 4.2 test 成员全量报告（m00296）对照

总 68 = 62 通过 / 2 失败 / 4 跳过。门禁 62/3/3 EXIT=0，CLI 17/17、unit 11/11。slow：T-213 FAILED、T-401 FAILED、T-402 SKIPPED。与 leader 复跑结论一致。

### 4.3 基线核查（T-214）

tests\baseline\cli_baseline.txt（108 行 / 19 场景，3757B）与 build\baseline_before.txt 的 SHA256 均为 61F520BE04D850F9CFA4FC41454470D2D5BA7577CA8FD031EA3DD9B89D8C8B36 → 逐字节相同，**CLI 输出零回归（H1 满足）**。

### 4.4 slow 失败复核

**T-213（tests\cli\test_cli_video.py::test_interrupt_video_returns_130）——复核结论：测试设计缺陷，非 CLI 回归。**

- 用例做法：等进度 marker → sleep 3 → `os.kill(proc.pid, signal.CTRL_BREAK_EVENT)` → 期望 returncode==130。
- CLI 实际中断契约（char_art_converter.py，受保护未改动）：:79 仅注册 `signal.signal(signal.SIGINT, global_signal_handler)`；130 仅来自 KeyboardInterrupt（:260-265）或处理器启动前 exit_flag 已置位（:249-251）；2 秒强制退出计时器走 sys.exit(1)。
- 关键矛盾：Windows 下 CTRL_BREAK_EVENT 投递为 **SIGBREAK**，不是 SIGINT；CLI 未注册 SIGBREAK 处理器 → 行为由 OS 默认决定，任何环境下都不可能稳定得到 130。
- 跨环境漂移证据：test 成员环境子进程 240s 不退出（TimeoutExpired）；leader 复跑子进程 6.42s 以 rc=1 退出（assert 1 == 130）。两种失败形态均指向"信号未被 CLI 的 SIGINT 优雅路径接管"。
- 结论：CLI 的 Ctrl+C（SIGINT）优雅中断实现正确且受保护未动；本用例选错信号类型。建议（下一迭代）：子进程以 CREATE_NEW_PROCESS_GROUP 创建并用 CTRL_C_EVENT（SIGINT）验证 130 契约；或保留 CTRL_BREAK 但断言 0xC000013A（STATUS_CONTROL_C_EXIT，3221225786）验证硬中断。**不构成 H1 违约。**

**T-401（tests\smoke\test_build_artifacts.py::test_cli_exe_build_artifact）——复核结论：环境限制（沙箱 TEMP），非构建产物缺陷。**

- 失败现象：`dist\char_art_converter.exe --version` rc=4294967295，stderr=[PYI-22724:ERROR] Could not create temporary directory!（test 成员环境报 PYI-16564，同类 bootloader 错误）。
- 复核实验：产物存在（1,954,992,015 字节，构建于 2026-07-25）；在沙箱内将 TEMP/TMP 重定向到工作区 build\ 目录后运行，输出 `char_art_converter.exe 1.0.0`，**EXIT=0**。
- 结论：exe 本体与打包健康；失败原因是执行沙箱对子进程写入系统 TEMP（C:\Users\ruiru\AppData\Local\Temp）的限制。发版前需在正常控制台环境复核一次 `--version`。

**T-402（GUI exe 构建产物）**：char_art_converter_gui.exe 未构建，按计划 §7 发版前执行 pyinstaller char_art_converter_gui.spec。当前 skip 合规。

## 五、遗留问题

| # | 问题 | 性质 | 建议 |
|---|------|------|------|
| 1 | T-213 信号类型错误（CTRL_BREAK vs SIGINT） | 测试设计缺陷 | 下迭代改用 CTRL_C_EVENT + CREATE_NEW_PROCESS_GROUP，或改断言 0xC000013A |
| 2 | T-401 沙箱 TEMP 限制 | 执行环境限制 | 发版前在正常控制台复核 exe --version |
| 3 | T-307/308/309 offscreen GUI 视频 worker 挂起 | 已知问题（Lead 批准 skip） | 在带 ffmpeg 的实体/真窗口环境补跑 GUI 视频链路；CLI T-207 已覆盖视频功能 |
| 4 | T-402 GUI exe 未构建 | 未到发版阶段 | 发版前构建并执行 T-402 |
| 5 | pytest-qt 未安装（pip 被审批拒绝） | 环境 | GUI 测试现用 PySide6 直连夹具；有条件时装 pytest-qt 重跑对照 |
| 6 | 根目录未跟踪残留 | 卫生 | pip-metadata-*/pip-unpack-*/（9 个）、.smoke/、build/、dist/、pytest-of-ruiru/、pytest_*.log 待清理 |

## 六、验收结论

**验收通过。** 四项核查全部完成：

1. 约束核查 ✓ —— 受保护文件零改动、改动范围严格限定在计划 §6.6/§12 允许清单内；
2. 实现核查 ✓ —— C1–C5 与文档要求逐条对应落实；
3. 测试核查 ✓ —— 门禁 62/3/3 复跑通过、CLI 17/17、基线 diff=0（H1 CLI 零退化成立）；
4. 遗留复核 ✓ —— T-213 判定为测试信号选型错误、T-401 判定为沙箱环境限制，均给出可执行建议，均不阻塞验收。

CLI 保留约束（H1）结论：**满足**。
