"""
字符画转换器 GUI 包

按 docs/gui_refactor_plan.md §7 组织：
- app.py            应用工厂（QApplication、主题、全局样式）
- theme.py          语义颜色 token（light/dark）→ QSS
- main_window.py    主窗口（表单 → Namespace、状态机、信号连接）
- worker.py         ConversionWorker(QThread)（方案 B）
- progress_bridge.py  进度汇对接（50ms 节流 + 阶段映射）
- widgets/          文件选择、参数面板、预览、日志抽屉
- resources/        SVG 图标与两套 QSS 主题文件
"""

__version__ = "1.0.0"
