"""
ConversionWorker：在 QThread 中按 CLI 相同顺序构造并运行转换处理器。

方案 B（docs/gui_refactor_plan.md §6.3）：
- 不导入 char_art_converter.py（避免其模块级 SIGINT 注册劫持 GUI 进程）；
- 复刻 main() 的 try/except 骨架与退出码语义：成功 0 / 一般错误 1 /
  文件不存在 2 / 用户中断 130；
- 构造处理器前发预热阶段进度事件（§6.5 C5）。
"""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

from src.configs.message_config import ERROR_MESSAGES
from src.enums.file_type import FileType
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PREWARM_DESCRIPTION = "正在初始化 GPU 环境（首次运行可能较慢）…"


class ConversionWorker(QThread):
    """转换任务线程。构造一次运行一次，不可复用。"""

    progress = Signal(int, int, str, str, int)   # done, total, description, extra, position
    thread_count = Signal(int)                   # GIF/视频多线程帧处理的线程数（CLI 语义）
    finished_ok = Signal(int, dict)              # exit_code, payload(产物路径/错误信息)
    log_ready = Signal()                         # setup_logging 完成后通知主线程挂 GUI Handler

    def __init__(self, ns: Namespace, parent=None):
        super().__init__(parent)
        self.ns = ns
        self._processor = None

    def run(self) -> None:
        """与 CLI main() 保持一致的执行顺序与错误映射。"""
        setup_logging(verbose=bool(getattr(self.ns, 'debug', False)))
        self.log_ready.emit()

        # C5：构造处理器（含 torch 首次加载）前给出预热提示
        self.progress.emit(0, 0, PREWARM_DESCRIPTION, "", 0)

        try:
            file_type = FileType.from_path(Path(self.ns.input))
            if file_type == FileType.IMAGE:
                self._processor = ImageProcessor(self.ns)
            elif file_type == FileType.VIDEO:
                self._processor = VideoProcessor(self.ns)
            else:
                # CLI 对 TEXT/AUDIO 静默返回 0；GUI 刻意收紧为显式拒绝
                raise ValueError(ERROR_MESSAGES['unsupported_format'].format(
                    Path(self.ns.input).suffix or Path(self.ns.input).name,
                    ""))
            # GIF/视频才走多线程帧处理：主条(position 0) + 每线程一条(position 1..N)
            if file_type == FileType.VIDEO or (
                file_type == FileType.IMAGE and getattr(self._processor, 'is_animated', False)
            ):
                self.thread_count.emit(self._processor.max_workers)
            self._processor.start()
            self.finished_ok.emit(0, self._collect_outputs())
        except KeyboardInterrupt:
            logger.info("转换任务已被用户取消")
            self.finished_ok.emit(130, {})
        except FileNotFoundError as e:
            logger.error("输入文件不存在: %s", e)
            self.finished_ok.emit(2, {"error": str(e)})
        except Exception as e:
            logger.exception("转换任务失败")
            self.finished_ok.emit(1, {"error": str(e)})

    def cancel(self) -> None:
        """同时置位 should_stop 与 global_exit_flag（停止检查点两处都读）。"""
        p = self._processor
        if p is None:
            return
        p.should_stop = True
        flag = getattr(p, 'global_exit_flag', None)
        if flag is not None and hasattr(flag, 'value'):
            flag.value = True                # DummyFlag 场景
        else:
            p.global_exit_flag = True        # 布尔标志场景（args.exit_flag 路径）

    def _collect_outputs(self) -> Dict[str, Any]:
        """收集产物路径供成功态展示（处理器属性因类型而异）。"""
        payload: Dict[str, Any] = {"input": str(self.ns.input)}
        p = self._processor
        for key in ("output_dir", "text_path", "image_path", "video_path", "frame_path"):
            value = getattr(p, key, None)
            if value is not None:
                payload[key] = str(value)
        return payload
