"""
进度桥接：把内核进度汇（progress_bar_utils 的 sink 回调）转成 Qt 信号。

- 描述 → 阶段标题映射（未命中时原样显示，§7.3）；
- 50ms 节流在桥接侧完成（§6.2），内核 sink 本身不过滤事件：
  事件流自身驱动刷新（间隔不足时合并为最新事件，下一条通过时间门的
  事件立即发出）；任务结束时由主线程调用 flush() 补发尾部事件；
- 任务结束时必须 detach()，恢复内核默认 tqdm 行为。
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from src.utils.progress_bar_utils import set_progress_sink

THROTTLE_MS = 50

# (前缀, 阶段标题)，按顺序匹配，未命中时原样显示
_PHASE_RULES: List[Tuple[str, str]] = [
    ("正在初始化 GPU 环境", "初始化 GPU 环境"),
    ("获取视频信息：", "获取视频信息"),
    ("统计视频帧数", "统计视频帧数"),
    ("获取视频编码器信息", "获取视频编码器信息"),
    ("处理视频帧", "处理视频帧"),
    ("处理帧", "处理帧"),
    ("从临时文件夹加载帧图片", "加载帧图片"),
    ("GPU生成字符文本", "GPU 生成字符文本"),
    ("CPU生成字符文本", "CPU 生成字符文本"),
    ("GPU生成", "GPU 生成"),
    ("CPU生成", "CPU 生成"),
    ("动画图像合成", "合成动画"),
    ("保存", "保存产物"),
]


def phase_title(description: str) -> str:
    """描述 → 阶段标题。纯函数，便于单测。"""
    text = (description or "").strip()
    for prefix, title in _PHASE_RULES:
        if text.startswith(prefix):
            return title
    return text or "处理中"


class ProgressBridge(QObject):
    """持有内核 sink 回调，节流后转发为 Qt 信号。"""

    progress = Signal(int, int, str, str, int)   # done, total, description, extra, position

    def __init__(self, parent=None, throttle_ms: int = THROTTLE_MS):
        super().__init__(parent)
        self._throttle_ms = max(10, int(throttle_ms))
        self._pending: Dict[int, Tuple[int, int, str, str]] = {}
        self._last_emit = 0.0
        self._lock = threading.Lock()
        self.attached = False

    def attach(self) -> None:
        """安装内核进度汇。重复调用幂等。"""
        if not self.attached:
            set_progress_sink(self._on_event)
            self.attached = True

    def detach(self) -> None:
        """卸载进度汇，恢复内核默认 tqdm 行为（CLI 语义）。"""
        if self.attached:
            set_progress_sink(None)
            self.attached = False
        with self._lock:
            self._pending = {}

    def _on_event(self, done: int, total: int, description: str, extra: str, position: int) -> None:
        """内核 sink 回调（worker 线程）：每个 position 保留最新事件，≥50ms 批量发出。"""
        with self._lock:
            self._pending[int(position)] = (int(done), int(total), str(description), str(extra))
            now = time.monotonic()
            if now - self._last_emit >= self._throttle_ms / 1000.0:
                self._flush_locked()

    def flush(self) -> None:
        """主线程在任务结束/取消时调用，补发未发出的尾部事件。"""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        pending = sorted(self._pending.items())
        self._pending = {}
        self._last_emit = time.monotonic()
        for position, (done, total, description, extra) in pending:
            self.progress.emit(done, total, description, extra, position)
