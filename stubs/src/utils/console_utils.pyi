from _typeshed import Incomplete
from typing import Any, Optional

logger: Incomplete
CTRL_C_EVENT: int
CTRL_BREAK_EVENT: int
CTRL_CLOSE_EVENT: int
CTRL_LOGOFF_EVENT: int
CTRL_SHUTDOWN_EVENT: int
_console_handler: Optional[Any]

def _handle_console_event(ctrl_type: int) -> bool: ...
def install_console_interrupt_handler() -> bool: ...
