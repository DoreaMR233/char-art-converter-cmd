from _typeshed import Incomplete
from typing import Any

logger: Incomplete

def init_pytorch_and_gpu() -> tuple[Any | None, bool, Any | None]: ...
def setup_gpu_memory_limit(torch: Any | None, torch_cuda_available: bool, gpu_memory_limit: float | None = None) -> None: ...
