

"""
GPU加速工具模块

该模块提供了GPU加速相关的工具函数，负责PyTorch和CUDA的初始化、
GPU内存管理和优化，为字符画转换提供硬件加速支持。

主要功能：
- PyTorch和CUDA环境初始化
- GPU设备检测和信息获取
- GPU内存限制设置和优化
- 内存分配策略配置
- 错误处理和降级机制

依赖：
- PyTorch（可选）：用于GPU加速计算
- logging：用于日志记录

"""
import logging
from typing import Tuple, Optional, Any

from src.configs import WARNING_MESSAGES, GPU_MEMORY_LIMIT

# 初始化日志器
logger = logging.getLogger(__name__)

# GPU并行计算支持
# 全局GPU相关变量

def init_pytorch_and_gpu() -> Tuple[Optional[Any], bool, Optional[Any]]:
    """
    初始化PyTorch和GPU环境
    
    尝试导入PyTorch并检查CUDA可用性，返回PyTorch模块、CUDA可用性和默认设备
    
    Returns:
        Tuple[Optional[Any], bool, Optional[Any]]: 
        - PyTorch模块对象，如果导入失败则为None
        - 布尔值，表示CUDA是否可用
        - 默认CUDA设备对象，如果不可用则为None
    """

    local_cuda_available: bool = False
    local_device: Optional[Any] = None
    local_torch: Optional[Any]
    
    try:
        import torch as local_torch
        # 类型断言：local_torch 不会为 None
        assert local_torch is not None
        try:
            if hasattr(local_torch, 'cuda') and hasattr(local_torch.cuda, 'is_available'):
                local_cuda_available = local_torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"{WARNING_MESSAGES['gpu_check_error'].format(e)}")
        
        if local_cuda_available:
            try:
                # 获取CUDA版本信息
                assert local_torch is not None
                cuda_version = getattr(local_torch.version, 'cuda', '未知')
                
                logger.info(f"GPU加速可用: PyTorch {getattr(local_torch, '__version__', '未知版本')}, CUDA {cuda_version}")
                
                # 安全地获取GPU数量
                num_gpus = 0
                try:
                    assert local_torch is not None
                    if hasattr(local_torch.cuda, 'device_count'):
                        num_gpus = local_torch.cuda.device_count()
                    logger.info(f"可用GPU数量: {num_gpus}")

                    # 遍历所有 GPU 并打印信息
                    for i in range(num_gpus):
                        try:
                            # 获取当前 GPU 的属性
                            assert local_torch is not None
                            if hasattr(local_torch.cuda, 'get_device_properties'):
                                props = local_torch.cuda.get_device_properties(i)
                                logger.info(f"GPU {i}:")
                                logger.info(f"  名称: {getattr(props, 'name', '未知')}")
                                logger.info(f"  计算能力: {getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}")
                                logger.info(f"  总全局内存: {getattr(props, 'total_memory', 0) / (1024**2):.1f} MB")
                                logger.info(f"  多处理器数量: {getattr(props, 'multi_processor_count', '未知')}")
                        except Exception as e:
                            logger.warning(f"{WARNING_MESSAGES['gpu_info_error'].format(i, e)}")
                except Exception as e:
                    logger.warning(WARNING_MESSAGES['gpu_count_error'].format(e))
                
                # 设置默认设备
                try:
                    assert local_torch is not None
                    local_device = local_torch.device('cuda:0')
                except Exception as e:
                    logger.warning(WARNING_MESSAGES['default_device_error'].format(e))
            except Exception as e:
                logger.warning(WARNING_MESSAGES['gpu_details_init_error'].format(e))
        else:
            logger.info("GPU加速不可用: CUDA未检测到或无法使用")
    except ImportError:
        local_torch = None
        local_cuda_available = False
        logger.info("提示: PyTorch未安装，将使用CPU模式运行。")
    except Exception as e:
        local_torch = None
        local_cuda_available = False
        logger.warning(WARNING_MESSAGES['pytorch_init_error'].format(e))
    
    return local_torch, local_cuda_available, local_device

def setup_gpu_memory_limit(torch: Optional[Any], torch_cuda_available: bool, 
                          gpu_memory_limit: Optional[float] = None) -> None:
    """
    设置GPU内存限制
    
    配置PyTorch的GPU内存使用限制和优化内存分配策略
    
    Args:
        torch: Optional[Any] PyTorch模块对象
        torch_cuda_available: bool 布尔值，表示CUDA是否可用
        gpu_memory_limit: Optional[float] 可选的GPU内存使用限制（0.0-1.0之间的浮点数，表示百分比）
    """
    # 设置GPU内存限制和优化
    if torch is not None and torch_cuda_available:
        try:
            # 获取设备信息
            device = torch.device('cuda:0')
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            if gpu_memory_limit is not None:
                # 使用用户指定的内存限制（百分比）
                memory_limit = total_memory * gpu_memory_limit
                logger.info(f"使用用户指定的GPU内存限制: {gpu_memory_limit*100:.0f}%")
            else:
                # 使用配置文件中的默认限制
                
                memory_limit = int(total_memory * GPU_MEMORY_LIMIT)
                logger.info(f"使用默认GPU内存限制: {GPU_MEMORY_LIMIT*100:.0f}%")
            
            # 确保内存限制不超过可用内存的90%
            safe_memory_limit = min(memory_limit, int(total_memory * 0.9))
            
            # 设置内存限制
            try:
                # 尝试使用PyTorch的内存管理API
                if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'CUDAPoolAllocator'):
                    # 对于较新版本的PyTorch
                    memory_fraction = gpu_memory_limit if gpu_memory_limit is not None else 0.8  # 默认使用80%
                    torch.cuda.memory.set_per_process_memory_fraction(memory_fraction)
                    logger.info(f"GPU内存限制设置为: {safe_memory_limit / 1024**3:.1f}GB (总可用: {total_memory / 1024**3:.1f}GB)")
                else:
                    # 使用PyTorch的内存分配器设置
                    logger.info(f"设置GPU内存使用建议限制: {safe_memory_limit / 1024**3:.1f}GB (总可用: {total_memory / 1024**3:.1f}GB)")
            except Exception as e:
                logger.warning(WARNING_MESSAGES['gpu_memory_limit_failed'].format(e))
            
            # 配置更高效的内存分配策略
            try:
                # 使用PyTorch推荐的内存管理方法
                torch.cuda.empty_cache()  # 释放缓存内存
                logger.info("GPU内存缓存已清理")
            except Exception as e:
                logger.warning(WARNING_MESSAGES['gpu_cache_clean_failed'].format(e))
            
        except Exception as e:
            logger.warning(WARNING_MESSAGES['gpu_memory_limit_failed'].format(e))
            # 即使内存限制设置失败，仍尝试使用GPU（只是不限制内存）
            logger.info("继续使用GPU，但不设置内存限制")

