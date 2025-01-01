from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path

from rapid_table.logger import get_logger

root_dir = Path(__file__).resolve().parent
logger = get_logger("params")

@dataclass
class BaseConfig:
    model_type: str = "slanet-plus"
    model_path: str = str(root_dir / "models" / "slanet-plus.onnx")
    use_cuda: bool = False
    device: str = "cpu"
    encoder_path: str = None
    decoder_path: str = None
    vocab_path: str = None


def accept_kwargs_as_dataclass(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 2 and isinstance(args[1], cls):
                # 如果已经传递了 ModelConfig 实例，直接调用函数
                return func(*args, **kwargs)
            else:
                # 提取 cls 中定义的字段
                cls_fields = {field.name for field in fields(cls)}
                # 过滤掉未定义的字段
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in cls_fields}
                # 发出警告对于未定义的字段
                for k in (kwargs.keys() - cls_fields):
                    logger.warning(f"Warning: '{k}' is not a valid field in {cls.__name__} and will be ignored.")
                # 创建 ModelConfig 实例并调用函数
                config = cls(**filtered_kwargs)
                return func(args[0], config=config)
        return wrapper
    return decorator