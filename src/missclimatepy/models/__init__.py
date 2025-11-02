# src/missclimatepy/models/__init__.py
from __future__ import annotations
from typing import Any
from .rf import build_rf

_MODEL_BUILDERS = {
    "rf": build_rf,  # RandomForestRegressor
    # "xgb": build_xgb,  # future
    # "mlp": build_mlp,  # future
}

def build_model(name: str, **kwargs) -> Any:
    name = name.lower()
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_BUILDERS)}")
    return _MODEL_BUILDERS[name](**kwargs)
