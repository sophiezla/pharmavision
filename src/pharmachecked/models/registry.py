"""Model registry: every model is built from its config by a registered builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

Builder = Callable[[Any], nn.Module]
_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator registering a builder ``fn(model_cfg) -> nn.Module`` under ``name``."""

    def wrap(fn: Builder) -> Builder:
        if name in _REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return wrap


def build_model(model_cfg: Any) -> nn.Module:
    """Build the model named by ``model_cfg.builder`` (or ``model_cfg.name``).

    Raises:
        KeyError: If no builder is registered under that name.
    """
    _import_builtin_models()
    name = model_cfg.get("builder") or model_cfg.get("name")
    if name not in _REGISTRY:
        raise KeyError(f"no model registered as {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](model_cfg)


def registered() -> list[str]:
    """Names of all registered models."""
    _import_builtin_models()
    return sorted(_REGISTRY)


def _import_builtin_models() -> None:
    """Import the modules whose builders self-register."""
    from pharmachecked.models import cnn_small  # noqa: F401
