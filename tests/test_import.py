"""The package and every subpackage import, and the core dependencies are installed."""

import importlib
import importlib.util

import pytest

SUBPACKAGES = [
    "data",
    "models",
    "training",
    "evaluation",
    "interpret",
    "synthesis",
    "adversarial",
    "federated",
    "deploy",
    "utils",
]

CORE_DEPENDENCIES = [
    "torch",
    "torchvision",
    "timm",
    "hydra",
    "omegaconf",
    "numpy",
    "pandas",
    "sklearn",
    "skimage",
    "xgboost",
    "imagehash",
    "PIL",
    "matplotlib",
    "optuna",
    "statsmodels",
    "torchmetrics",
    "open_clip",
    "peft",
    "cleanlab",
    "pytorch_grad_cam",
    "captum",
    "onnx",
    "onnxruntime",
    "gradio",
    "mlflow",
]


def test_package_version() -> None:
    import pharmachecked

    assert pharmachecked.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"pharmachecked.{name}")


@pytest.mark.parametrize("module", CORE_DEPENDENCIES)
def test_core_dependency_installed(module: str) -> None:
    assert importlib.util.find_spec(module) is not None, f"{module} is not installed"
