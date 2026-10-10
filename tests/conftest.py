"""Shared test setup."""

from pathlib import Path

import pytest

from pharmachecked.data.toy import ToyData, make_toy_dataset
from pharmachecked.utils.tracking import disable_mlflow_network_extras

# Applies to the test process and every subprocess it starts.
disable_mlflow_network_extras()


@pytest.fixture(scope="session")
def toy(tmp_path_factory: pytest.TempPathFactory) -> ToyData:
    """The 64-image procedurally generated toy set (tests/fixtures convention, built on the fly)."""
    return make_toy_dataset(Path(tmp_path_factory.mktemp("toy")))
