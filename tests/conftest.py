"""Shared test setup."""

from pharmachecked.utils.tracking import disable_mlflow_network_extras

# Applies to the test process and every subprocess it starts.
disable_mlflow_network_extras()
