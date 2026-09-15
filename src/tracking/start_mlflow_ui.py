"""Start the MLflow UI connected to this project's tracking database."""

import subprocess
import sys

from mlflow_config import ARTIFACT_URI, TRACKING_URI


if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            TRACKING_URI,
            "--default-artifact-root",
            ARTIFACT_URI,
        ],
        check=True,
    )
