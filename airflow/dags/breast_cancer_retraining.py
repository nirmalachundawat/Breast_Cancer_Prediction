"""Weekly Airflow workflow for validating, retraining, and monitoring the model."""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# Override these in the Airflow deployment environment. The defaults suit the
# Docker layout documented in airflow/README.md.
PROJECT_ROOT = os.getenv("BREAST_CANCER_PROJECT_ROOT", "/opt/airflow/project")
PYTHON_BIN = os.getenv("BREAST_CANCER_PYTHON_BIN", "python")

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="breast_cancer_retraining",
    description="Validate data, retrain MLflow models, and create a drift report.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["breast-cancer", "mlflow", "retraining"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_BIN}" src/data/validate_data.py'
        ),
    )

    run_tests = BashOperator(
        task_id="run_tests",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_BIN}" -m unittest discover -s tests -v'
        ),
    )

    train_and_register = BashOperator(
        task_id="train_and_register_models",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_BIN}" src/tracking/mlflow_tracking.py'
        ),
    )

    create_drift_report = BashOperator(
        task_id="create_drift_report",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_BIN}" -m monitoring.model_monitor'
        ),
    )

    validate_data >> run_tests >> train_and_register >> create_drift_report
