# Airflow retraining pipeline

The `breast_cancer_retraining` DAG runs weekly:

1. Validates `data/raw/breast_cancer.csv`.
2. Runs the project test suite.
3. Retrains all four models and registers new MLflow versions.
4. Generates the prediction data-drift report.

Airflow does not support native Windows installation. Run it through Docker or
WSL. The DAG expects the repository to be available at `/opt/airflow/project`
inside the Airflow environment.

Set these environment variables if your deployment uses different locations:

```text
BREAST_CANCER_PROJECT_ROOT=/opt/airflow/project
BREAST_CANCER_PYTHON_BIN=python
```

Copy or mount `airflow/dags/breast_cancer_retraining.py` into Airflow's `dags`
directory, then enable **breast_cancer_retraining** in the Airflow UI.
