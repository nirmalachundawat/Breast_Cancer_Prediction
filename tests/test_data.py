from pathlib import Path
import pandas as pd


def test_dataset_exists():
    path = Path("data/raw/breast_cancer.csv")
    assert path.exists(), "Dataset does not exist"


def test_dataset_shape():
    path = Path("data/raw/breast_cancer.csv")
    df = pd.read_csv(path)

    assert df.shape[0] == 569
    assert df.shape[1] == 32


def test_target_column_exists():
    path = Path("data/raw/breast_cancer.csv")
    df = pd.read_csv(path)

    assert "diagnosis" in df.columns