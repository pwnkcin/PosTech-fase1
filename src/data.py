from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "PCOS_infertility.csv"
DROP_COLS = ["Sl. No", "Patient File No."]
TARGET = "PCOS (Y/N)"

RENAME_MAP = {
    "I   beta-HCG(mIU/mL)": "beta_HCG_I",
    "II    beta-HCG(mIU/mL)": "beta_HCG_II",
    "AMH(ng/mL)": "AMH",
}


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the PCOS biomarker dataset, matching analysis.ipynb's cleaning exactly."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=DROP_COLS)

    feature_cols = [c for c in df.columns if c != TARGET]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols].rename(columns=RENAME_MAP)
    y = df[TARGET].astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    """Median imputation + standardization, matching analysis.ipynb's base_steps."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
