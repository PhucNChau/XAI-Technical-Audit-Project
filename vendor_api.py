"""Proprietary Vendor API wrapper for Project 2 (Public Health Assistance Scoring Oracle).

Provides black-box access to the pre-trained commercial scoring engine:
    probs = vendor_api.predict_proba(X)
    preds = vendor_api.predict(X)
"""

from typing import Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATHS = [
    Path(__file__).resolve().parent / "data" / "vendor_model.joblib",
    Path(__file__).resolve().parent.parent / "data" / "vendor_model.joblib",
]


FEATURE_COLS = [
    "AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIG", "MIL",
    "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "PINCP", "ESR", "ST", "FER"
]


class _VendorSVMOracle:
    def __init__(self):
        self._pipeline = None

    def _prepare_input(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in FEATURE_COLS):
                return X[FEATURE_COLS]
            cols_to_drop = [c for c in ["PUBCOV", "STATE", "target", "label"] if c in X.columns]
            if cols_to_drop:
                return X.drop(columns=cols_to_drop)
        return X

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return

        for p in MODEL_PATHS:
            if p.exists():
                try:
                    self._pipeline = joblib.load(p)
                    return
                except Exception:
                    pass

        # Fallback: Deterministic scoring oracle with statutory Medicare policy cliff at AGEP = 65
        class _FallbackCoverageModel:
            def predict_proba(self, X):
                if isinstance(X, pd.DataFrame):
                    age = X["AGEP"].values if "AGEP" in X.columns else X.iloc[:, 0].values
                    pincp = X["PINCP"].values if "PINCP" in X.columns else np.full(len(X), 25000.0)
                    dis = X["DIS"].values if "DIS" in X.columns else np.ones(len(X))
                else:
                    arr = np.asarray(X)
                    age = arr[:, 0]
                    pincp = arr[:, 14] if arr.shape[1] > 14 else np.full(len(arr), 25000.0)
                    dis = arr[:, 4] if arr.shape[1] > 4 else np.ones(len(arr))

                # Statutory step function at age 65: Medicare eligibility triggers dramatic jump in coverage
                cliff_jump = np.where(age >= 65, 2.8, -0.6)
                poverty_effect = -0.00004 * np.clip(pincp, 0, 100000)
                disability_effect = np.where(dis == 1, 1.2, 0.0)
                z = cliff_jump + poverty_effect + disability_effect
                p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        self._pipeline = _FallbackCoverageModel()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        X_prep = self._prepare_input(X)
        return self._pipeline.predict_proba(X_prep)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        X_prep = self._prepare_input(X)
        return self._pipeline.predict(X_prep)


_oracle = _VendorSVMOracle()


def predict(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    return _oracle.predict(X)


def predict_proba(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    return _oracle.predict_proba(X)

