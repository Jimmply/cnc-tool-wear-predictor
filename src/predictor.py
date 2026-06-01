"""
CNC Tool Wear State Classifier and RUL Regressor
=================================================
XGBoost-based models for:
  - wear_state classification  (Fresh / Worn / Critical)
  - remaining useful life (RUL) regression  (cycles to end-of-life)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from data_generator import FEATURE_COLS

logger = logging.getLogger(__name__)

WEAR_ORDER = ["Fresh", "Worn", "Critical"]


@dataclass
class TrainResults:
    classification_report: str
    rul_mae: float
    feature_importances: pd.Series
    label_encoder: LabelEncoder


class ToolWearPredictor:
    """
    Wraps an XGBClassifier (wear state) and XGBRegressor (RUL).

    Usage::

        predictor = ToolWearPredictor()
        results = predictor.fit(df)
        predictions = predictor.predict(df)
    """

    def __init__(self) -> None:
        self._clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self._reg = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._le = LabelEncoder()
        self._trained = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> TrainResults:
        """Train on the full dataset and return evaluation metrics."""
        X = df[FEATURE_COLS].values
        y_state = self._le.fit_transform(df["wear_state"])
        y_rul = df["rul_cycles"].values.astype(float)

        X_tr, X_te, ys_tr, ys_te, yr_tr, yr_te = train_test_split(
            X, y_state, y_rul, test_size=0.20, random_state=42, stratify=y_state
        )

        self._clf.fit(X_tr, ys_tr)
        self._reg.fit(X_tr, yr_tr)
        self._trained = True

        report = classification_report(
            ys_te,
            self._clf.predict(X_te),
            target_names=self._le.classes_,
        )
        rul_mae = mean_absolute_error(yr_te, self._reg.predict(X_te))
        importances = (
            pd.Series(self._clf.feature_importances_, index=FEATURE_COLS)
            .sort_values(ascending=True)
        )

        logger.info("Trained. RUL MAE=%.1f cycles", rul_mae)
        return TrainResults(report, rul_mae, importances, self._le)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions appended to tool_id / cut_number columns."""
        if not self._trained:
            raise RuntimeError("Call fit() before predict().")

        X = df[FEATURE_COLS].values
        state_enc = self._clf.predict(X)
        proba = self._clf.predict_proba(X)
        rul = np.clip(self._reg.predict(X), 0, None)

        out = df[["tool_id", "cut_number"] + FEATURE_COLS].copy()
        out["predicted_state"] = self._le.inverse_transform(state_enc)
        out["predicted_rul"] = rul.round(0).astype(int)
        for i, cls in enumerate(self._le.classes_):
            out[f"prob_{cls.lower()}"] = proba[:, i].round(4)
        return out

    def latest_per_tool(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the most recent prediction row for each tool (fleet view)."""
        preds = self.predict(df)
        return (
            preds.sort_values("cut_number")
            .groupby("tool_id", as_index=False)
            .last()
        )

    def save(self, path: str | Path) -> None:
        """Persist the trained models and label encoder to disk."""
        if not self._trained:
            raise RuntimeError("Nothing to save — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "reg": self._reg, "le": self._le}, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ToolWearPredictor":
        """Load a previously saved predictor from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._clf = data["clf"]
        obj._reg = data["reg"]
        obj._le = data["le"]
        obj._trained = True
        return obj
