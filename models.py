"""Aqar Hub v22.0 — CatBoost Ensemble with Interval Prediction"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import List, Tuple, Union


def _catboost_params(seed=42):
    return dict(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bagging_temperature=0.3,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


class CatBoostEnsemble:
    """
    5-Fold CatBoost ensemble.
    Provides point prediction + uncertainty interval from fold disagreement.
    """

    def __init__(self, models: List[CatBoostRegressor], col_cat: List[str], columns: List[str]):
        self.models = models
        self.col_cat = list(col_cat)
        self.columns = list(columns)

    def _align(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Align input to training columns with strict validation."""
        if isinstance(X, pd.DataFrame):
            missing = set(self.columns) - set(X.columns)
            extra = set(X.columns) - set(self.columns)
            if missing:
                raise ValueError(
                    f"Input missing {len(missing)} required columns: {sorted(missing)[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
            if extra and len(extra) > 3:
                # Warn but don't fail for a few extra columns (robustness)
                pass
            return X[self.columns].copy()
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.columns):
                raise ValueError(
                    f"Array shape mismatch: expected {len(self.columns)} features, "
                    f"got {X.shape[1]}"
                )
            return pd.DataFrame(X, columns=self.columns)
        else:
            raise TypeError(f"X must be DataFrame or ndarray, got {type(X).__name__}")

    def _validate_predictions(self, preds: np.ndarray, context: str = "") -> None:
        """Defensive check for NaN/Inf in predictions."""
        if np.any(np.isnan(preds)):
            raise RuntimeError(f"{context} predictions contain NaN values")
        if np.any(np.isinf(preds)):
            raise RuntimeError(f"{context} predictions contain Inf values")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Point prediction (log-space)."""
        Xa = self._align(X)
        preds = np.array([m.predict(Xa[self.col_cat]) for m in self.models])
        self._validate_predictions(preds, "Log-space")
        return preds.mean(axis=0).astype(np.float32)

    def predict_interval_log(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lo, med, hi) in log-space.
        FOR DIAGNOSTICS ONLY — do NOT use for confidence calculation.
        """
        Xa = self._align(X)
        preds = np.array([m.predict(Xa[self.col_cat]) for m in self.models])
        self._validate_predictions(preds, "Log-space interval")
        
        med = preds.mean(axis=0)
        std = preds.std(axis=0)
        
        # Defensive: std should not be negative (numerical safety)
        std = np.maximum(std, 0.0)
        
        lo = np.maximum(med - 1.5 * std, med * 0.5)
        hi = med + 1.5 * std
        
        # Ensure monotonicity: lo <= med <= hi
        lo = np.minimum(lo, med)
        hi = np.maximum(hi, med)
        
        return lo.astype(np.float32), med.astype(np.float32), hi.astype(np.float32)

    def predict_price_interval(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lo, med, hi) in EGP (price-space).
        
        CRITICAL: width = (hi - lo) / med MUST be computed here (price-space).
        expm1() is non-linear — log-space width severely underestimates risk.
        """
        lo_log, med_log, hi_log = self.predict_interval_log(X)
        
        lo_price = np.expm1(lo_log)
        med_price = np.expm1(med_log)
        hi_price = np.expm1(hi_log)
        
        # Safety floor: prices can't be negative (defensive for production)
        lo_price = np.maximum(lo_price, 0.0)
        med_price = np.maximum(med_price, 0.0)
        hi_price = np.maximum(hi_price, 0.0)
        
        # Ensure monotonicity after expm1 (numerical safety)
        lo_price = np.minimum(lo_price, med_price)
        hi_price = np.maximum(hi_price, med_price)
        
        self._validate_predictions(med_price, "Price-space")
        
        return lo_price.astype(np.float32), med_price.astype(np.float32), hi_price.astype(np.float32)

    def predict_price(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Point prediction in EGP."""
        return np.expm1(self.predict(X))