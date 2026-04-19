from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .base_agent import BaseAgent


@dataclass
class QualityReport:
    missing: dict[str, int]
    duplicates: int
    outliers: dict[str, int]
    imbalance: dict[str, float]
    summary: dict[str, Any]


class DataQualityAgent(BaseAgent):
    def __init__(self, text_col: str = "text", label_col: str = "label"):
        super().__init__(None)
        self.text_col = text_col
        self.label_col = label_col

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        work = self._enrich(df)
        numeric_cols = [c for c in ["text_len", "word_count"] if c in work.columns]

        missing = work.isna().sum().to_dict()
        missing[self.text_col] = int(work[self.text_col].astype(str).str.strip().eq("").sum() + work[self.text_col].isna().sum())
        duplicates = int(work.duplicated(subset=[self.text_col]).sum())
        outliers = {col: int(self._iqr_mask(work[col]).sum()) for col in numeric_cols}
        label_dist = work[self.label_col].value_counts(normalize=True, dropna=False).round(4).to_dict()
        imbalance = {
            "label_distribution": label_dist,
            "imbalance_ratio": float(work[self.label_col].value_counts().max() / max(work[self.label_col].value_counts().min(), 1)) if not work.empty else 0.0,
        }
        return QualityReport(
            missing=missing,
            duplicates=duplicates,
            outliers=outliers,
            imbalance=imbalance,
            summary={
                "rows": int(len(work)),
                "empty_text_rows": int(work[self.text_col].astype(str).str.strip().eq("").sum()),
                "avg_text_len": float(work["text_len"].mean()) if "text_len" in work else 0.0,
            },
        ).__dict__

    def fix(self, df: pd.DataFrame, strategy: dict[str, str]) -> pd.DataFrame:
        work = self._enrich(df.copy())

        missing_strategy = strategy.get("missing", "drop")
        if missing_strategy == "drop":
            work = work[~work[self.text_col].isna() & work[self.text_col].astype(str).str.strip().ne("")].copy()
        elif missing_strategy == "fill_unknown":
            work[self.text_col] = work[self.text_col].fillna("unknown_text")
            work.loc[work[self.text_col].astype(str).str.strip().eq(""), self.text_col] = "unknown_text"
        else:
            raise ValueError(f"Unsupported missing strategy: {missing_strategy}")

        dup_strategy = strategy.get("duplicates", "drop")
        if dup_strategy in {"drop", "keep_first"}:
            keep = "first"
            work = work.drop_duplicates(subset=[self.text_col], keep=keep)
        elif dup_strategy == "keep_last":
            work = work.drop_duplicates(subset=[self.text_col], keep="last")
        else:
            raise ValueError(f"Unsupported duplicates strategy: {dup_strategy}")

        outlier_strategy = strategy.get("outliers", "clip_iqr")
        for col in [c for c in ["text_len", "word_count"] if c in work.columns]:
            if outlier_strategy == "clip_iqr":
                q1, q3 = work[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                work[col] = work[col].clip(lower=lower, upper=upper)
            elif outlier_strategy == "remove_iqr":
                mask = ~self._iqr_mask(work[col])
                work = work[mask].copy()
            elif outlier_strategy == "none":
                pass
            else:
                raise ValueError(f"Unsupported outlier strategy: {outlier_strategy}")

        label_strategy = strategy.get("labels", "normalize")
        if label_strategy == "normalize":
            work[self.label_col] = work[self.label_col].astype(str).str.strip().str.lower()
        elif label_strategy != "none":
            raise ValueError(f"Unsupported labels strategy: {label_strategy}")

        return work.drop(columns=[c for c in ["text_len", "word_count"] if c in work.columns], errors="ignore").reset_index(drop=True)

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        before = self._enrich(df_before)
        after = self._enrich(df_after)

        metrics = [
            ("rows", len(before), len(after)),
            ("missing_text", int(before[self.text_col].isna().sum() + before[self.text_col].astype(str).str.strip().eq("").sum()), int(after[self.text_col].isna().sum() + after[self.text_col].astype(str).str.strip().eq("").sum())),
            ("duplicate_text", int(before.duplicated(subset=[self.text_col]).sum()), int(after.duplicated(subset=[self.text_col]).sum())),
            ("text_len_outliers", int(self._iqr_mask(before["text_len"]).sum()), int(self._iqr_mask(after["text_len"]).sum())),
            ("avg_text_len", float(before["text_len"].mean()), float(after["text_len"].mean())),
            ("n_classes", int(before[self.label_col].nunique(dropna=False)), int(after[self.label_col].nunique(dropna=False))),
        ]
        return pd.DataFrame(metrics, columns=["metric", "before", "after"])

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.text_col not in work.columns:
            raise ValueError(f"Expected text column '{self.text_col}'")
        if self.label_col not in work.columns:
            raise ValueError(f"Expected label column '{self.label_col}'")
        work["text_len"] = work[self.text_col].fillna("").astype(str).str.len()
        work["word_count"] = work[self.text_col].fillna("").astype(str).str.split().str.len()
        return work

    @staticmethod
    def _iqr_mask(series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)
