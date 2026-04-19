from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .base_agent import BaseAgent


class ActiveLearningAgent(BaseAgent):
    def __init__(self, model: str = "logreg", random_state: int = 42):
        super().__init__(None)
        self.model_name = model
        self.random_state = random_state
        self.pipeline = self._make_model()

    def _make_model(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=self.random_state)),
        ])

    def fit(self, labeled_df: pd.DataFrame) -> Pipeline:
        self.pipeline = self._make_model()
        self.pipeline.fit(labeled_df["text"], labeled_df["label"])
        return self.pipeline

    def query(self, pool_df: pd.DataFrame, strategy: str, batch_size: int = 20) -> list[int]:
        if pool_df.empty:
            return []
        if strategy == "random":
            rng = np.random.default_rng(self.random_state)
            return rng.choice(pool_df.index.to_numpy(), size=min(batch_size, len(pool_df)), replace=False).tolist()

        probs = self.pipeline.predict_proba(pool_df["text"])
        if strategy == "entropy":
            uncertainty = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
            selected = np.argsort(-uncertainty)[:batch_size]
        elif strategy == "margin":
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            selected = np.argsort(margins)[:batch_size]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        return pool_df.iloc[selected].index.tolist()

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
        self.fit(labeled_df)
        preds = self.pipeline.predict(test_df["text"])
        return {
            "accuracy": float(accuracy_score(test_df["label"], preds)),
            "f1_macro": float(f1_score(test_df["label"], preds, average="macro")),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
        test_df: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        labeled = labeled_df.copy().reset_index(drop=True)
        pool = pool_df.copy().reset_index(drop=True)
        history = []

        for iteration in range(n_iterations + 1):
            metrics = self.evaluate(labeled, test_df)
            history.append({
                "iteration": iteration,
                "n_labeled": int(len(labeled)),
                **metrics,
            })
            if iteration == n_iterations or pool.empty:
                break
            self.fit(labeled)
            selected_idx = self.query(pool, strategy=strategy, batch_size=batch_size)
            newly_labeled = pool.loc[selected_idx].copy()
            labeled = pd.concat([labeled, newly_labeled], ignore_index=True)
            pool = pool.drop(index=selected_idx).reset_index(drop=True)
        return history

    def report(self, history: list[dict[str, Any]], output_path: str = "learning_curve.png") -> str:
        df = pd.DataFrame(history)
        plt.figure(figsize=(8, 5))
        plt.plot(df["n_labeled"], df["f1_macro"], marker="o")
        plt.xlabel("Number of labeled samples")
        plt.ylabel("Macro F1")
        plt.title("Active Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def save_model(self, output_path: str = "models/final_model.joblib") -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, output_path)
        return output_path

    def prepare_splits(
        self,
        df: pd.DataFrame,
        initial_size: int = 50,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=self.random_state,
        )
        initial_df, pool_df = train_test_split(
            train_df,
            train_size=min(initial_size, len(train_df) - 1),
            stratify=train_df["label"],
            random_state=self.random_state,
        )
        return initial_df.reset_index(drop=True), pool_df.reset_index(drop=True), test_df.reset_index(drop=True)
