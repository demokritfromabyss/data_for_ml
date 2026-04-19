from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .base_agent import BaseAgent

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


DEFAULT_LABELS = ["world", "sports", "business", "sci_tech"]
KEYWORDS = {
    "sports": ["match", "league", "goal", "coach", "player", "tournament", "season", "championship", "nba", "nfl", "soccer", "tennis"],
    "business": ["stocks", "market", "bank", "shares", "earnings", "investor", "trade", "economy", "company", "profit", "inflation", "tariff"],
    "sci_tech": ["ai", "technology", "software", "chip", "research", "scientists", "robot", "space", "internet", "startup", "app", "device"],
    "world": ["president", "minister", "war", "election", "government", "country", "diplomat", "military", "border", "summit", "leaders", "parliament"],
}


class AnnotationAgent(BaseAgent):
    def __init__(self, modality: str = "text", method: str = "keyword_heuristic", config: str | dict | None = None):
        super().__init__(config)
        self.modality = modality
        self.method = method
        self.labels = self.config.get("labels_list", DEFAULT_LABELS)
        self._classifier = None

    def auto_label(self, df: pd.DataFrame, modality: str | None = None) -> pd.DataFrame:
        modality = modality or self.modality
        if modality != "text":
            raise NotImplementedError("This project implementation supports only text modality.")

        work = df.copy()
        preds = []
        for text in work["text"].fillna("").astype(str):
            if self.method == "zero_shot":
                label, conf = self._zero_shot_predict(text)
            else:
                label, conf = self._keyword_predict(text)
            preds.append((label, conf))

        work["auto_label"] = [p[0] for p in preds]
        work["confidence"] = [p[1] for p in preds]
        if "label" not in work.columns:
            work["label"] = work["auto_label"]
        return work

    def generate_spec(self, df: pd.DataFrame, task: str, output_path: str = "annotation_spec.md") -> str:
        examples = {}
        label_col = "label" if "label" in df.columns else "auto_label"
        for label in self.labels:
            subset = df[df[label_col] == label]["text"].head(3).tolist()
            examples[label] = subset

        md = [
            f"# Annotation Specification: {task}",
            "",
            "## Goal",
            "Assign each news text to exactly one topic class.",
            "",
            "## Classes",
            "- **world** — politics, diplomacy, conflicts, international affairs, government actions.",
            "- **sports** — games, athletes, tournaments, leagues, scores, transfers.",
            "- **business** — companies, markets, finance, economic policy, earnings, trade.",
            "- **sci_tech** — technology products, AI, science discoveries, software, hardware, startups.",
            "",
            "## Borderline cases",
            "- If article is about a sports club's finances, prefer **sports** if the sporting event/team is central.",
            "- If article is about a tech company's earnings, prefer **business** if the focus is markets/revenue.",
            "- If article is about science policy by a government, prefer **world** if politics dominates.",
            "- If article mentions technology casually but discusses markets, prefer **business**.",
            "",
            "## Examples",
        ]
        for label, sample_texts in examples.items():
            md.append(f"### {label}")
            for i, text in enumerate(sample_texts, 1):
                md.append(f"{i}. {text}")
            if not sample_texts:
                md.append("1. Add examples after collection stage.")
            md.append("")

        Path(output_path).write_text("\n".join(md), encoding="utf-8")
        return output_path

    def check_quality(self, df_labeled: pd.DataFrame, reference_col: str | None = "label") -> dict[str, Any]:
        label_col = "auto_label" if "auto_label" in df_labeled.columns else "label"
        result = {
            "label_dist": df_labeled[label_col].value_counts(normalize=True).round(4).to_dict(),
            "confidence_mean": float(df_labeled["confidence"].mean()) if "confidence" in df_labeled else None,
            "n_records": int(len(df_labeled)),
        }
        if reference_col and reference_col in df_labeled.columns and label_col in df_labeled.columns:
            try:
                result["kappa"] = float(cohen_kappa_score(df_labeled[reference_col], df_labeled[label_col]))
                result["agreement"] = float((df_labeled[reference_col] == df_labeled[label_col]).mean())
            except Exception:
                result["kappa"] = None
                result["agreement"] = None
        return result

    def export_to_labelstudio(self, df: pd.DataFrame, output_path: str = "labelstudio_import.json") -> str:
        label_col = "auto_label" if "auto_label" in df.columns else "label"
        records = []
        for idx, row in df.iterrows():
            item = {
                "id": int(idx),
                "data": {"text": row["text"]},
                "predictions": [
                    {
                        "model_version": self.method,
                        "score": float(row.get("confidence", 1.0)),
                        "result": [
                            {
                                "id": f"pred-{idx}",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [row[label_col]]},
                            }
                        ],
                    }
                ],
            }
            records.append(item)
        Path(output_path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def export_low_confidence(self, df: pd.DataFrame, threshold: float = 0.65, output_path: str = "review_queue.csv") -> str:
        if "confidence" not in df.columns:
            raise ValueError("DataFrame must contain 'confidence' column before exporting review queue.")
        review_df = df[df["confidence"] < threshold].copy()
        review_df.to_csv(output_path, index=False)
        return output_path

    def _keyword_predict(self, text: str) -> tuple[str, float]:
        normalized = re.sub(r"\s+", " ", text.lower())
        scores = {}
        for label, words in KEYWORDS.items():
            score = sum(1 for word in words if word in normalized)
            scores[label] = score
        best_label = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = 0.4 if total == 0 else min(0.99, 0.5 + scores[best_label] / max(total, 1))
        if total == 0:
            best_label = "world"
        return best_label, float(confidence)

    def _zero_shot_predict(self, text: str) -> tuple[str, float]:
        if pipeline is None:
            return self._keyword_predict(text)
        if self._classifier is None:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.config.get("zero_shot_model", "facebook/bart-large-mnli"),
            )
        result = self._classifier(text, candidate_labels=self.labels, multi_label=False)
        return result["labels"][0], float(result["scores"][0])
