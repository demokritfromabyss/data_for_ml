from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import re

import pandas as pd
import requests
from requests import RequestException
from bs4 import BeautifulSoup

from .base_agent import BaseAgent

try:
    from datasets import load_dataset as hf_load_dataset
except Exception:  # pragma: no cover
    hf_load_dataset = None


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataCollectionAgent/1.0; +https://example.com/bot)"
}

AG_NEWS_LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}


@dataclass
class SourceSpec:
    type: str
    name: str | None = None
    source: str | None = None
    split: str | None = None
    sample_n: int | None = None
    url: str | None = None
    selector: str | None = None
    assigned_label: str | None = None
    endpoint: str | None = None
    params: dict[str, Any] | None = None
    text_field: str | None = None
    label_field: str | None = None
    records_path: str | None = None


class DataCollectionAgent(BaseAgent):
    STANDARD_COLUMNS = ["text", "label", "source", "collected_at"]

    def __init__(self, config: str | dict | None = None, timeout: int = 20):
        super().__init__(config)
        self.timeout = timeout

    def scrape(self, url: str, selector: str, assigned_label: str | None = None, sample_n: int | None = None) -> pd.DataFrame:
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
            response.raise_for_status()
        except RequestException as exc:
            print(f"[WARN] Skipping scrape source {url}: {exc}")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        parser = "xml" if url.endswith(".xml") or "<rss" in response.text[:500].lower() else "lxml"
        soup = BeautifulSoup(response.text, parser)
        elements = soup.select(selector)

        rows = []
        collected_at = datetime.now(timezone.utc).isoformat()
        seen = set()
        for element in elements:
            text = self._clean_text(element.get_text(" ", strip=True))
            if not text or len(text) < 12:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            rows.append({
                "text": text,
                "label": assigned_label or "unknown",
                "source": f"scrape:{url}",
                "collected_at": collected_at,
            })
            if sample_n and len(rows) >= sample_n:
                break

        return pd.DataFrame(rows, columns=self.STANDARD_COLUMNS)

    def fetch_api(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        text_field: str = "text",
        label_field: str | None = "label",
        records_path: str | None = None,
        source_name: str | None = None,
    ) -> pd.DataFrame:
        response = requests.get(endpoint, params=params or {}, headers=DEFAULT_HEADERS, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        records = payload
        if records_path:
            for part in records_path.split('.'):
                records = records[part]

        if not isinstance(records, list):
            raise ValueError("API payload after records_path must be a list of records")

        collected_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for rec in records:
            text = rec.get(text_field)
            if not text:
                continue
            rows.append({
                "text": self._clean_text(str(text)),
                "label": rec.get(label_field, "unknown") if label_field else "unknown",
                "source": source_name or endpoint,
                "collected_at": collected_at,
            })

        return pd.DataFrame(rows, columns=self.STANDARD_COLUMNS)

    def load_dataset(self, name: str, source: str = "hf", split: str = "train", sample_n: int | None = None) -> pd.DataFrame:
        source = source.lower()
        if source == "hf":
            if hf_load_dataset is None:
                raise ImportError("datasets package is not installed. Run: pip install datasets")
            dataset = hf_load_dataset(name, split=split)
            if sample_n:
                dataset = dataset.select(range(min(sample_n, len(dataset))))
            records = dataset.to_pandas()
            return self._normalize_open_dataset(records, name)
        raise NotImplementedError(f"Unsupported dataset source: {source}")

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        valid = [df[self.STANDARD_COLUMNS].copy() for df in sources if df is not None and not df.empty]
        if not valid:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)
        merged = pd.concat(valid, ignore_index=True)
        merged["text"] = merged["text"].fillna("").astype(str).map(self._clean_text)
        merged["label"] = merged["label"].fillna("unknown").astype(str).str.strip().str.lower()
        merged["source"] = merged["source"].fillna("unknown").astype(str)
        merged["collected_at"] = merged["collected_at"].fillna(datetime.now(timezone.utc).isoformat())
        merged = merged[merged["text"].str.len() > 0].reset_index(drop=True)
        return merged

    def run(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        frames = []
        for raw_source in sources:
            spec = SourceSpec(**raw_source)
            if spec.type == "hf_dataset":
                frames.append(self.load_dataset(spec.name, source=spec.source or "hf", split=spec.split or "train", sample_n=spec.sample_n))
            elif spec.type == "scrape":
                frames.append(self.scrape(spec.url, spec.selector, assigned_label=spec.assigned_label, sample_n=spec.sample_n))
            elif spec.type == "api":
                frames.append(self.fetch_api(
                    endpoint=spec.endpoint,
                    params=spec.params,
                    text_field=spec.text_field or "text",
                    label_field=spec.label_field,
                    records_path=spec.records_path,
                    source_name=spec.name or spec.endpoint,
                ))
            else:
                raise ValueError(f"Unsupported source type: {spec.type}")
        return self.merge(frames)

    def _normalize_open_dataset(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        name_lower = name.lower()
        collected_at = datetime.now(timezone.utc).isoformat()

        if name_lower == "ag_news":
            out = pd.DataFrame({
                "text": df["text"].astype(str).map(self._clean_text),
                "label": df["label"].map(AG_NEWS_LABELS).fillna("unknown"),
                "source": "hf:ag_news",
                "collected_at": collected_at,
            })
            return out[self.STANDARD_COLUMNS]

        text_col = self._find_first_existing(df, ["text", "sentence", "content", "title"])
        label_col = self._find_first_existing(df, ["label", "labels", "category", "topic"])
        if text_col is None:
            raise ValueError(f"Could not infer text column for dataset '{name}'")

        out = pd.DataFrame({
            "text": df[text_col].astype(str).map(self._clean_text),
            "label": df[label_col].astype(str).str.lower() if label_col else "unknown",
            "source": f"hf:{name}",
            "collected_at": collected_at,
        })
        return out[self.STANDARD_COLUMNS]

    @staticmethod
    def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text
