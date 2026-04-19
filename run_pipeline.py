from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from agents import (
    ActiveLearningAgent,
    AnnotationAgent,
    DataCollectionAgent,
    DataQualityAgent,
)

ROOT = Path(__file__).resolve().parent


def save_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    collector = DataCollectionAgent(config=str(ROOT / "config.yaml"))
    raw_df = collector.run(config["sources"])
    raw_path = ROOT / "data" / "raw" / "collected.csv"
    raw_df.to_csv(raw_path, index=False)

    quality_agent = DataQualityAgent(
        text_col=config.get("text_column", "text"),
        label_col=config.get("label_column", "label"),
    )
    quality_report = quality_agent.detect_issues(raw_df)
    clean_df = quality_agent.fix(raw_df, strategy=config["quality_strategy"])
    clean_path = ROOT / "data" / "raw" / "clean.csv"
    clean_df.to_csv(clean_path, index=False)
    comparison = quality_agent.compare(raw_df, clean_df)
    comparison.to_csv(ROOT / "reports" / "quality_comparison.csv", index=False)
    save_markdown(
        ROOT / "reports" / "quality_report.md",
        "# Quality Report\n\n```json\n"
        + json.dumps(quality_report, indent=2, ensure_ascii=False)
        + "\n```\n",
    )

    annotation_cfg = config.get("annotation", {})
    annotation_agent = AnnotationAgent(
        modality=annotation_cfg.get("modality", "text"),
        method=annotation_cfg.get("method", "keyword_heuristic"),
        config=str(ROOT / "config.yaml"),
    )
    labeled_df = annotation_agent.auto_label(clean_df)
    labeled_path = ROOT / "data" / "labeled" / "auto_labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)

    spec_path = annotation_agent.generate_spec(
        labeled_df,
        task="news_topic_classification",
        output_path=str(ROOT / "reports" / "annotation_spec.md"),
    )
    metrics = annotation_agent.check_quality(labeled_df, reference_col="label")
    annotation_agent.export_to_labelstudio(labeled_df, output_path=str(ROOT / "labelstudio_import.json"))
    review_path = annotation_agent.export_low_confidence(
        labeled_df,
        threshold=float(annotation_cfg.get("low_conf_threshold", 0.65)),
        output_path=str(ROOT / "review_queue.csv"),
    )
    save_markdown(
        ROOT / "reports" / "annotation_report.md",
        "# Annotation Report\n\n```json\n"
        + json.dumps(metrics, indent=2, ensure_ascii=False)
        + "\n```\n",
    )

    al_cfg = config.get("active_learning", {})
    al_agent = ActiveLearningAgent(model=al_cfg.get("model", "logreg"))
    initial_df, pool_df, test_df = al_agent.prepare_splits(
        clean_df,
        initial_size=int(al_cfg.get("initial_size", 80)),
    )
    entropy_history = al_agent.run_cycle(
        labeled_df=initial_df,
        pool_df=pool_df,
        strategy=al_cfg.get("strategy", "entropy"),
        n_iterations=int(al_cfg.get("n_iterations", 5)),
        batch_size=int(al_cfg.get("batch_size", 25)),
        test_df=test_df,
    )
    random_history = al_agent.run_cycle(
        labeled_df=initial_df,
        pool_df=pool_df,
        strategy="random",
        n_iterations=int(al_cfg.get("n_iterations", 5)),
        batch_size=int(al_cfg.get("batch_size", 25)),
        test_df=test_df,
    )
    al_agent.fit(pd.concat([initial_df, pool_df], ignore_index=True))
    model_path = al_agent.save_model(str(ROOT / "models" / "final_model.joblib"))
    al_agent.report(entropy_history, output_path=str(ROOT / "reports" / "learning_curve_entropy.png"))
    pd.DataFrame(entropy_history).to_csv(ROOT / "reports" / "al_history_entropy.csv", index=False)
    pd.DataFrame(random_history).to_csv(ROOT / "reports" / "al_history_random.csv", index=False)
    save_markdown(
        ROOT / "reports" / "al_report.md",
        "# Active Learning Report\n\n## Entropy\n\n```json\n"
        + json.dumps(entropy_history, indent=2, ensure_ascii=False)
        + "\n```\n\n## Random\n\n```json\n"
        + json.dumps(random_history, indent=2, ensure_ascii=False)
        + "\n```\n",
    )

    print("Pipeline finished successfully.")
    print(f"Raw data: {raw_path}")
    print(f"Clean data: {clean_path}")
    print(f"Labeled data: {labeled_path}")
    print(f"Annotation spec: {spec_path}")
    print(f"Review queue: {review_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
