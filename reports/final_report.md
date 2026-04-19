# Final Project Report

## 1. Task and dataset
This project builds a multi-agent pipeline for multiclass news topic classification.
The classes are `world`, `sports`, `business`, and `sci_tech`.

The collected dataset contains 2126 rows from two source types:
- HuggingFace dataset (`ag_news`)
- scraped RSS news headlines

Class distribution:
| label    |   count |
|:---------|--------:|
| world    |     533 |
| sports   |     338 |
| business |     458 |
| sci_tech |     797 |

## 2. What each agent does
### DataCollectionAgent
Collects rows from HuggingFace and RSS sources, then merges them into a unified schema:
`text`, `label`, `source`, `collected_at`.

### DataQualityAgent
Detects missing values, duplicates, class imbalance, and text-length outliers.
For the uploaded collected dataset:
- missing values: {'text': 0, 'label': 0, 'source': 0, 'collected_at': 0, 'text_len': 0}
- duplicate texts: 14

### AnnotationAgent
Auto-labels texts, assigns confidence, exports low-confidence rows for review, and produces an annotation specification.

Measured against the available labels:
- auto-label accuracy: 0.4858
- auto-label macro F1: 0.4849
- kappa: 0.2782

### ActiveLearningAgent / final model stage
A TF-IDF + Logistic Regression model was trained and saved as `final_model.joblib`.

Evaluation on the uploaded labeled data:
- final model accuracy: 0.9167
- final model macro F1: 0.9149

## 3. Human-in-the-loop
The explicit human review point is `review_queue.csv`.
The queue contains 513 low-confidence examples.
These examples should be manually corrected before final retraining.

## 4. Metrics by stage
- collected rows: 2126
- labeled rows after cleaning/annotation: 2112
- auto-label mean confidence: 0.8421
- final model accuracy: 0.9167
- final model macro F1: 0.9149

## 5. Retrospective
What worked:
- a unified schema across sources
- a clear low-confidence review queue
- a strong final baseline model

What did not fully work:
- class imbalance remains noticeable
- auto-label quality is moderate rather than high
- AL iteration history is not included in the uploaded artifacts

What to improve next:
- rebalance minority classes
- replace heuristic labeling with stronger zero-shot or supervised labeling
- save AL history at every iteration and add a learning-curve figure
