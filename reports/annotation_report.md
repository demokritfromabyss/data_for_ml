# Annotation Report

## Annotation setup
- Task: news topic classification
- Classes: world, sports, business, sci_tech
- Auto-labeled rows: 2112
- Review queue rows: 513

## Auto-label quality against ground truth
- Accuracy: 0.4858
- Macro F1: 0.4849
- Cohen's kappa: 0.2782
- Mean confidence: 0.8421

## Label distributions
### Ground truth
| label    |   count |
|:---------|--------:|
| world    |     530 |
| sports   |     337 |
| business |     454 |
| sci_tech |     791 |

### Auto labels
| auto_label   |   count |
|:-------------|--------:|
| world        |     701 |
| sports       |     183 |
| business     |     361 |
| sci_tech     |     867 |

## Human-in-the-loop
All rows with low confidence were exported to `review_queue.csv`.
In the uploaded file:
- review queue size: 513
- confidence min/max: 0.40 / 0.40
- mean confidence in queue: 0.40

## Interpretation
The auto-labeler is usable as a first-pass annotator, but not accurate enough to replace human validation.
The weakest area is confusion between `world`, `business`, and `sci_tech`, which is expected for short news texts.

## Files
- Figure: `figures/auto_label_confusion_matrix.png`
- Spec basis: `annotation_spec.md`
