# AL / Model Report

## Available artifact
The uploaded files include a trained model (`final_model.joblib`) but do not include active learning history per iteration.
So this report documents the final model quality and leaves the learning-curve section to be filled after running the AL notebook.

## Final model evaluation on uploaded labeled data
- Accuracy: 0.9167
- Macro F1: 0.9149

## Confusion matrix
|          |   world |   sports |   business |   sci_tech |
|:---------|--------:|---------:|-----------:|-----------:|
| world    |     469 |       20 |         15 |         26 |
| sports   |       8 |      312 |          0 |         17 |
| business |      10 |        0 |        375 |         69 |
| sci_tech |       7 |        2 |          2 |        780 |

## Suggested wording for the report
The final TF-IDF + Logistic Regression baseline performs strongly on the cleaned and labeled dataset.
Without iteration history it is not possible to quantify the gain of entropy vs random sampling in this report version.
To finalize the AL section, run the AL experiment notebook and add the history plot.

## Files
- Figure: `figures/final_model_confusion_matrix.png`
