# Quality Report

## Dataset snapshot
- Rows in collected dataset: 2126
- Columns: text, label, source, collected_at
- Missing values: {'text': 0, 'label': 0, 'source': 0, 'collected_at': 0, 'text_len': 0}
- Duplicate texts: 14

## Class distribution
| label    |   count |
|:---------|--------:|
| world    |     533 |
| sports   |     338 |
| business |     458 |
| sci_tech |     797 |

## Source distribution
| source                                                             |   count |
|:-------------------------------------------------------------------|--------:|
| hf:ag_news                                                         |    2000 |
| scrape:https://rss.nytimes.com/services/xml/rss/nyt/World.xml      |      56 |
| scrape:https://rss.nytimes.com/services/xml/rss/nyt/Business.xml   |      50 |
| scrape:https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml |      20 |

## Text length
- Mean: 237.55
- Median: 236.00
- Min: 14
- Max: 959

## Findings
1. Missing values are absent in the uploaded collected dataset.
2. There are 14 duplicate texts, so deduplication is justified.
3. The dataset is imbalanced: `sci_tech` dominates, while `sports` is the smallest class.
4. Text length varies widely, so very short or very long records can be treated as outliers depending on the selected strategy.

## Recommended cleaning strategy
```python
strategy = {
    "missing": "drop",
    "duplicates": "drop",
    "outliers": "clip_iqr"
}
```

## Files
- Figure: `figures/class_distribution.png`
- Figure: `figures/text_length_distribution.png`
