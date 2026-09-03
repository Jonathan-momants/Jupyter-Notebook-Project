# Momants classification notebooks

Synthetic test data is stored in `data/tests/`.

## Notebook

Open the relevant English-named notebook and run its cells from top to bottom:

- `momants_sentiment.ipynb`
- `momants_answer_check.ipynb`
- `momants_intent.ipynb`
- `momants_topic.ipynb`

## Commandoregel

First check whether an export is recognized:

```bash
python momants_sentiment.py path/to/export.csv --check-only
```

Then run sentiment classification:

```bash
python momants_sentiment.py path/to/export.csv --output-directory results
```

Results from all classifiers are written beneath `results/`. Historical files
with translated names are retained in `results/archive/`.

The processors support Momants CSV files with headers and the headerless
22-field format. Privacy fields such as `raw_json` and `chat_sender` are never
selected.