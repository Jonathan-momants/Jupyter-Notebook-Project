# Momants sentimentclassificatie

Open `00_momants_sentiment.ipynb` en voer de cellen van boven naar beneden uit.
De eerste modeldownload kan enkele minuten duren.

## Belangrijk

- Gebruik uitsluitend nep- of testgesprekken.
- De notebook leest alleen `created_at`, `text`, `from_agent`, `message_type`,
  `conversation_id` en `agent_id`.
- `raw_json`, `chat_sender`, `media`, `media_url` en `file` worden niet
  ingelezen.
- Het prototype doet alleen sentimentclassificatie.
- De databron is nu `data/voorbeeld_gesprekken.csv`.
- Er wordt niets naar de Momants-database geschreven.

De Python-afhankelijkheden staan in `pyproject.toml`.