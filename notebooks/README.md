# Momants CSV-sentimentprocessor

Er wordt geen voorbeeld- of nep-CSV meegeleverd.

## Notebook

Open `momants_sentiment.ipynb`, vul `CSV_PAD` in en voer de cellen van boven
naar beneden uit.

## Commandoregel

Controleer eerst of een export goed wordt herkend:

```bash
python momants_sentiment.py pad/naar/export.csv --alleen-controleren
```

Voer daarna de sentimentclassificatie uit:

```bash
python momants_sentiment.py pad/naar/export.csv --uitvoermap resultaten
```

Dit schrijft:

- `resultaten/sentiment_per_bericht.csv`
- `resultaten/sentiment_per_gesprek.csv`

De berichttekst staat standaard niet in de uitvoer. Voeg alleen wanneer nodig
`--tekst-opnemen` toe.

De processor ondersteunt Momants-CSV's met kolomkoppen en het headerloze
22-veldenformaat uit de aangeleverde structuur. Privacyvelden zoals `raw_json`
en `chat_sender` worden niet geselecteerd.