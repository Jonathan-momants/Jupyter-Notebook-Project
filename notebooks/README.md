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

Dit schrijft één tabel: `resultaten/sentiment_per_gesprek.csv`. Per gesprek
bevat die het sentiment en de zekerheid van zowel het eerste als het laatste
bruikbare bezoekersbericht. De oorspronkelijke berichttekst staat niet in de
uitvoer.

De processor ondersteunt Momants-CSV's met kolomkoppen en het headerloze
22-veldenformaat uit de aangeleverde structuur. Privacyvelden zoals `raw_json`
en `chat_sender` worden niet geselecteerd.