# Momants Sentiment Notebook

Een Python/Jupyter-leeromgeving die synthetische bezoekersgesprekken groepeert en met een meertalig Hugging Face-model op sentiment classificeert.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `jupyter lab` — start JupyterLab vanuit de projectomgeving

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `notebooks/00_momants_sentiment.ipynb` — volledige sentimentanalyse als leerpad
- `notebooks/README.md` — korte werkwijze voor notebooks
- `data/voorbeeld_gesprekken.csv` — uitsluitend synthetische testgesprekken
- `pyproject.toml` — Python- en Jupyter-afhankelijkheden

## Architecture decisions

- De notebook gebruikt één laadfunctie zodat later alleen de bron van lokaal CSV-pad naar endpoint hoeft te veranderen.
- Alleen zes expliciet toegestane kolommen worden ingelezen; uitgesloten privacykolommen komen niet in het DataFrame.
- Gesprekssentiment is voorlopig het sentiment van het laatste bruikbare bezoekersbericht, omdat dit de eindtoestand eenvoudig uitlegbaar benadert.
- Het model-ID en de labelmapping staan los bovenaan zodat een modelwissel beperkt blijft.

## Product

- Groepeert berichtregels per gesprek en sorteert ze chronologisch.
- Negeert botberichten, lege tekst en kale URL's.
- Classificeert ieder bruikbaar bezoekersbericht met een meertalig sentimentmodel.
- Toont één lokale eindtabel met gesprekssentiment, zekerheid en uitleg.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Gebruik geen echte bezoekersdata zolang telefoonnummers in vrije tekst nog niet worden gefilterd.
- De eerste notebookrun downloadt het Hugging Face-model en duurt daardoor langer.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
