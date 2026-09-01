---
name: Jupyter artifact-routing
description: Niet-obvious routingvoorwaarden voor een volledig werkende JupyterLab-preview in dit multi-artifactproject.
---

Gebruik de echte, bewerkbare Jupyter-notebook als primaire werk- en previewsurface; bouw geen aparte appweergave om de notebook heen. Start JupyterLab als de beheerde service van het root-webartifact en laat het luisteren op de door Replit geïnjecteerde `PORT`. Reserveer `/api` niet voor een ander artifact zolang Jupyter op `/` draait.

**Why:** Een losse workflow kan lokaal gezond zijn maar buiten de artifact-router onbereikbaar blijven. JupyterLab gebruikt zelf root-relative `/api/*`- en WebSocket-routes; een specifieker API-artifact op `/api` onderschept die verzoeken en veroorzaakt een half geladen interface met 502-fouten.

**How to apply:** Houd de artifact-preview op `/`, stuur die server-side naar een schone vaste Jupyter-workspace die `notebooks/momants_sentiment.ipynb` direct opent, en wis die workspace-layout bij een workflowstart. Laat het dev-command de geïnjecteerde poort gebruiken en verplaats ongebruikte artifact-routes weg van `/api`. Controleer visueel dat de cellen in de editor staan en dat de kernel-WebSocket actief is.