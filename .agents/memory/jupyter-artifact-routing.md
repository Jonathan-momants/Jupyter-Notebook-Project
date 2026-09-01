---
name: Jupyter artifact-routing
description: Niet-obvious routingvoorwaarden voor een volledig werkende JupyterLab-preview in dit multi-artifactproject.
---

Start JupyterLab als de beheerde service van het root-webartifact en laat het luisteren op de door Replit geïnjecteerde `PORT`. Reserveer `/api` niet voor een ander artifact zolang Jupyter op `/` draait.

**Why:** Een losse workflow kan lokaal gezond zijn maar buiten de artifact-router onbereikbaar blijven. JupyterLab gebruikt zelf root-relative `/api/*`- en WebSocket-routes; een specifieker API-artifact op `/api` onderschept die verzoeken en veroorzaakt een half geladen interface met 502-fouten.

**How to apply:** Houd de Jupyter-preview op `/`, laat zijn dev-command de geïnjecteerde poort gebruiken, en verplaats eventuele ongebruikte artifact-routes weg van `/api`. Controleer zowel de notebookpagina als de kernel-WebSocket.