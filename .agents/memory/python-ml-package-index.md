---
name: Python ML package-index
description: Omgevingsregel voor het combineren van CPU-PyTorch en Hugging Face Transformers.
---

Koppel in `tool.uv.sources` uitsluitend `torch` aan de expliciete PyTorch CPU-index. Laat Transformers en alle andere Python-pakketten via de gewone Python-index oplossen.

**Why:** De automatische Torch-installatiestap kan een brede lijst pakketten aan de expliciete CPU-index koppelen. Transformers bestaat daar niet, waardoor de resolver ten onrechte meldt dat er geen Linux-versie beschikbaar is.

**How to apply:** Controleer na wijzigingen aan Python ML-afhankelijkheden of de bronmapping compact blijft. Herstel een brede mapping vóór het synchroniseren van de omgeving.