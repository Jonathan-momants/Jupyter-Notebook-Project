---
name: Local intent tokenizer warning
description: Runtime warning emitted when loading the existing local SetFit intent model.
---

The existing local intent model emits a Transformers warning that its saved tokenizer has an incorrect regex pattern and mentions `fix_mistral_regex=True`.

**Why:** This appeared reproducibly during direct classification of the full export after a dependency-version change. The intent model is independently validated and was explicitly out of scope for modification, so changing its load behavior inside unrelated work could silently change established predictions.

**How to apply:** Treat the warning as a separate intent-axis validation task. Do not suppress it or enable the suggested tokenizer option until old-versus-new predictions and the direct intent metrics have been compared.