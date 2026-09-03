---
name: Answer-key wording parity
description: Canonical wording contract between synthetic visitor exports and topic answer keys.
---

Every canonical answer-key phrase must occur literally in the synthetic visitor export, including punctuation, while every visitor phrase must still resolve to one consistent normalized label pair.

**Why:** Normalized matching can hide punctuation drift and report complete scored coverage even though a literal audit still finds unused canonical phrases.

**How to apply:** When editing either fixture, check both literal key coverage and unique normalized matching before trusting the classification metrics.