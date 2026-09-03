---
name: Topic classification strategy
description: Durable model choice and validation rules for Momants topic classification.
---

Classify topics bottom-up with normalized multilingual sentence embeddings: compare each visitor message directly with all active event-specific subtopic descriptions and internal None prototypes, then derive the main topic from the winning seed row. Do not restore NLI zero-shot classification or a main-topic-first hierarchy.

**Why:** Both short-label and richly described NLI variants performed worse than the largest-class baseline and showed systematic label bias. The embedding approach validated above the required main-topic accuracy and macro-recall while rejecting all smalltalk in the synthetic benchmark.

**How to apply:** Keep similarity explicitly described as cosine similarity rather than probability. Validate every model or taxonomy change against the normalized-text answer key, including macro-recall, smalltalk rejection, the largest-class baseline, all-five-topic coverage, and the full confusion matrix.