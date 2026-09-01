---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Ik zie brand bij de foodtrucks.
- text: hoeveel kost een parkeerplek
- text: Wat kost een consumptie gemiddeld?
- text: het toilet bij vak 3 is verstopt
- text: ik heb betaald maar geen bevestiging ontvangen
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-multilingual-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 6 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                          | Examples                                                                                                                                                                                          |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Informatie opvragen            | <ul><li>'Hoe laat gaan de deuren open?'</li><li>'Waar kan ik het programma vinden?'</li><li>'Is er een kluisje te huur op het terrein?'</li></ul>                                                 |
| Probleem of Incident oplossen  | <ul><li>'Mijn ticket scant niet bij de ingang.'</li><li>'Ik heb mijn polsbandje kapot getrokken.'</li><li>'Er is geld van mijn cashcard afgeschreven maar ik zie geen saldo.'</li></ul>           |
| Transactie / Mutatie uitvoeren | <ul><li>'Ik wil een extra ticket bijkopen.'</li><li>'Kan ik mijn ticket omzetten naar een ander type?'</li><li>'Ik wil mijn parkeerplek annuleren.'</li></ul>                                     |
| Actievere Navigatiehulp        | <ul><li>'Hoe kom ik van de camping naar het hoofdpodium?'</li><li>'Waar vind ik het dichtstbijzijnde toilet vanaf hier?'</li><li>'Kun je me de route naar de parkeerplaats laten zien?'</li></ul> |
| Systeem bedienen               | <ul><li>'Kun je me doorverbinden met een medewerker?'</li><li>'Stop met berichten sturen.'</li><li>'Kan ik dit gesprek opnieuw beginnen?'</li></ul>                                               |
| Noodgeval melden               | <ul><li>'Er ligt iemand bewusteloos bij podium 2, help!'</li><li>'Ik zie brand bij de foodtrucks.'</li><li>'Er is een vechtpartij aan de gang bij de ingang.'</li></ul>                           |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("hoeveel kost een parkeerplek")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 7.4891 | 12  |

| Label                          | Training Sample Count |
|:-------------------------------|:----------------------|
| Informatie opvragen            | 22                    |
| Probleem of Incident oplossen  | 20                    |
| Transactie / Mutatie uitvoeren | 15                    |
| Actievere Navigatiehulp        | 12                    |
| Systeem bedienen               | 11                    |
| Noodgeval melden               | 12                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 16)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 2
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0435 | 1    | 0.2124        | -               |

### Framework Versions
- Python: 3.11.14
- SetFit: 1.1.3
- Sentence Transformers: 5.7.0
- Transformers: 4.57.6
- PyTorch: 2.13.0+cpu
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->