---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Mijn invoer wordt niet opgeslagen.
- text: Wat zijn de voorwaarden van deze dienst?
- text: Maak een nieuwe aanvraag voor mij aan.
- text: Schrijf mij in voor de volgende beschikbare datum.
- text: Ik wil deze bestelling annuleren.
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
| Label                          | Examples                                                                                                                                                                                     |
|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Informatie opvragen            | <ul><li>'Welke openingstijden gelden er?'</li><li>'Wat zijn de voorwaarden van deze dienst?'</li><li>'Hoeveel kost het standaardpakket?'</li></ul>                                           |
| Probleem of Incident oplossen  | <ul><li>'De pagina geeft steeds een foutmelding.'</li><li>'Mijn invoer wordt niet opgeslagen.'</li><li>'De verbinding valt telkens weg.'</li></ul>                                           |
| Transactie / Mutatie uitvoeren | <ul><li>'Ik wil mijn afspraak verplaatsen.'</li><li>'Kun je mijn adres aanpassen?'</li><li>'Ik wil deze bestelling annuleren.'</li></ul>                                                     |
| Actievere Navigatiehulp        | <ul><li>'Waar moet ik klikken om mijn gegevens te vinden?'</li><li>'Kun je mij stap voor stap naar het formulier leiden?'</li><li>'Ik kan de juiste pagina niet vinden.'</li></ul>           |
| Systeem bedienen               | <ul><li>'Open het instellingenmenu.'</li><li>'Start de controle opnieuw.'</li><li>'Log mij uit van dit apparaat.'</li></ul>                                                                  |
| Noodgeval melden               | <ul><li>'Er is direct hulp nodig vanwege een gevaarlijke situatie.'</li><li>'Ik moet met spoed iemand spreken.'</li><li>'Er is een noodgeval en we hebben nu ondersteuning nodig.'</li></ul> |

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
preds = model("Ik wil deze bestelling annuleren.")
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
| Word count   | 3   | 6.5278 | 10  |

| Label                          | Training Sample Count |
|:-------------------------------|:----------------------|
| Informatie opvragen            | 6                     |
| Probleem of Incident oplossen  | 6                     |
| Transactie / Mutatie uitvoeren | 6                     |
| Actievere Navigatiehulp        | 6                     |
| Systeem bedienen               | 6                     |
| Noodgeval melden               | 6                     |

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
| 0.1111 | 1    | 0.2011        | -               |

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