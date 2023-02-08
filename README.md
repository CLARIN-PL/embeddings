# State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

[![CI Main](https://github.com/CLARIN-PL/embeddings/actions/workflows/python_poetry_main.yml/badge.svg)](https://github.com/CLARIN-PL/embeddings/actions/workflows/python_poetry_main.yml)

:construction:️ The library is currently in an active development state. Some functionalities may be
subject to change before the stable release. Users can track our
milestones [here](https://github.com/CLARIN-PL/embeddings/milestones).

# Installation

```bash
pip install clarinpl-embeddings
```

# Example

Text-classification with polemo2 dataset and transformer-based embeddings

```python
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline

pipeline = LightningClassificationPipeline(
    dataset_name_or_path="clarin-pl/polemo2-official",
    embedding_name_or_path="allegro/herbert-base-cased",
    input_column_name="text",
    target_column_name="target",
    output_path="."
)

print(pipeline.run())

```

#### :warning: As for now, default pipeline model hyperparameters may provide poor results. It will be subject to change in further releases. We encourage users to use [Optimized Pipelines](#optimized-pipelines) to select appropriate hyperparameters.

# Conventions

We use many of the HuggingFace concepts such as models (https://huggingface.co/models) or
datasets (https://huggingface.co/datasets) to make our library as easy to use as it is possible. We
want to enable users to create, customise, test, and execute NLP / NLU / SLU tasks in the fastest
possible manner. Moreover, we present easy to use static embeddings, that were trained by CLARIN-PL.

# Pipelines

We share predefined pipelines for common NLP tasks with corresponding scripts. For Transformer based
pipelines we utilize [PyTorch Lighting](https://www.pytorchlightning.ai) ⚡ trainers with
Transformers [AutoModels](https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModel)
. 


### Transformer embedding based pipelines (e.g. Bert, RoBERTA, Herbert):

| Task                | Class                                                                                   | Script                                                                                                  | 
|---------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Text classification | [LightningClassificationPipeline](embeddings/pipeline/lightning_classification.py)      | [evaluate_lightning_document_classification.py](examples/evaluate_lightning_document_classification.py) |
| Sequence labelling  | [LightningSequenceLabelingPipeline](embeddings/pipeline/lightning_sequence_labeling.py) | [evaluate_lightning_sequence_labeling.py](examples/evaluate_lightning_sequence_labeling.py)             |


# Running tasks scripts

All up-to-date examples can be found under [examples](examples/) path.

```bash
cd examples
```

## Run classification task

The example with non-default arguments

```bash
python evaluate_lightning_document_classification.py \
    --embedding-name-or-path allegro/herbert-base-cased \
    --dataset-name clarin-pl/polemo2-official \
    --input-columns-name text \
    --target-column-name target
```

## Run sequence labeling task

The example with default language model and dataset.

```bash
python evaluate_lightning_sequence_labeling.py
```

# Compatible datasets

As most datasets in HuggingFace repository should be compatible with our pipelines, there are
several datasets that were tested by the authors.

| dataset name                                                                                  | task type                                          | input_column_name(s)         | target_column_name  | description                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------|------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [clarin-pl/kpwr-ner](https://huggingface.co/datasets/clarin-pl/kpwr-ner)                      | sequence labeling (named entity recognition)       | tokens                       | ner                 | KPWR-NER is a part of the  Polish Corpus of Wrocław University of Technology (KPWr). Its objective is recognition of named entities, e.g., people, institutions etc.                      |
| [clarin-pl/polemo2-official](https://huggingface.co/datasets/clarin-pl/polemo2-official )       | classification  (sentiment analysis)             | text                         | target              | A corpus of consumer reviews from 4 domains: medicine, hotels, products and school.                                                                                                       |
| [clarin-pl/2021-punctuation-restoration](https://huggingface.co/datasets/clarin-pl/2021-punctuation-restoration)                      | punctuation restoration                           | text_in                      | text_out            | Dataset contains original texts and ASR output. It is a part of PolEval 2021 Competition.                                                                                                 |
| [clarin-pl/nkjp-pos](https://huggingface.co/datasets/clarin-pl/nkjp-pos)                      | sequence labeling (part-of-speech tagging)       | tokens                       | pos_tags            | NKJP-POS is a part of the National Corpus of Polish. Its objective is part-of-speech tagging, e.g., nouns, verbs, adjectives, adverbs, etc.                                               |
| [clarin-pl/aspectemo](https://huggingface.co/datasets/clarin-pl/aspectemo)                      | sequence labeling (sentiment classification)     | tokens                       | labels              | AspectEmo Corpus is an extended version of a publicly available PolEmo 2.0 corpus of Polish customer reviews used in many projects on the use of different methods in sentiment analysis. |
| [laugustyniak/political-advertising-pl](https://huggingface.co/datasets/laugustyniak/political-advertising-pl)                      | sequence labeling (political advertising )                         | tokens                       | tags                | First publicly open dataset for detecting specific text chunks and categories of political advertising in the Polish language.                                                            |
| [laugustyniak/abusive-clauses-pl](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl)                      | classification  (abusive-clauses)                           | text                          | class               | Dataset with Polish abusive clauses examples.                                                                                                                                             |
| [allegro/klej-dyk](https://huggingface.co/datasets/allegro/klej-dyk)                             | pair classification (question answering)*        | (question, answer)           | target              | The Did You Know (pol. Czy wiesz?) dataset consists of human-annotated question-answer pairs.                                                                                             |
| [allegro/klej-psc](https://huggingface.co/datasets/allegro/klej-psc)                             | pair classification (text summarization)*        | (extract_text, summary_text) | label               | The Polish Summaries Corpus contains news articles and their summaries.                                                                                                                   |
| [allegro/klej-cdsc-e](https://huggingface.co/datasets/allegro/klej-cdsc-e)                    | pair classification (textual entailment)*        | (sentence_A, sentence_B)     | entailment_judgment | The polish sentence pairs which are human-annotated for textualentailment.                                                                                                                |

<br />

[//]: # ()
<sup>*only pair classification task is supported for now</sup>

# Passing task model and task training parameters to predefined pipelines

Model and training parameters can be controlled via `task_model_kwargs` and
`task_train_kwargs` parameters that can be populated using the advanced config. Tutorial on how to
use configs can be found in `/tutorials` directory of the repository. Two types of config are
defined in our library: BasicConfig and AdvancedConfig. In summary, the BasicConfig takes arguments
and automatically assign them into proper keyword group, while the AdvancedConfig takes as the input
keyword groups that should be already correctly mapped.

The list of available config can be found below:



#### **Lightning**:

- LightningBasicConfig
- LightningAdvancedConfig

## Example with `polemo2` dataset

### Lightning pipeline

```python
from embeddings.config.lightning_config import LightningBasicConfig
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline

config = LightningBasicConfig(
    learning_rate=0.01, max_epochs=1, max_seq_length=128, finetune_last_n_layers=0,
    accelerator="cpu"
)

pipeline = LightningClassificationPipeline(
    embedding_name_or_path="allegro/herbert-base-cased",
    dataset_name_or_path="clarin-pl/polemo2-official",
    input_column_name=["text"],
    target_column_name="target",
    load_dataset_kwargs={
        "train_domains": ["hotels", "medicine"],
        "dev_domains": ["hotels", "medicine"],
        "test_domains": ["hotels", "medicine"],
        "text_cfg": "text",
    },
    output_path=".",
    config=config
)
```

You can also define an Advanced config with populated keyword arguments. In general, the keywords
are passed to the object when constructing specific pipelines. We can identify and trace the keyword
arguments to find the possible arguments that can be set in the config kwargs.

```python
from embeddings.config.lightning_config import LightningAdvancedConfig

config = LightningAdvancedConfig(
    finetune_last_n_layers=0,
    task_train_kwargs={
        "max_epochs": 1,
        "devices": "auto",
        "accelerator": "cpu",
        "deterministic": True,
    },
    task_model_kwargs={
        "learning_rate": 5e-4,
        "use_scheduler": False,
        "optimizer": "AdamW",
        "adam_epsilon": 1e-8,
        "warmup_steps": 100,
        "weight_decay": 0.0,
    },
    datamodule_kwargs={
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
    },
    dataloader_kwargs={"num_workers": 0},
)
```

# Available embedding models for Polish

Instead of the `allegro/herbert-base-cased` model, user can pass any model
from [HuggingFace Hub](https://huggingface.co/models) that is compatible
with [Transformers](https://huggingface.co/transformers/) or with our library.

| Embedding                                                                   | Type         | Description                                                      |
|-----------------------------------------------------------------------------|--------------|------------------------------------------------------------------|
| [clarin-pl/herbert-kgr10](https://huggingface.co/clarin-pl/herbert-kgr10)   | bert         | HerBERT Large  trained on supplementary data - the KGR10 corpus. |
| ...                                                                         |

# Optimized pipelines

## Transformers embeddings

| Task                          | Optimized Pipeline                                                                         |
|-------------------------------|--------------------------------------------------------------------------------------------|
| Lightning Text Classification | [OptimizedLightingClassificationPipeline](embeddings/pipeline/lightning_hps_pipeline.py)   | 
| Lightning Sequence Labeling   | [OptimizedLightingSequenceLabelingPipeline](embeddings/pipeline/lightning_hps_pipeline.py) |


## Example with Text Classification

Optimized pipelines can be run via following snippet of code:

```python

from embeddings.config.lighting_config_space import LightingTextClassificationConfigSpace
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingClassificationPipeline

pipeline = OptimizedLightingClassificationPipeline(
    config_space=LightingTextClassificationConfigSpace(
        embedding_name_or_path="allegro/herbert-base-cased"
    ),
    dataset_name_or_path="clarin-pl/polemo2-official",
    input_column_name="text",
    target_column_name="target",
).persisting(best_params_path="best_prams.yaml", log_path="hps_log.pickle")
df, metadata = pipeline.run()
```

### Training model with obtained parameters

After the parameters search process we can train model with best parameters found. But firstly we
have to set `output_path` parameter, which is not automatically generated
from `OptimizedLightingClassificationPipeline`.

```python
metadata["output_path"] = "."
```

Now we are able to train the pipeline

```python
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline

pipeline = LightningClassificationPipeline(**metadata)
results = pipeline.run()
```

### Selection of best embedding model.

Instead of performing search with single embedding model we can search with multiple embedding
models via passing them as list to ConfigSpace.

```python
pipeline = OptimizedLightingClassificationPipeline(
    config_space=LightingTextClassificationConfigSpace(
        embedding_name_or_path=["allegro/herbert-base-cased", "clarin-pl/roberta-polish-kgr10"]
    ),
    dataset_name_or_path="clarin-pl/polemo2-official",
    input_column_name="text",
    target_column_name="target",
).persisting(best_params_path="best_prams.yaml", log_path="hps_log.pickle")
```
