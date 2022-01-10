# State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

[![CI Main](https://github.com/CLARIN-PL/embeddings/actions/workflows/python_poetry_main.yml/badge.svg)](https://github.com/CLARIN-PL/embeddings/actions/workflows/python_poetry_main.yml)

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
    embedding_name="allegro/herbert-base-cased",
    input_column_name="text",
    target_column_name="target",
    output_path="."
)

print(pipeline.run())

```

**Important Remark**: As the library is still under active development, default model hyperparameters **MAY BE NOT OPTIMAL**. 
We encourage running [OptimizedPipelines](https://github.com/cLARIN-PL/embeddings/#optimized-pipelines) as the first step to select appropriate hyperparameters. 

# Conventions

We use many of the HuggingFace concepts such as models (https://huggingface.co/models) or  datasets (https://huggingface.co/datasets) to make our library as easy to use as it is possible. We want to enable users to create, customise, test, and execute NLP/NLU/SLU tasks in the fastest possible manner. 
Moreover, we present easy to use static embeddings, that were trained by CLARIN-PL.


# Pipelines

We share predefined pipelines for common NLP tasks with corresponding scripts. 
For Transformer based pipelines we utilize [PyTorch Lighting](https://www.pytorchlightning.ai) trainers with Transformers [AutoModels](https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModel). 
For static embedding based pipelines we use [Flair](https://github.com/flairNLP/flair) library under the hood.

**REMARK**: As currently we haven't blocked transformers based pipelines from **flair** pipelines we **may remove it in the nearest future.** We encourage to use **Lightning** based pipelines for transformers.
### Transformer embedding based pipelines (e.g. Bert, RoBERTA, Herbert):

| Task                | Class                                                                                   | Script                                                                                                  | 
|---------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Text classification | [LightningClassificationPipeline](embeddings/pipeline/lightning_classification.py)      | [evaluate_lightning_document_classification.py](examples/evaluate_lightning_document_classification.py) |
| Sequence labelling  | [LightningSequenceLabelingPipeline](embeddings/pipeline/lightning_sequence_labeling.py) | [evaluate_lightning_sequence_labeling.py](examples/evaluate_lightning_sequence_labeling.py)             |


### Static embedding based pipelines (e.g. word2vec, fasttext)

| Task                         | Class                                                                               | Script                                                                                        |
|------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Text classification          | [FlairClassificationPipeline](embeddings/pipeline/flair_classification.py)          | [evaluate_document_classification.py](examples/evaluate_document_classification.py)           |
| Sequence labelling           | [FlairSequenceLabelingPipeline](embeddings/pipeline/flair_sequence_labeling.py)     | [evaluate_sequence_labelling.py](examples/evaluate_sequence_labelling.py)                     |
| Sequence pair classification | [FlairPairClassificationPipeline](embeddings/pipeline/flair_pair_classification.py) | [evaluate_document_pair_classification.py](examples/evaluate_document_pair_classification.py) |


 
## Writing custom HuggingFace-based pipeline

```python
from pathlib import Path

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)

dataset = HuggingFaceDataset("clarin-pl/polemo2-official")
data_loader = HuggingFaceDataLoader()
transformation = ClassificationCorpusTransformation("text", "target")
embedding = AutoFlairDocumentEmbedding.from_hub("allegro/herbert-base-cased")
task = TextClassification(Path("."))
model = FlairModel(embedding, task)
evaluator = TextClassificationEvaluator()

pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
result = pipeline.run()
```

# Running tasks scripts

All up-to-date examples can be found under [examples](examples/) path.

```bash
cd examples
```

## Run classification task

The example with non-default arguments

```bash
python evaluate_lightning_document_classification.py \
    --embedding-name allegro/herbert-base-cased \
    --dataset-name clarin-pl/polemo2-official \
    --input-columns-name text \
    --target-column-name target
```

## Run sequence labeling task

The example with default language model and dataset. 

```bash
python evaluate_lightning_sequence_labeling.py
```

## Run pair classification task

The example with static embedding model.

```bash
python evaluate_document_pair_classification.py \
    --embedding-name clarin-pl/word2vec-kgr10
```


# Compatible datasets
As most datasets in huggingface repository should be compatible with our pipelines, there are several datasets that were tested by the authors.

| dataset name               	| task type                                 	| input_column_name(s)       	| target_column_name  	| description                                                        	|
|----------------------------	|-------------------------------------------	|----------------------------	|---------------------	|------------------------------------------------------------	        |
| [nkjp-ner](https://huggingface.co/datasets/nkjp-ner)                   	| classification (named entity recognition) 	| sentence                   	| target              	| The manually annotated 1-million word subcorpus of the National Corpus of Polish.                 	   |
| [clarin-pl/polemo2-official](https://huggingface.co/datasets/clarin-pl/polemo2-official ) 	| classification  (sentiment analysis)          | text      | target              	| A corpus of consumer reviews from 4 domains: medicine, hotels, products and school.	               |
| [cdsc<br>(cdsc-e)*](https://huggingface.co/datasets/cdsc)             | textual entailment, pair classification           | (sentence_A, sentence_B)   	| entailment_judgment 	| The polish sentence pairs which are human-annotated for semantic relatedness and entailment.          |
| [dyk**](https://huggingface.co/datasets/dyk)                        	| question answering, pair classification   	    | (question, answer)           	| target              	| The Did You Know (pol. Czy wiesz?) dataset consists of human-annotated question-answer pairs         |
| [psc**](https://huggingface.co/datasets/psc)                        	| text summarization, pair classification   	    | (extract_text, summary_text) 	| label               	| The Polish Summaries Corpus contains news articles and their summaries.                        	   |
<br />
<sup>*to load the dataset pass name='cdsc-e' in load_dataset_kwargs: HuggingFaceDataset("cdsc", name="cdsc-e")</sup><br />
<sup>**only pair classification task is supported for now</sup>

# Passing task model and task training parameters to predefined flair pipelines
Model and training parameters can be controlled via `task_model_kwargs` and 
`task_train_kwargs` parameters. 

## Example with `polemo2` dataset.

```python
from embeddings.pipeline.flair_classification import FlairClassificationPipeline

pipeline = FlairClassificationPipeline(
    dataset_name="clarin-pl/polemo2-official",
    embedding_name="allegro/herbert-base-cased",
    input_column_name="text",
    target_column_name="target",
    output_path=".",
    task_model_kwargs={
        "loss_weights": {
            "plus": 2.0,
            "minus": 2.0
        }
    },
    task_train_kwargs={
        "learning_rate": 0.01,
        "max_epochs": 20
    }
)

print(pipeline.run())
```

# Static embeddings

Computed vectors are stored in [Flair](https://github.com/flairNLP/flair) structures - [Sentences](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md).

## Document embeddings

```python
from flair.data import Sentence

from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding

sentence = Sentence("Myśl z duszy leci bystro, Nim się w słowach złamie.")

embedding = AutoFlairDocumentEmbedding.from_hub("clarin-pl/word2vec-kgr10")
embedding.embed([sentence])

print(sentence.embedding)
```

## Word embeddings

```python
from flair.data import Sentence

from embeddings.embedding.auto_flair import AutoFlairWordEmbedding

sentence = Sentence("Myśl z duszy leci bystro, Nim się w słowach złamie.")

embedding = AutoFlairWordEmbedding.from_hub("clarin-pl/word2vec-kgr10")
embedding.embed([sentence])

for token in sentence:
    print(token)
    print(token.embedding)
```

# Available embedding models for Polish

Instead of the `allegro/herbert-base-cased` model, you can pass any model from [Hugging Face Hub](https://huggingface.co/models) that is compatible with [Transformers](https://huggingface.co/transformers/) or with our library. 

| Embedding                                                                   | Type         | Description                                                      |
|-----------------------------------------------------------------------------|--------------|------------------------------------------------------------------|
| [clarin-pl/herbert-kgr10](https://huggingface.co/clarin-pl/herbert-kgr10)   | bert         | HerBERT Large  trained on supplementary data - the KGR10 corpus. |
| [clarin-pl/fastText-kgr10](https://huggingface.co/clarin-pl/fastText-kgr10) | static, word | FastText trained on trained on the KGR10 corpus.                 |
| [clarin-pl/word2vec-kgr10](https://huggingface.co/clarin-pl/word2vec-kgr10) | static, word | Word2vec trained on trained on the KGR10 corpus.                 |
| ...                                                                         |


# Optimized pipelines.

## Transformers embeddings

| Task                          | Optimized Pipeline                                                                       |
|-------------------------------|------------------------------------------------------------------------------------------|
| Lightning Text Classification | [OptimizedLightingClassificationPipeline](embeddings/pipeline/lightning_hps_pipeline.py) | 
| Lightning Sequence Labeling   | -                                                                                        |



## Static embeddings

| Task                           | Optimized Pipeline                                                                    |
|--------------------------------|---------------------------------------------------------------------------------------|
| Flair Text Classification      | [OptimizedFlairClassificationPipeline](embeddings/pipeline/flair_hps_pipeline.py)     | 
| Flair Pair Text Classification | [OptimizedFlairPairClassificationPipeline](embeddings/pipeline/flair_hps_pipeline.py) |
| Flair Sequence Labeling        | [OptimizedFlairSequenceLabelingPipeline](embeddings/pipeline/flair_hps_pipeline.py)   |


## Example with Text Classification

Optimized pipelines can be run via following snippet of code:

```python

from embeddings.hyperparameter_search.lighting_configspace import LightingTextClassificationConfigSpace
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingClassificationPipeline

pipeline = OptimizedLightingClassificationPipeline(
    config_space=LightingTextClassificationConfigSpace(
        embedding_name="allegro/herbert-base-cased"
    ),
    dataset_name="clarin-pl/polemo2-official",
    input_column_name="text",
    target_column_name="target",
).persisting(best_params_path="best_prams.yaml", log_path="hps_log.pickle")
df, metadata = pipeline.run()
```

### Training model with obtained parameters

After the parameters search process we can train model with best parameters found.
But firstly we have to set `output_path` parameter, which is not automatically generated from `OptimizedLightingClassificationPipeline`.

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

Instead of performing search with single embedding model we can search with multiple embedding models via passing them as list to ConfigSpace.

```python
pipeline = OptimizedLightingClassificationPipeline(
    config_space=LightingTextClassificationConfigSpace(
        embedding_name=["allegro/herbert-base-cased", "clarin-pl/roberta-polish-kgr10"]
    ),
    dataset_name="clarin-pl/polemo2-official",
    input_column_name="text",
    target_column_name="target",
).persisting(best_params_path="best_prams.yaml", log_path="hps_log.pickle")
```
