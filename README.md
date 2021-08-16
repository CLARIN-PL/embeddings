State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

# Installation

```bash
pip install clarinpl-embeddings
```

# Example

Text-classification with polemo2 dataset and transformer-based embeddings

```python
from embeddings.pipeline.hugging_face_classification import HuggingFaceClassificationPipeline

pipeline = HuggingFaceClassificationPipeline(
    dataset_name="clarin-pl/polemo2-official",
    embedding_name="allegro/herbert-base-cased",
    input_column_name="text",
    target_column_name="target",
    output_path="."
)

print(pipeline.run())

```

# Conventions

We use many of the HuggingFace concepts such as models (https://huggingface.co/models) or  datasets (https://huggingface.co/datasets) to make our library as easy to use as it is possible. We want to enable users to create, customise, test, and execute NLP/NLU/SLU tasks in the fastest possible manner.


# Pipelines

We share predefined pipelines for common NLP tasks with corresponding scripts.

| Task | Class | Script |
| ---- | ---- | ---- |
| Text classification | [HuggingFaceClassificationPipeline](embeddings/pipeline/hugging_face_classification.py) | [evaluate_document_classification.py](examples/evaluate_document_classification.py) |
| Sequence labelling | [HuggingFaceSequenceLabelingPipeline](embeddings/pipeline/hugging_face_sequence_labeling.py) | [evaluate_sequence_labelling.py](examples/evaluate_sequence_labelling.py) |
| Sequence pair classification | [HuggingFacePairClassificationPipeline](embeddings/pipeline/hugging_face_pair_classification.py)| [evaluate_document_pair_classification.py](examples/evaluate_document_pair_classification.py) |

 
## Writing custom HuggingFace-based pipeline

```python
from pathlib import Path

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
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
embedding = FlairTransformerDocumentEmbedding("allegro/herbert-base-cased")
task = TextClassification(Path("."))
model = FlairModel(embedding, task)
evaluator = TextClassificationEvaluator()

pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
result = pipeline.run()
```

# Running tasks scripts

```bash
cd .\examples
```

## Run classification task

```bash
python evaluate_document_classification.py \
    --embedding-name allegro/herbert-base-cased \
    --dataset-name clarin-pl/polemo2-official \
    --input-column-name text \
    --target-column-name target

```

## Run sequence labeling task

The example with default language model and dataset. 

```bash
python evaluate_sequence_tagging.py
```

# Compatible datasets
There are several datasets available in the huggingface repository that are 
compatible with our pipeline.

| dataset name               	| task type                                 	| input_column_name(s)       	| target_column_name  	| url                                                        	|
|----------------------------	|-------------------------------------------	|----------------------------	|---------------------	|------------------------------------------------------------	|
| nkjp-ner                   	| classification (named entity recognition) 	| sentence                   	| target              	| https://huggingface.co/datasets/nkjp-ner                   	|
| clarin-pl/polemo2-official 	| classification                            	| text                       	| target              	| https://huggingface.co/datasets/clarin-pl/polemo2-official 	|
| cdsc<br>(cdsc-e)*             | pair classification                       	| (sentence_A, sentence_B)   	| entailment_judgment 	| https://huggingface.co/datasets/cdsc                       	|
| dyk**                        	| question answering, pair classification   	| (question, answer)           	| target              	| https://huggingface.co/datasets/dyk                        	|
| psc**                        	| text summarization, pair classification   	| (extract_text, summary_text) 	| label               	| https://huggingface.co/datasets/psc                        	|
<br />
<sup>*to load the dataset pass name='cdsc-e' in load_dataset_kwargs: HuggingFaceDataset("cdsc", name="cdsc-e")</sup><br />
<sup>**only pair classification task is supported for now</sup>

# Passing task model and task training parameters to predefined pipelines

Model and training parameters can be controlled via `task_model_kwargs` and 
`task_train_kwargs` parameters. 

## Example with `polemo2` dataset.   

```python
from embeddings.pipeline.hugging_face_classification import HuggingFaceClassificationPipeline

pipeline = HuggingFaceClassificationPipeline(
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
