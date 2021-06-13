State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

# Installation

```bash
pip install clarinpl-embeddings
```

# Conventions

We use many of the HuggingFace concepts such as models (https://huggingface.co/models) or  datasets (https://huggingface.co/datasets) to make our library as easy to use as it is possible. We want to enable users to create, customise, test, and execute NLP/NLU/SLU tasks in the fastest possible manner.

## HuggingFace-based pipeline

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

# Run example tasks

```bash
cd .\examples
```

## Run classification task

```bash
python evaluate_document_classification.py --embedding-name allegro/herbert-base-cased --dataset-name clarin-pl/polemo2-official --input-column-name text --target-column-name target
```

## Run sequence tagging task

to be announced

