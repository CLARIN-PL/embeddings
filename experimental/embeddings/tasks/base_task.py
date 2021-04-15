import abc
from pathlib import Path
from typing import Dict, Tuple
from typing import Optional

import flair
from flair.data import FlairDataset, Corpus
from flair.trainers import ModelTrainer
from flair.training_utils import Result


class BaseTask(abc.ABC):
    def __init__(self, model: flair.nn.Model, output_path: str):
        self.output_path = Path(output_path)
        self.model = model
        self.trainer: Optional[ModelTrainer] = None

    def fit(self, dataset_name: Corpus, embedding_name: str) -> Dict:

        if embedding in ['transformer-alike']:
            embeddings = TransformerDocumentEmbeddings(embedding, fine_tune=False)
        elif embedding in ['static-embedding']:
            embeddings = load_word_embeddings(embedding)

        dataset = load_dataset(dataset_name)

        # label_dict = flair_dataset.make_label_dictionary()
        self.model = TextClassifier(embeddings, label_dictionary=dataset.label_dict)

        self.trainer = ModelTrainer(model, dataset.flair_format)
        log = self.trainer.train(base_path=self.output_path)
        return log

    def evaluate(self, dataset: BaseDataset) -> Tuple[Result, float]:
        return self.model.evaluate(dataset)
