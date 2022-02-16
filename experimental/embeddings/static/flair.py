import logging
from pathlib import Path
from pickle import UnpicklingError
from typing import Optional

import flair
import gensim
import numpy as np
import torch
from flair.embeddings import WordEmbeddings
from flair.file_utils import cached_path
from torch import nn

logging.getLogger("flair").setLevel(logging.INFO)


class WordEmbeddingsPL(WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText. This class is borrowed from
    flairNLP library and fixes error with loading embeddings for polish language.

    WordEmbeddings class source code:
    https://github.com/flairNLP/flair/blob/cbe683c51f1664bee8ca72eae317df29902af54e/flair/embeddings/token.py#L106
    """

    def __init__(
        self,
        embeddings: str,
        field: Optional[str] = None,
        fine_tune: bool = False,
        force_cpu: bool = True,
        stable: bool = False,
    ):
        super(WordEmbeddings, self).__init__()
        self.embeddings = embeddings

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        if fine_tune and force_cpu and flair.device.type != "cpu":
            raise ValueError(
                "Cannot train WordEmbeddings on cpu if the model is trained on gpu, set force_cpu=False"
            )

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/token"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() in ["glove", "en-glove"]:
            cached_path(f"{hu_path}/glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/glove.gensim", cache_dir=cache_dir)

        elif embeddings.lower() in ["turian", "en-turian"]:
            cached_path(f"{hu_path}/turian.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/turian", cache_dir=cache_dir)

        elif embeddings.lower() in ["extvec", "en-extvec"]:
            cached_path(f"{hu_path}/extvec.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/extvec.gensim", cache_dir=cache_dir)

        elif embeddings.lower() in ["pubmed", "en-pubmed"]:
            cached_path(
                f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(
                f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim", cache_dir=cache_dir
            )

        elif embeddings.lower() in ["crawl", "en-crawl"]:
            cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(
                f"{hu_path}/en-fasttext-crawl-300d-1M", cache_dir=cache_dir
            )

        elif embeddings.lower() in ["news", "en-news", "en"]:
            cached_path(f"{hu_path}/en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(
                f"{hu_path}/en-fasttext-news-300d-1M", cache_dir=cache_dir
            )

        elif embeddings.lower() in ["twitter", "en-twitter"]:
            cached_path(f"{hu_path}/twitter.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/twitter.gensim", cache_dir=cache_dir)

        elif len(embeddings.lower()) == 2:
            cached_path(
                f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(
                f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M", cache_dir=cache_dir
            )

        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(
                f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M", cache_dir=cache_dir
            )

        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )
        else:
            embeddings_path = Path(embeddings)

        self.name = str(embeddings_path)
        self.static_embeddings = not fine_tune
        self.fine_tune = fine_tune
        self.force_cpu = force_cpu
        self.field = field
        self.stable = stable

        if embeddings_path.suffix == ".bin":
            precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings_path), binary=True
            )
        else:
            try:
                precomputed_word_embeddings = gensim.models.KeyedVectors.load(str(embeddings_path))
            except UnpicklingError:
                # For polish models usually gensim cannot unpickle non-binary files with a method
                # gensim.models.KeyedVectors.load
                logging.warning(
                    "Couldn't unpickle model file. Unpickle model with different method."
                )
                precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    str(embeddings_path), binary=False
                )

        self.embedding_length: int = precomputed_word_embeddings.vector_size

        vectors = np.row_stack(
            (
                precomputed_word_embeddings.vectors,
                np.zeros(self.embedding_length, dtype="float"),
            )
        )
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(vectors), freeze=not fine_tune
        )

        try:
            # gensim version 4
            self.vocab = precomputed_word_embeddings.key_to_index
        except AttributeError:
            # gensim version 3
            self.vocab = {k: v.index for k, v in precomputed_word_embeddings.vocab.items()}

        if stable:
            self.layer_norm = nn.LayerNorm(self.embedding_length, elementwise_affine=fine_tune)
        else:
            self.layer_norm = None

        self.device = None
        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    @embedding_length.setter
    def embedding_length(self, value: int) -> None:
        self._embedding_length = value
