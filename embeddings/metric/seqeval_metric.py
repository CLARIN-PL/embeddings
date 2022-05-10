import importlib
from typing import Any, Dict, List, Optional, Union

import datasets
from seqeval.metrics import accuracy_score, classification_report

_CITATION = """\
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
"""

_DESCRIPTION = """\
seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.
This is well-tested by using the Perl script conlleval, which can be used for
measuring the performance of a system that has processed the CoNLL-2000 shared task data.
seqeval supports following formats:
IOB1
IOB2
IOE1
IOE2
IOBES
See the [README.md] file at https://github.com/chakki-works/seqeval for more information.
"""

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.
Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    suffix: True if the IOB prefix is after type, False otherwise. default: False
    scheme: Specify target tagging scheme. Should be one of ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"].
        default: None
    mode: Whether to count correct entity labels with incorrect I/B tags as true positives or not.
        If you want to only count exact matches, pass mode="strict". default: None.
    sample_weight: Array-like of shape (n_samples,), weights for individual samples. default: None
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be on of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.
Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:
    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = datasets.load_metric("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0
"""


class SeqevalMetric(datasets.Metric):
    # the class is rewritten basing on
    # https://github.com/huggingface/datasets/blob/master/metrics/seqeval/seqeval.py
    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/chakki-works/seqeval",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(  # type: ignore
                {
                    "predictions": datasets.Sequence(
                        datasets.Value("string", id="label"), id="sequence"
                    ),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="label"), id="sequence"
                    ),
                }
            ),
            codebase_urls=["https://github.com/chakki-works/seqeval"],
            reference_urls=["https://github.com/chakki-works/seqeval"],
        )

    def _compute(  # type: ignore
        self,
        predictions,
        references,
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = "warn",
    ) -> Dict[str, Any]:
        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(
                    f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}"
                )
        report = classification_report(
            y_true=references,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        macro_score = report.pop("macro avg")
        weighted_score = report.pop("weighted avg")
        micro_score = report.pop("micro avg")

        classes = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "support": score["support"],
            }
            for type_name, score in report.items()
        }
        scores = {
            "classes": classes,
            "precision_micro": micro_score["precision"],
            "recall_micro": micro_score["recall"],
            "f1_micro": micro_score["f1-score"],
            "precision_macro": macro_score["precision"],
            "recall_macro": macro_score["recall"],
            "f1_macro": macro_score["f1-score"],
            "precision_weighted": weighted_score["precision"],
            "recall_weighted": weighted_score["recall"],
            "f1_weighted": weighted_score["f1-score"],
            "accuracy": accuracy_score(y_true=references, y_pred=predictions),
        }
        return scores

    def __str__(self) -> str:
        return "Seqeval"
