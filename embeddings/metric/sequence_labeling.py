from enum import Enum
from typing import Final, Optional, Set

from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.seqeval_metric import SeqevalMetric
from embeddings.metric.unit_seqeval_metric import UnitSeqevalMetric


class EvaluationMode(str, Enum):
    UNIT = "unit"
    CONLL = "conll"
    STRICT = "strict"


class TaggingScheme(str, Enum):
    IOB1 = "IOB1"
    IOB2 = "IOB2"
    IOE1 = "IOE1"
    IOE2 = "IOE2"
    IOBES = "IOBES"
    BILOU = "BILOU"


SEQEVAL_EVALUATION_MODES: Final[Set[str]] = {EvaluationMode.CONLL, EvaluationMode.STRICT}


def get_sequence_labeling_metric(
    evaluation_mode: EvaluationMode, tagging_scheme: Optional[TaggingScheme] = None
) -> HuggingFaceMetric:
    if evaluation_mode in SEQEVAL_EVALUATION_MODES:
        if evaluation_mode == "strict" and not tagging_scheme:
            raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
        elif evaluation_mode == "conll" and tagging_scheme:
            raise ValueError("Tagging scheme can be set only in strict mode!")
        return HuggingFaceMetric(
            metric=SeqevalMetric(),
            compute_kwargs={
                "mode": evaluation_mode if evaluation_mode == "strict" else None,
                "scheme": tagging_scheme,
            },
        )
    elif evaluation_mode == "unit":
        return UnitSeqevalMetric()
    else:
        raise ValueError(
            f"Evaluation mode {evaluation_mode} not supported. Must be one of "
            f"[unit, conll, strict]."
        )
