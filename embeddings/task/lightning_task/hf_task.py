from enum import Enum


class HuggingFaceTaskName(Enum):
    text_classification = "sequence-classification"
    sequence_labeling = "token-classification"
    question_answering = "question-answering"
