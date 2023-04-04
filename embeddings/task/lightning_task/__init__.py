from typing import Union

from .lightning_task import ClassificationLightningTask
from .question_answering import QuestionAnsweringTask

SUPPORTED_HF_TASKS = Union[ClassificationLightningTask, QuestionAnsweringTask]
