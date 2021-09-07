import optuna
from typing import Tuple, Dict, Union, Generic
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline
import tempfile


def get_sequence_labeling_configspace(
    trial: optuna.trial.Trial,
) -> Tuple[int, Dict[str, Union[int, float, bool, str]], Dict[str, Union[int, float]]]:
    hidden_size = trial.suggest_int("hidden_size", low=128, high=2048, step=128)
    use_rnn = trial.suggest_categorical("use_rnn", [False, True])
    rnn_cfg = {}
    if use_rnn:
        rnn_cfg = {
            "rnn_layers": trial.suggest_int("rnn_layers", 1, 3),
            "rnn_type": trial.suggest_categorical("rnn_type", ["LSTM", "GRU"]),
        }

    task_model_kwargs = {
        "use_rnn": use_rnn,
        "dropout": trial.suggest_discrete_uniform("dropout", low=0, high=0.5, q=0.05),
        "word_dropout": trial.suggest_discrete_uniform("word_dropout", low=0, high=0.1, q=0.01),
        "locked_dropout": trial.suggest_discrete_uniform("locked_dropout", low=0, high=0.5, q=0.05),
        "reproject_embeddings": trial.suggest_categorical("reproject_embeddings", [False, True]),
        # "train_initial_hidden_state": trial.suggest_categorical(
        #     "train_initial_hidden_state", [False, True]
        # ),
        "use_crf": trial.suggest_categorical("use_crf", [False, True]),
        **rnn_cfg,
    }
    task_train_kwargs = {
        "learning_rate": trial.suggest_loguniform(
            "learning_rate", low=0.001, high=0.1,
        ),
        "mini_batch_size": trial.suggest_int("mini_batch_size", low=16, high=256, step=16),
        "max_epochs": trial.suggest_int("epochs", low=1, high=5, step=1),
        "param_selection_mode": True,
        "save_final_model": False
    }
    return hidden_size, task_model_kwargs, task_train_kwargs


def objective(trial: optuna.trial.Trial) -> float:
    hidden_size, task_model_kwargs, task_train_kwargs = get_sequence_labeling_configspace(trial)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = HuggingFaceSequenceLabelingPipeline(
            dataset_name="clarin-pl/kpwr-ner",
            embedding_name="clarin-pl/roberta-polish-kgr10",
            input_column_name="tokens",
            target_column_name="ner",
            output_path=tmp_dir,
            hidden_size=hidden_size,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
        )
        result = pipeline.run()

    return result["seqeval__mode_None__scheme_None"]["overall_f1"]
