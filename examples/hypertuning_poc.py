import optuna
from embeddings.hyperparameter_search.poc import objective


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=441),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=10, timeout=600)
    print(study.trials_dataframe())


if __name__ == "__main__":
    main()
