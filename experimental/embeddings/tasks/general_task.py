import mlflow


def task():
    dataset = load_dataset("dataset_name")
    embedding = load_embedding("embedding_name")
    model = load_model("model_type_and_name")

    metrics = evaluate(dataset, embedding, model)

    # we could use mlflow internally to log results WDYT?
    mlflow.log_metrics(metrics)


if __name__ == "__main__":
    task()
