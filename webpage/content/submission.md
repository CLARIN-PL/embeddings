---
title: Submission
type: docs
url: "/submission"
geekdocNav: false
geekdocBreadcrumb: false
---

{{< pageHeader >}}

# Contribute to LEPISZCZE

We invite the community to contribute to `LEPISZCZE` by submitting model results. You can either manually fill in your submissions or use the `embeddings` library for automatic generation.

## Table of Contents
  * [1A. Manually Filled Submissions](#1a-manually-filled-submissions)
  * [1B. Automatically Generated Submissions](#1b-generation-submission-using-embeddings-library) 
  * [2. Submitting submission as PR](#2-submit-via-pull-request)

## 1A. Manually Filled Submissions

Submissions **must include** the following information:

| Required Submission Keys | Description |
| -- | -- |
| **submission_name** | Name of the submission file, following the convention: `{dataset_name}_{model_name}`. For the model `allegro/herbert-large-cased`, replace `/` with `__`. Example: for the `abusive_clauses` dataset and `allegro/herbert-large-cased` model, the name will be `abusive_clauses_allegro__herbert-large-cased`. |
| **dataset_name** | HuggingFace repository name of the dataset. Example: [`laugustyniak/abusive-clauses`](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl). |
| **dataset_version** | Version of the dataset, e.g., `0.0.1`. |
| **embedding_name** | HuggingFace repository name of the used embedding or model. Example: [`allegro/herbert-large-cased`](https://huggingface.co/allegro/herbert-large-cased). |
| **leaderboard_task_name** | Choose a task name from LEPISZCZE's leaderboard. Options: `Abusive Clauses Detection`, `Aspect-based Sentiment Analysis`, etc. |
| **metrics**| List of metrics as dictionaries. If no retraining was done, provide a single-element list. |
| **metrics_avg**| A dictionary of averaged metrics. If only one run was conducted, this matches the first element of the **metrics** sequence. |
| **metrics_std** | Mean standard deviation. For a single run, fill zeros for all provided metrics. |
| **averaged_over** | Number of runs performed. If only one evaluation was done, set to 1. |

There are also optional submission keys, but we strongly recommend including all information to improve reproducibility:

| Optional Submission Keys | Description |
| --- | --- |
| **hparams** | Mapping of hyperparameters with their values. |
| **packages** | Mapping of packages used for model training and evaluation, along with their versions. |

Submissions should be in `.json` format.


### Examples 

{{< collapse title="Information Retrieval sample submission file without optional fields." >}}
{
  "submission_name": "msmarco_bm_25",
  "dataset_name": "MSMARCO",
  "dataset_version": "0.0.1",
  "embedding_name": "BM25",
  "hparams": {},
  "packages": [],
  "leaderboard_task_name": "Information Retrieval",
  "metrics": [
    {
      "NDCG@10": 31.50,
      "MRR@10": 56.36
    }
  ],
  "metrics_avg": {
      "NDCG@10": 31.50,
      "MRR@10": 56.36
    },
  "metrics_std": {
      "NDCG@10": 0.0,
      "MRR@10": 0.0
    },
  "averaged_over": 1
}
{{< /collapse >}}


{{< collapse title="Question Answering sample submission file with packages provided." >}}

{
    "submission_name": "qa_all_Aleksandra__herbert-base-cased-finetuned-squad",
    "dataset_name": "qa_all",
    "dataset_version": "0.0.0",
    "embedding_name": "Aleksandra/herbert-base-cased-finetuned-squad",
    "hparams":  {
      "finetune_last_n_layers": 3,
      "task_model_kwargs": {
        "adam_epsilon": 1e-08,
        "eval_batch_size": 32,
        "learning_rate": 5e-06,
        "optimizer": "AdamW",
        "train_batch_size": 32,
        "use_scheduler": false,
        "warmup_steps": 100,
        "weight_decay": 0.0001
      },
      "... (more packages)"

    }
    "packages": [
        "absl-py==1.4.0",
        "aiobotocore==2.5.0",
        "aiohttp-retry==2.8.3",
        "aiohttp==3.8.4",
        "aioitertools==0.11.0",
        "... (more packages)"
    ]
    "leaderboard_task_name": "Question Answering",
    "metrics": [
      {
        "HasAns_exact": 53.78787878787879,
        "HasAns_f1": 69.3131673937266,
        "NoAns_f1": 91.44496609285342,
        "exact": 70.10169491525424,
        "f1": 78.9011127284667
      }
    ],
    "metrics_avg": {
      "HasAns_exact": 53.78787878787879,
      "HasAns_f1": 69.3131673937266,
      "NoAns_f1": 91.44496609285342,
      "exact": 70.10169491525424,
      "f1": 78.9011127284667
    },
    "metrics_std": {
      "f1": 0,
      "exact": 0,
      "HasAns_f1": 0,
      "HasAns_exact": 0,
      "NoAns_f1": 0
    },
    "averaged_over": 1
}

{{< /collapse >}}



## 1B. Generation Submission using Embeddings library


- Install `embeddings` package
    
    ```bash
    pip install clarinpl-embeddings
    ```
    
- Put your data in accordance with comments
    
    ```python
    import datasets
    import numpy as np
    
    from embeddings.evaluator.evaluation_results import Predictions
    from embeddings.evaluator.leaderboard import get_dataset_task
    from embeddings.evaluator.submission import AveragedSubmission
    from embeddings.utils.utils import get_installed_packages
    
    DATASET_NAME = "clarin-pl/polemo2-official"
    TARGET_COLUMN_NAME = "target"
    
    hparams = {"hparam_name_1": 0.2, "hparam_name_2": 0.1}  # put your hyperparameters here!
    
    dataset = datasets.load_dataset(DATASET_NAME)
    y_true = np.array(dataset["test"][TARGET_COLUMN_NAME])
    # put your predictions from multiple runs below!
    predictions = [
        Predictions(
            y_true=y_true, y_pred=np.random.randint(low=0, high=4, size=len(y_true))
        )
        for _ in range(5)
    ]
    
    # make sure you are running on a training env or put exported packages below!
    packages = get_installed_packages() 
    submission = AveragedSubmission.from_predictions(
        submission_name="your_submission_name",  # put your submission here!
        dataset_name=DATASET_NAME,
        dataset_version=dataset["train"].info.version.version_str,
        embedding_name="your_embedding_model",  # put your embedding name here!
        predictions=predictions,
        hparams=hparams,
        packages=packages,
        task=get_dataset_task(DATASET_NAME),
    )
    
    submission.save_json()
    ```

## 2. Submit via pull request

- clone repository
    
    ```bash
    git clone https://github.com/CLARIN-PL/embeddings.git
    cd embeddings
    ```
    
- checkout to new branch
    
    ```bash
    git checkout -b submission/[your_submission_name]
    ```
    
- move or copy submissions in json format to directory `webpage/data/results`
- commit and push
    
    ```bash
    git add .
    git commit -m "submit results"
    git push
    ```
    
- create pull request on [https://github.com/CLARIN-PL/embeddings/pulls](https://github.com/CLARIN-PL/embeddings/pulls)
- stay in touch with us in case of any problems