---
title: Submission
type: docs
url: "/submission"
geekdocNav: false
geekdocBreadcrumb: false
---

{{< pageHeader >}}

# Submit

We would like to encourage community to contribute to LEPISZCZE by submitting results of models. 

## 1. Predict on a test subset

Fine-tune your model and predict labels of test subset. Preferably you should repeat that a few times to get avaraged metrics. You should use at least one dataset from the table below.

```
| dataset_name                           | input                            | target              |
|----------------------------------------|----------------------------------|---------------------|
| allegro/klej-cdsc-e                    | ['sentence_A', 'sentence_B']     | entailment_judgment |
| allegro/klej-dyk                       | ['question', 'answer']           | target              |
| allegro/klej-polemo2-in                | sentence                         | target              |
| allegro/klej-polemo2-out               | sentence                         | target              |
| allegro/klej-psc                       | ['extract_text', 'summary_text'] | label               |
| clarin-pl/2021-punctuation-restoration | tokens                           | tags                |
| clarin-pl/aspectemo                    | tokens                           | labels              |
| clarin-pl/kpwr-ner                     | tokens                           | ner                 |
| clarin-pl/nkjp-pos                     | tokens                           | pos_tags            |
| clarin-pl/polemo2-official             | text                             | target              |
| laugustyniak/abusive-clauses-pl        | text                             | class               |
| laugustyniak/political-advertising-pl  | tokens                           | tags                |
```

## 2. Get list of your packages with versions

Prepare list of your installed packages in a training environment. It should look like `["torch==1.11.0", ...]`. You can also do it using the below script.

## 3. Generate submission file

Is possible to do it in two ways.

### Method 1: Use our library (recommended)

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
    

### Method 2: Manually

- Fill out the submission file according to the following scheme
    - 
        
        ```json
        {
          "submission_name": [str],
          "dataset_name": [str],
          "dataset_version": [str],
          "embedding_name": [str],
          "hparams": [dict[str, Any]],
          "packages": [list[str]],
          "config": [dict[str, Any] or null], # any additional config if you need
          "leaderboard_task_name": [str], # see table below
          "metrics": [
            {
              "accuracy": [float],
              "f1_macro": [float],
              "f1_micro": [float],
              "f1_weighted": [float],
              "recall_macro": [float],
              "recall_micro": [float],
              "recall_weighted": [float],
              "precision_macro": [float],
              "precision_micro": [float],
              "precision_weighted": [float],
              "classes": {
                "0": {
                  "precision": [float],
                  "recall": [float],
                  "f1": [float],
                  "support": [int]
                },
                "1": {
                  "precision": [float],
                  "recall": [float],
                  "f1": [float],
                  "support": [int]
                },
                "2": {
                  "precision": [float],
                  "recall": [float],
                  "f1": [float],
                  "support": [int]
                },
                "3": {
                  "precision": [float],
                  "recall": [float],
                  "f1": [float],
                  "support": [int]
                }
              } 
            }
          ],
          "metrics_avg": {
            "accuracy": [float],
            "f1_macro": [float],
            "f1_micro": [float],
            "f1_weighted": [float],
            "recall_macro": [float],
            "recall_micro": [float],
            "recall_weighted": [float],
            "precision_macro": [float],
            "precision_micro": [float],
            "precision_weighted": [float],
            "classes": {
              "0": {
                "precision": [float],
                "recall": [float],
                "f1": [float],
                "support": [int]
              },
              "1": {
                "precision": [float],
                "recall": [float],
                "f1": [float],
                "support": [int]
              }
            }
          },
          "metrics_median": {
            "accuracy": [float],
            "f1_macro": [float],
            "f1_micro": [float],
            "f1_weighted": [float],
            "recall_macro": [float],
            "recall_micro": [float],
            "recall_weighted": [float],
            "precision_macro": [float],
            "precision_micro": [float],
            "precision_weighted": [float],
            "classes": {
              "0": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              },
              "1": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              }
            }
          },
          "metrics_std": {
            "accuracy": [float],
            "f1_macro": [float],
            "f1_micro": [float],
            "f1_weighted": [float],
            "recall_macro": [float],
            "recall_micro": [float],
            "recall_weighted": [float],
            "precision_macro": [float],
            "precision_micro": [float],
            "precision_weighted": [float],
            "classes": {
              "0": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              },
              "1": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              },
              "2": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              },
              "3": {
                "precision": [float],
                "recall": [float],
                "f1": [float]
              }
            }
          },
          "averaged_over": [int]
        }
        ```
        
    
    Leaderboard task mapping:
    
    - 
        
        ```
        | dataset_name                           | leaderboard_task_name           |
        |:---------------------------------------|:--------------------------------|
        | allegro/klej-cdsc-e                    | Entailment Classification       |
        | allegro/klej-dyk                       | Q&A Classification              |
        | allegro/klej-polemo2-in                | Sentiment Analysis              |
        | allegro/klej-polemo2-out               | Sentiment Analysis              |
        | allegro/klej-psc                       | Paraphrase Classification       |
        | clarin-pl/2021-punctuation-restoration | Punctuation Restoration         |
        | clarin-pl/aspectemo                    | Aspect-based Sentiment Analysis |
        | clarin-pl/kpwr-ner                     | Named Entity Recognition        |
        | clarin-pl/nkjp-pos                     | Part-of-speech Tagging          |
        | clarin-pl/polemo2-official             | Sentiment Analysis              |
        | laugustyniak/abusive-clauses-pl        | Abusive Clauses Detection       |
        | laugustyniak/political-advertising-pl  | Political Advertising Detection |
        ```
        
    
    Example submission:
    
    - 
        
        ```json
        {
          "submission_name": "your_submission_name",
          "dataset_name": "clarin-pl/polemo2-official",
          "dataset_version": "0.0.0",
          "embedding_name": "your_embedding_model",
          "hparams": {
            "hparam_name_1": 0.2,
            "hparam_name_2": 0.1
          },
          "packages": [
            "absl-py==1.0.0",
            "aiohttp==3.8.1",
            "aiosignal==1.2.0",
            "alembic==1.7.7",
            "annoy==1.17.0",
            "appdirs==1.4.4",
            "async-timeout==4.0.2",
            "attrs==21.4.0",
            "autopage==0.5.0",
            "black==21.12b0",
            "bpemb==0.3.3",
            "cachetools==5.0.0",
            "catalogue==2.0.7",
            "certifi==2021.10.8",
            "charset-normalizer==2.0.12",
            "clarinpl-embeddings==0.0.1",
            "click==8.0.4",
            "cliff==3.10.1",
            "cmaes==0.8.2",
            "cmd2==2.4.0",
            "colorlog==6.6.0",
            "conllu==4.4.1",
            "coverage==6.2",
            "cycler==0.11.0",
            "datasets==2.0.0",
            "deprecated==1.2.13",
            "dill==0.3.4",
            "docker-pycreds==0.4.0",
            "fasteners==0.17.3",
            "filelock==3.6.0",
            "flair==0.10",
            "fonttools==4.31.2",
            "frozenlist==1.3.0",
            "fsspec==2022.3.0",
            "ftfy==6.1.1",
            "future==0.18.2",
            "gdown==3.12.2",
            "gensim==4.1.2",
            "gitdb==4.0.9",
            "gitpython==3.1.27",
            "google-auth-oauthlib==0.4.6",
            "google-auth==2.6.2",
            "greenlet==1.1.2",
            "grpcio==1.45.0",
            "h5py==3.6.0",
            "huggingface-hub==0.4.0",
            "idna==3.3",
            "importlib-metadata==3.10.1",
            "iniconfig==1.1.1",
            "isort==5.10.1",
            "janome==0.4.2",
            "joblib==1.1.0",
            "kiwisolver==1.4.2",
            "konoha==4.6.5",
            "langdetect==1.0.9",
            "lxml==4.8.0",
            "lz4==4.0.0",
            "mako==1.2.0",
            "markdown==3.3.5",
            "markupsafe==2.1.1",
            "matplotlib==3.5.1",
            "more-itertools==8.8.0",
            "mpld3==0.3",
            "multidict==6.0.2",
            "multiprocess==0.70.12.2",
            "mypy-extensions==0.4.3",
            "mypy==0.931",
            "numpy==1.22.3",
            "oauthlib==3.2.0",
            "optuna==2.10.0",
            "overrides==3.1.0",
            "packaging==21.3",
            "pandas==1.4.2",
            "pastel==0.2.1",
            "pathspec==0.9.0",
            "pathtools==0.1.2",
            "pbr==5.8.1",
            "pillow==9.1.0",
            "pip==22.0.3",
            "platformdirs==2.5.1",
            "pluggy==1.0.0",
            "poethepoet==0.11.0",
            "prettytable==3.2.0",
            "promise==2.3",
            "protobuf==3.20.0",
            "psutil==5.9.0",
            "py==1.11.0",
            "pyarrow==7.0.0",
            "pyasn1-modules==0.2.8",
            "pyasn1==0.4.8",
            "pydantic==1.9.0",
            "pydeprecate==0.3.1",
            "pyflakes==2.4.0",
            "pymagnitude==0.1.143",
            "pyparsing==3.0.7",
            "pyperclip==1.8.2",
            "pysocks==1.7.1",
            "pytest-env==0.6.2",
            "pytest==6.2.5",
            "python-dateutil==2.8.2",
            "pytorch-lightning==1.5.4",
            "pytz==2022.1",
            "pyyaml==6.0",
            "regex==2022.3.15",
            "requests-oauthlib==1.3.1",
            "requests==2.27.1",
            "responses==0.18.0",
            "rsa==4.8",
            "sacremoses==0.0.49",
            "scikit-learn==1.0.2",
            "scipy==1.6.1",
            "segtok==1.5.11",
            "sentencepiece==0.1.95",
            "sentry-sdk==1.5.8",
            "seqeval==1.2.2",
            "setproctitle==1.2.2",
            "setuptools-scm==6.4.2",
            "setuptools==60.9.3",
            "shortuuid==1.0.8",
            "six==1.16.0",
            "smart-open==5.2.1",
            "smmap==5.0.0",
            "sqlalchemy==1.4.34",
            "sqlitedict==2.0.0",
            "srsly==2.4.2",
            "stevedore==3.5.0",
            "tabulate==0.8.9",
            "tensorboard-data-server==0.6.1",
            "tensorboard-plugin-wit==1.8.1",
            "tensorboard==2.8.0",
            "termcolor==1.1.0",
            "threadpoolctl==3.1.0",
            "tokenizers==0.12.0",
            "toml==0.10.2",
            "tomli==1.2.3",
            "torch==1.11.0",
            "torchmetrics==0.7.3",
            "tqdm==4.64.0",
            "transformers==4.17.0",
            "typer==0.4.1",
            "types-pyyaml==6.0.5",
            "types-requests==2.26.1",
            "types-setuptools==57.4.12",
            "typing-extensions==4.1.1",
            "urllib3==1.26.9",
            "wandb==0.12.11",
            "wcwidth==0.2.5",
            "werkzeug==2.1.1",
            "wheel==0.37.1",
            "wikipedia-api==0.5.4",
            "wrapt==1.14.0",
            "xxhash==3.0.0",
            "yarl==1.7.2",
            "yaspin==2.1.0",
            "zipp==3.8.0"
          ],
          "config": null,
          "leaderboard_task_name": "Sentiment Analysis",
          "metrics": [
            {
              "accuracy": 0.25133120340788073,
              "f1_macro": 0.24305388727769806,
              "f1_micro": 0.25133120340788073,
              "f1_weighted": 0.2592122225975472,
              "recall_macro": 0.24923419165599014,
              "recall_micro": 0.25133120340788073,
              "recall_weighted": 0.25133120340788073,
              "precision_macro": 0.250706133452257,
              "precision_micro": 0.25133120340788073,
              "precision_weighted": 0.2814074697895528,
              "classes": {
                "0": {
                  "precision": 0.1401468788249694,
                  "recall": 0.23583934088568487,
                  "f1": 0.1758157389635317,
                  "support": 971
                },
                "1": {
                  "precision": 0.37796713329275716,
                  "recall": 0.2515188335358445,
                  "f1": 0.3020428015564202,
                  "support": 2469
                },
                "2": {
                  "precision": 0.2833432128037937,
                  "recall": 0.26206140350877194,
                  "f1": 0.272287097692965,
                  "support": 1824
                },
                "3": {
                  "precision": 0.20136730888750776,
                  "recall": 0.2475171886936593,
                  "f1": 0.22206991089787526,
                  "support": 1309
                }
              }
            }
          ],
          "metrics_avg": {
            "accuracy": 0.25148334094020997,
            "f1_macro": 0.24386686208637165,
            "f1_micro": 0.25148334094020997,
            "f1_weighted": 0.25917365911061924,
            "recall_macro": 0.25114832650274266,
            "recall_micro": 0.25148334094020997,
            "recall_weighted": 0.25148334094020997,
            "precision_macro": 0.25173539681216733,
            "precision_micro": 0.25148334094020997,
            "precision_weighted": 0.2823295048949504,
            "classes": {
              "0": {
                "precision": 0.1491199181324669,
                "recall": 0.2529351184346035,
                "f1": 0.18761415689081587,
                "support": 971
              },
              "1": {
                "precision": 0.38471810493455105,
                "recall": 0.25500202511138115,
                "f1": 0.3066556824977757,
                "support": 2469
              },
              "2": {
                "precision": 0.2753297642149423,
                "recall": 0.24791666666666667,
                "f1": 0.2608826500612311,
                "support": 1824
              },
              "3": {
                "precision": 0.19777379996670916,
                "recall": 0.24873949579831933,
                "f1": 0.2203149588956639,
                "support": 1309
              }
            }
          },
          "metrics_median": {
            "accuracy": 0.25133120340788073,
            "f1_macro": 0.24305388727769806,
            "f1_micro": 0.25133120340788073,
            "f1_weighted": 0.2592122225975472,
            "recall_macro": 0.24923419165599014,
            "recall_micro": 0.25133120340788073,
            "recall_weighted": 0.25133120340788073,
            "precision_macro": 0.2518172754574511,
            "precision_micro": 0.25133120340788073,
            "precision_weighted": 0.2830023617231186,
            "classes": {
              "0": {
                "precision": 0.15257352941176472,
                "recall": 0.25437693099897013,
                "f1": 0.19088098918083463
              },
              "1": {
                "precision": 0.38210399032648124,
                "recall": 0.2559740785743216,
                "f1": 0.3058089294287086
              },
              "2": {
                "precision": 0.2770100502512563,
                "recall": 0.24451754385964913,
                "f1": 0.2581967213114754
              },
              "3": {
                "precision": 0.20117994100294986,
                "recall": 0.2475171886936593,
                "f1": 0.22206991089787526
              }
            }
          },
          "metrics_std": {
            "accuracy": 0.0020636938941504023,
            "f1_macro": 0.002395043464221569,
            "f1_micro": 0.0020636938941504023,
            "f1_weighted": 0.001804131744309885,
            "recall_macro": 0.0031851498240062473,
            "recall_micro": 0.0020636938941504023,
            "recall_weighted": 0.0020636938941504023,
            "precision_macro": 0.002483426417726602,
            "precision_micro": 0.0020636938941504023,
            "precision_weighted": 0.003110579696499446,
            "classes": {
              "0": {
                "precision": 0.007267764926971178,
                "recall": 0.013847774512026038,
                "f1": 0.009440705319110149
              },
              "1": {
                "precision": 0.008947979588165176,
                "recall": 0.005151908826620602,
                "f1": 0.004827091289540694
              },
              "2": {
                "precision": 0.008777409909573826,
                "recall": 0.010412337512223268,
                "f1": 0.009398814618195977
              },
              "3": {
                "precision": 0.006275964317457789,
                "recall": 0.008095622179825186,
                "f1": 0.006391013712195844
              }
            }
          },
          "averaged_over": 5
        }
        ```
        
- save as `[your_submission_name].json`

## 4. Submit via pull request

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