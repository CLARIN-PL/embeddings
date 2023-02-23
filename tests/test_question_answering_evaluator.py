import random
from pathlib import Path
from typing import Any, Dict

import datasets
import numpy as np
import pytest
import torch
from _pytest.tmpdir import TempdirFactory
from tqdm import tqdm

from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.question_answering_evaluator import QuestionAnsweringEvaluator
from embeddings.task.lightning_task.question_answering import QuestionAnsweringTask

torch.manual_seed(42)


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_dict() -> datasets.DatasetDict:
    train_dict = {
        "id": [
            "57262779271a42140099d5f1",
            "572fb38ea23a5019007fc8cb",
            "572953656aef051400154cf2",
            "5727951d708984140094e183",
            "56df787656340a1900b29bf8",
            "56ce0d8662d2951400fa69e8",
            "5a53e3bbbdaabd001a3867c0",
            "5706b9072eaba6190074ac7e",
            "5731309ce6313a140071cce8",
            "56cf69144df3c31400b0d745",
        ],
        "title": [
            "East_India_Company",
            "Transistor",
            "Bermuda",
            "Nonprofit_organization",
            "United_Nations_Population_Fund",
            "Frédéric_Chopin",
            "Armenians",
            "House_music",
            "Kievan_Rus%27",
            "Frédéric_Chopin",
        ],
        "context": [
            "The company, which benefited from the imperial patronage, soon expanded its commercial trading "
            "operations, eclipsing the Portuguese Estado da Índia, which had established bases in Goa, Chittagong, "
            "and Bombay, which Portugal later ceded to England as part of the dowry of Catherine de Braganza. The "
            "East India Company also launched a joint attack with the Dutch United East India Company on Portuguese "
            "and Spanish ships off the coast of China, which helped secure their ports in China. The company "
            "established trading posts in Surat (1619), Madras (1639), Bombay (1668), and Calcutta (1690). By 1647, "
            "the company had 23 factories, each under the command of a factor or master merchant and governor if so "
            "chosen, and 90 employees in India. The major factories became the walled forts of Fort William in "
            "Bengal, Fort St George in Madras, and Bombay Castle.",
            "Because the electron mobility is higher than the hole mobility for all semiconductor materials, "
            "a given bipolar n–p–n transistor tends to be swifter than an equivalent p–n–p transistor. GaAs has the "
            "highest electron mobility of the three semiconductors. It is for this reason that GaAs is used in "
            "high-frequency applications. A relatively recent FET development, the high-electron-mobility transistor "
            "(HEMT), has a heterostructure (junction between different semiconductor materials) of aluminium gallium "
            "arsenide (AlGaAs)-gallium arsenide (GaAs) which has twice the electron mobility of a GaAs-metal barrier "
            "junction. Because of their high speed and low noise, HEMTs are used in satellite receivers working at "
            "frequencies around 12 GHz. HEMTs based on gallium nitride and aluminium gallium nitride (AlGaN/GaN "
            "HEMTs) provide a still higher electron mobility and are being developed for various applications.",
            'Once known as "the Gibraltar of the West" and "Fortress Bermuda", Bermuda today is defended by forces of '
            "the British government. For the first two centuries of settlement, the most potent armed force operating "
            "from Bermuda was its merchant shipping fleet, which turned to privateering at every opportunity. The "
            "Bermuda government maintained a local militia. After the American Revolutionary War, Bermuda was "
            "established as the Western Atlantic headquarters of the Royal Navy. Once the Royal Navy established a "
            "base and dockyard defended by regular soldiers, however, the militias were disbanded following the War "
            "of 1812. At the end of the 19th century, the colony raised volunteer units to form a reserve for the "
            "military garrison.",
            "Some NPOs which are particularly well known, often for the charitable or social nature of their "
            "activities performed during a long period of time, include Amnesty International, Oxfam, "
            "Rotary International, Kiwanis International, Carnegie Corporation of New York, Nourishing USA, "
            "DEMIRA Deutsche Minenräumer (German Mine Clearers), FIDH International Federation for Human Rights, "
            "Goodwill Industries, United Way, ACORN (now defunct), Habitat for Humanity, Teach For America, "
            "the Red Cross and Red Crescent organizations, UNESCO, IEEE, INCOSE, World Wide Fund for Nature, "
            "Heifer International, Translators Without Borders and SOS Children's Villages.",
            "In America, nonprofit organizations like Friends of UNFPA (formerly Americans for UNFPA) worked to "
            "compensate for the loss of United States federal funding by raising private donations.",
            "At the age of 21 he settled in Paris. Thereafter, during the last 18 years of his life, he gave only "
            "some 30 public performances, preferring the more intimate atmosphere of the salon. He supported himself "
            "by selling his compositions and teaching piano, for which he was in high demand. Chopin formed a "
            "friendship with Franz Liszt and was admired by many of his musical contemporaries, including Robert "
            "Schumann. In 1835 he obtained French citizenship. After a failed engagement to Maria Wodzińska, "
            "from 1837 to 1847 he maintained an often troubled relationship with the French writer George Sand. A "
            "brief and unhappy visit to Majorca with Sand in 1838–39 was one of his most productive periods of "
            "composition. In his last years, he was financially supported by his admirer Jane Stirling, "
            "who also arranged for him to visit Scotland in 1848. Through most of his life, Chopin suffered from poor "
            "health. He died in Paris in 1849, probably of tuberculosis.",
            "Armenian literature dates back to 400 AD, when Mesrop Mashtots first invented the Armenian alphabet. "
            "This period of time is often viewed as the Golden Age of Armenian literature. Early Armenian literature "
            'was written by the "father of Armenian history", Moses of Chorene, who authored The History of Armenia. '
            "The book covers the time-frame from the formation of the Armenian people to the fifth century AD. The "
            "nineteenth century beheld a great literary movement that was to give rise to modern Armenian literature. "
            "This period of time, during which Armenian culture flourished, is known as the Revival period (Zartonki "
            "sherchan). The Revivalist authors of Constantinople and Tiflis, almost identical to the Romanticists of "
            "Europe, were interested in encouraging Armenian nationalism. Most of them adopted the newly created "
            "Eastern or Western variants of the Armenian language depending on the targeted audience, and preferred "
            "them over classical Armenian (grabar). This period ended after the Hamidian massacres, when Armenians "
            "experienced turbulent times. As Armenian history of the 1920s and of the Genocide came to be more openly "
            "discussed, writers like Paruyr Sevak, Gevork Emin, Silva Kaputikyan and Hovhannes Shiraz began a new era "
            "of literature.",
            "Towards the end of the 1990s and into the 2000s, producers such as Daft Punk, Stardust, Cassius, "
            "St. Germain and DJ Falcon began producing a new sound out of Paris's house scene. Together, "
            "they laid the groundwork for what would be known as the French house movement. By combining the "
            "harder-edged-yet-soulful philosophy of Chicago house with the melodies of obscure funk, state-of-the-art "
            "production techniques and the sound of analog synthesizers, they began to create the standards that "
            "would shape all house music.",
            "Vladimir's choice of Eastern Christianity may also have reflected his close personal ties with "
            "Constantinople, which dominated the Black Sea and hence trade on Kiev's most vital commercial route, "
            "the Dnieper River. Adherence to the Eastern Church had long-range political, cultural, and religious "
            "consequences. The church had a liturgy written in Cyrillic and a corpus of translations from Greek that "
            "had been produced for the Slavic peoples. This literature facilitated the conversion to Christianity of "
            "the Eastern Slavs and introduced them to rudimentary Greek philosophy, science, and historiography "
            "without the necessity of learning Greek (there were some merchants who did business with Greeks and "
            "likely had an understanding of contemporary business Greek). In contrast, educated people in medieval "
            "Western and Central Europe learned Latin. Enjoying independence from the Roman authority and free from "
            "tenets of Latin learning, the East Slavs developed their own literature and fine arts, quite distinct "
            "from those of other Eastern Orthodox countries.[citation needed] (See Old East Slavic language and "
            "Architecture of Kievan Rus for details ). Following the Great Schism of 1054, the Rus' church maintained "
            "communion with both Rome and Constantinople for some time, but along with most of the Eastern churches "
            "it eventually split to follow the Eastern Orthodox. That being said, unlike other parts of the Greek "
            "world, Kievan Rus' did not have a strong hostility to the Western world.",
            "In September 1828 Chopin, while still a student, visited Berlin with a family friend, zoologist Feliks "
            "Jarocki, enjoying operas directed by Gaspare Spontini and attending concerts by Carl Friedrich Zelter, "
            "Felix Mendelssohn and other celebrities. On an 1829 return trip to Berlin, he was a guest of Prince "
            "Antoni Radziwiłł, governor of the Grand Duchy of Posen—himself an accomplished composer and aspiring "
            "cellist. For the prince and his pianist daughter Wanda, he composed his Introduction and Polonaise "
            "brillante in C major for cello and piano, Op. 3.",
        ],
        "question": [
            "what were the walled forts of Fort William in Bengal, Fort St George in Madras and Bombay castle before "
            "they were forts?",
            "What are common applications of HEMT?",
            "Due to the British goverment's defense forces, what are two nicknames for bermuda?",
            "What is a well known NPO that helps people from low incomes become homeowners?",
            "What is one country in which nonprofit organizations try to make up for the loss of United States "
            "funding for the UNFPA?",
            "Where did he end up living when he was 21?",
            "What was Zartonki Sherchan interested in openly discussing?",
            "Daft Punk began producing a new sound out of what european city?",
            "What was considered too be Kiev's most important route for trade?",
            "When did Chopin visit Berlin?",
        ],
        "answers": [
            {"text": ["major factories"], "answer_start": [742]},
            {"text": ["satellite receivers"], "answer_start": [680]},
            {"text": ['the Gibraltar of the West" and "Fortress Bermuda"'], "answer_start": [15]},
            {"text": ["Habitat for Humanity"], "answer_start": [434]},
            {"text": ["America"], "answer_start": [3]},
            {"text": ["Paris"], "answer_start": [31]},
            {"text": [], "answer_start": []},
            {"text": ["Paris"], "answer_start": [158]},
            {"text": ["Dnieper River"], "answer_start": [200]},
            {"text": ["September 1828"], "answer_start": [3]},
        ],
    }
    validation_dict = train_dict.copy()
    train_dataset = datasets.Dataset.from_dict(train_dict)
    validation_dataset = datasets.Dataset.from_dict(validation_dict)
    return datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )


@pytest.fixture(scope="module")
def question_answering_data_module(
    tmp_path_module: Path, dataset_dict: datasets.DatasetDict
) -> QuestionAnsweringDataModule:
    dataset = dataset_dict
    dataset.save_to_disk(tmp_path_module / "data_sample")
    return QuestionAnsweringDataModule(
        dataset_name_or_path=tmp_path_module / "data_sample",
        tokenizer_name_or_path="hf-internal-testing/tiny-albert",
        train_batch_size=5,
        eval_batch_size=5,
        max_seq_length=256,
        doc_stride=128,
        batch_encoding_kwargs={
            "padding": "max_length",
            "truncation": "only_second",
            "return_offsets_mapping": True,
            "return_overflowing_tokens": True,
        },
    )


@pytest.fixture(scope="module")
def question_answering_task() -> QuestionAnsweringTask:
    return QuestionAnsweringTask(
        model_name_or_path="hf-internal-testing/tiny-albert",
        finetune_last_n_layers=-1,
        task_model_kwargs={
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.001,
            "use_scheduler": False,
            "warmup_steps": None,
            "train_batch_size": 5,
            "eval_batch_size": 5,
        },
        output_path=".",
        model_config_kwargs={},
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "auto",
            "deterministic": True,
        },
        early_stopping_kwargs={"monitor": "val/Loss", "patience": 1, "mode": "min"},
        model_checkpoint_kwargs={},
    )


@pytest.fixture(scope="module")
def scores(
    question_answering_data_module: QuestionAnsweringDataModule,
    question_answering_task: QuestionAnsweringTask,
) -> Dict[str, Any]:
    datamodule = question_answering_data_module
    task = question_answering_task
    datamodule.setup(stage="fit")
    task.build_task_model()
    task.fit(datamodule)
    datamodule.process_data(stage="test")
    predict_dataloader = datamodule.predict_dataloader()

    scores = {}

    for split_id, split in tqdm(enumerate(datamodule.splits)):
        dataloader = predict_dataloader[split_id]
        model_outputs = task.predict(dataloader)
        scores[split] = {
            "examples": datamodule.dataset_raw[split].to_pandas(),
            "outputs": model_outputs,
            "overflow_to_sample_mapping": datamodule.overflow_to_sample_mapping[split],
            "offset_mapping": datamodule.offset_mapping[split],
        }
    return scores


def test_question_answering_evaluator(scores: Dict[str, Any]):
    evaluator = QuestionAnsweringEvaluator()
    metrics, outputs = evaluator.evaluate(scores)
    validation_metrics = metrics["validation"]
    validation_outputs = outputs["validation"]
    sample_output = random.choice(validation_outputs)

    np.testing.assert_almost_equal(validation_metrics["exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["f1"], 3.8394892, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["total"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["HasAns_exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        validation_metrics["HasAns_f1"], 4.2660992, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(validation_metrics["HasAns_total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_f1"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_total"], 1.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["best_exact"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        validation_metrics["best_exact_thresh"], 0.0, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        validation_metrics["best_f1"], 13.5294117, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        validation_metrics["best_f1_thresh"], 0.0, decimal=pytest.decimal
    )

    assert "context" in sample_output.keys()
    assert "questions" in sample_output.keys()
    assert "answers" in sample_output.keys()
    assert "predicted_answer" in sample_output.keys()
    assert isinstance(sample_output["context"], str)
    assert isinstance(sample_output["questions"], str)
    assert sample_output["answers"] is None or isinstance(sample_output["answers"], dict)
    assert sample_output["predicted_answer"] is None or isinstance(
        sample_output["predicted_answer"], dict
    )
