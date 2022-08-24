from enum import Enum
from typing import Dict, Final, List


class LeaderboardDataset(str, Enum):
    abusive_clauses = "Abusive Clauses"
    aspectemo = "AspectEmo"
    cdsc_e = "CDSC-E"
    dialogue_acts = "Dialogue Acts"
    dyk = "DYK"
    kpwr_ner = "KPWr NER"
    nkjp_pos = "NKJP POS"
    polemo2 = "PolEmo 2.0"
    polemo2_in = "PolEmo 2.0 (In-domain)"
    polemo2_out = "PolEmo 2.0 (Out-domain)"
    political_advertising = "Political Advertising"
    psc = "PSC"
    punctuation_restoration = "Punctuation Restoration"


class LeaderboardTask(str, Enum):
    abusive_clauses_detection = "Abusive Clauses Detection"
    aspect_based_sentiment_analysis = "Aspect-based Sentiment Analysis"
    dialogue_acts_classification = "Dialogue Acts Classification"
    entailment_classification = "Entailment Classification"
    ner = "Named Entity Recognition"
    paraphrase_classification = "Paraphrase Classification"
    political_advertising_detection = "Political Advertising Detection"
    pos = "Part-of-speech Tagging"
    punctuation_restoration = "Punctuation Restoration"
    qa_classification = "Q&A Classification"
    sentiment_analysis = "Sentiment Analysis"


class LeaderboardDomain(str, Enum):
    image_captions = "image captions"
    legal_texts = "legal texts"
    misc = "misc."
    news = "news"
    online_reviews = "online reviews"
    social_media = "social media"
    wikinews = "Wikinews"
    wikipedia = "Wikipedia"
    wikipedia_talk = "Wikipedia Talk"


HUGGINGFACE_DATASET_LEADERBOARD_DATASET_MAPPING: Final[Dict[str, str]] = {
    "laugustyniak/abusive-clauses-pl": "abusive_clauses",
    "clarin-pl/aspectemo": "aspectemo",
    "allegro/klej-cdsc-e": "cdsc_e",
    "allegro/klej-dyk": "dyk",
    "clarin-pl/dialogue-acts": "dialogue_acts",
    "clarin-pl/kpwr-ner": "kpwr_ner",
    "clarin-pl/nkjp-pos": "nkjp_pos",
    "clarin-pl/polemo2-official": "polemo2",
    "allegro/klej-polemo2-in": "polemo2_in",
    "allegro/klej-polemo2-out": "polemo2_out",
    "laugustyniak/political-advertising-pl": "political_advertising",
    "allegro/klej-psc": "psc",
    "clarin-pl/2021-punctuation-restoration": "punctuation_restoration",
}

LEADERBOARD_DATASET_TASK_MAPPING: Final[Dict[str, LeaderboardTask]] = {
    "abusive_clauses": LeaderboardTask.abusive_clauses_detection,
    "aspectemo": LeaderboardTask.aspect_based_sentiment_analysis,
    "cdsc_e": LeaderboardTask.entailment_classification,
    "dialogue_acts": LeaderboardTask.dialogue_acts_classification,
    "dyk": LeaderboardTask.qa_classification,
    "kpwr_ner": LeaderboardTask.ner,
    "nkjp_pos": LeaderboardTask.pos,
    "polemo2": LeaderboardTask.sentiment_analysis,
    "polemo2_in": LeaderboardTask.sentiment_analysis,
    "polemo2_out": LeaderboardTask.sentiment_analysis,
    "political_advertising": LeaderboardTask.political_advertising_detection,
    "psc": LeaderboardTask.paraphrase_classification,
    "punctuation_restoration": LeaderboardTask.punctuation_restoration,
}

LEADERBOARD_DATASET_DOMAIN_MAPPING: Final[Dict[str, List[LeaderboardDomain]]] = {
    "abusive_clauses": [LeaderboardDomain.legal_texts],
    "aspectemo": [LeaderboardDomain.online_reviews],
    "cdsc_e": [LeaderboardDomain.image_captions],
    "dyk": [LeaderboardDomain.wikipedia],
    "kpwr_ner": [LeaderboardDomain.misc],
    "nkjp_pos": [LeaderboardDomain.misc],
    "polemo2": [LeaderboardDomain.online_reviews],
    "polemo2_in": [LeaderboardDomain.online_reviews],
    "polemo2_out": [LeaderboardDomain.online_reviews],
    "political_advertising": [LeaderboardDomain.social_media],
    "psc": [LeaderboardDomain.news],
    "punctuation_restoration": [LeaderboardDomain.wikipedia_talk, LeaderboardDomain.wikinews],
}

DATASET_TASK_MAPPING: Dict[str, str] = {
    "abusive_clauses": "text_classification",
    "aspectemo": "sequence_labeling",
    "cdsc_e": "text_classification",
    "dyk": "text_classification",
    "kpwr_ner": "sequence_labeling",
    "polemo2": "text_classification",
    "polemo2_in": "text_classification",
    "polemo2_out": "text_classification",
    "political_advertising": "sequence_labeling",
    "psc": "text_classification",
    "punctuation_restoration": "sequence_labeling",
    "nkjp_pos": "sequence_labeling",
}


def get_dataset_task(dataset_name: str) -> str:
    return DATASET_TASK_MAPPING[HUGGINGFACE_DATASET_LEADERBOARD_DATASET_MAPPING[dataset_name]]
