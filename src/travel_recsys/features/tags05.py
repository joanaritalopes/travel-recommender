import numpy as np
import pandas as pd
from transformers import pipeline

from travel_recsys.config import ARTIFACTS, DATA_PROCESSED
from travel_recsys.features.embeddings04 import chunk, pick_device

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

ACTIVITY_LABELS = [
    "beaches", "nightlife", "museums", "history", "architecture", "art",
    "hiking", "nature", "food", "shopping", "festivals", "relaxation",
]
ACTIVITY_SECTIONS = ["intro", "understand", "see", "do", "eat", "drink", "buy"]
ACTIVITY_TEMPLATE = "This place is good for {}."

BUDGET_LABELS = ["affordable", "expensive"]
BUDGET_SECTIONS = ["understand", "eat"]
BUDGET_TEMPLATE = "Visiting this place is {}."

MAX_CHUNKS_PER_SECTION = 2
TAG_THRESHOLD = 0.8
MAX_TAGS = 5


def sample_chunks(record, sections) -> list[str]:
    pieces = []
    for section in sections:
        pieces.extend(chunk(record[section])[:MAX_CHUNKS_PER_SECTION])
    print(pieces)
    return pieces


def city_scores(classifier, df, sections, labels, template, multi_label, agg) -> pd.DataFrame:
    rows = [
        (city, piece)
        for city, record in enumerate(df.to_dict("records"))
        for piece in sample_chunks(record, sections)
    ]
    cities = [city for city, _ in rows]
    texts = [piece for _, piece in rows]

    results = classifier(
        texts,
        candidate_labels=labels,
        hypothesis_template=template,
        multi_label=multi_label,
        batch_size=16,
    )
    scores = [[dict(zip(r["labels"], r["scores"]))[label] for label in labels] for r in results]

    table = pd.DataFrame(scores, columns=labels)
    table["city"] = cities
    print(table)
    return table.groupby("city").agg(agg).reindex(range(len(df)))


def top_tags(row) -> str:
    strong = row[row >= TAG_THRESHOLD].sort_values(ascending=False).head(MAX_TAGS)
    return ", ".join(strong.index)


def main():
    df = pd.read_json(DATA_PROCESSED / "city_texts.jsonl", lines=True)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=pick_device())

    activity = city_scores(
        classifier, df, ACTIVITY_SECTIONS, ACTIVITY_LABELS, ACTIVITY_TEMPLATE,
        multi_label=True, agg="max",
    )
    budget = city_scores(
        classifier, df, BUDGET_SECTIONS, BUDGET_LABELS, BUDGET_TEMPLATE,
        multi_label=False, agg="mean",
    )

    result = pd.concat([df[["geonameid", "title"]], activity, budget], axis=1)
    result["tags"] = activity.apply(top_tags, axis=1)
    result.to_csv(ARTIFACTS / "city_tags.csv", index=False)
    print(f"tagged {len(result)} cities")


if __name__ == "__main__":
    main()
