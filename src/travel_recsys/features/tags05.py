import numpy as np
import pandas as pd
from transformers import pipeline

from travel_recsys.config import ARTIFACTS, DATA_PROCESSED
from travel_recsys.features.embeddings04 import chunk, pick_device

# small model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0" -> trained specifically for zero-shot classification on many NLI datasets; smaller, faster
# large model: "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" -> more accurate, but ~2 to 3 times slower
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

ACTIVITY_LABELS = [
    "beaches", "nightlife", "museums", "history", "architecture", "art",
    "hiking", "nature", "food", "shopping", "festivals", "relaxation",
]
ACTIVITY_SECTIONS = ["intro", "understand", "see", "do", "eat", "drink", "buy"]
ACTIVITY_TEMPLATE = "This place is good for {}."

BUDGET_LABELS = ["affordable", "expensive"]
BUDGET_SECTIONS = ["understand", "eat", "drink"]
BUDGET_TEMPLATE = "Visiting this place is {}."

# there is 7 sections: so 7*2=21; max=3 would help long articles but at much higher cost
MAX_CHUNKS_PER_SECTION = 2
TAG_THRESHOLD = 0.75
MAX_TAGS = 5

def output_path():
    return ARTIFACTS / f"city_tags_{MODEL_NAME.split('/')[-1]}.csv"


def sample_chunks(record, sections) -> list[str]:
    pieces = []
    for section in sections:
        pieces.extend(chunk(record[section])[:MAX_CHUNKS_PER_SECTION])

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

    return table.groupby("city").agg(agg).reindex(range(len(df)))


def top_tags(row) -> str:
    strong = row[row >= TAG_THRESHOLD].sort_values(ascending=False).head(MAX_TAGS)

    return ", ".join(strong.index)


def high_pairs(corr, threshold) -> list[tuple[str, str, float]]:
    '''To check correlations for pairs.'''
    pairs = []
    labels = list(corr.columns)
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            value = float(corr.loc[first, second])
            if value >= threshold:
                pairs.append((first, second, round(value, 2)))
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def main():
    df = pd.read_json(DATA_PROCESSED / "city_texts.jsonl", lines=True)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=pick_device())

    activity_scores = city_scores(
        classifier, df, ACTIVITY_SECTIONS, ACTIVITY_LABELS, ACTIVITY_TEMPLATE,
        multi_label=True, agg="max",
    )
    budget_scores = city_scores(
        classifier, df, BUDGET_SECTIONS, BUDGET_LABELS, BUDGET_TEMPLATE,
        multi_label=False, agg="mean",
    )

    corr = activity_scores.corr(method="spearman")
    
    print(corr.round(2).to_string())
    print(f"\npairs with correlation >= {HIGH_CORRELATION}:")
    for first, second, value in high_pairs(corr, HIGH_CORRELATION):
        print(f"  {first} / {second}: {value}")

    result = pd.concat([df[["geonameid", "title"]], activity_scores, budget_scores], axis=1)
    result["tags"] = activity_scores.apply(top_tags, axis=1)
    result.to_csv(output_path(), index=False)
    print(f"tagged {len(result)} cities, saved to {output_path()}")


if __name__ == "__main__":
    main()
