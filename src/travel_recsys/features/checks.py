import pandas as pd
from transformers import pipeline

from travel_recsys.config import DATA_PROCESSED
from travel_recsys.features.embeddings04 import pick_device
from travel_recsys.features.tags05 import (
    ACTIVITY_LABELS,
    ACTIVITY_SECTIONS,
    ACTIVITY_TEMPLATE,
    MODEL_NAME,
    city_scores,
)

N_CITIES = 100
SEED = 42
HIGH_CORRELATION = 0.8


def high_pairs(corr, threshold) -> list[tuple[str, str, float]]:
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
    sample = df.sample(n=N_CITIES, random_state=SEED).reset_index(drop=True)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=pick_device())

    scores = city_scores(
        classifier, sample, ACTIVITY_SECTIONS, ACTIVITY_LABELS, ACTIVITY_TEMPLATE,
        multi_label=True, agg="max",
    )
    corr = scores.corr(method="spearman")

    print(corr.round(2).to_string())
    print(f"\npairs with correlation >= {HIGH_CORRELATION}:")
    for first, second, value in high_pairs(corr, HIGH_CORRELATION):
        print(f"  {first} / {second}: {value}")


if __name__ == "__main__":
    main()