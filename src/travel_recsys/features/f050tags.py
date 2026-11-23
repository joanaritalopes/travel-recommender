import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from travel_recsys.config import ARTIFACTS, DATA_PROCESSED
from travel_recsys.features.f04embeddings import chunk, pick_device

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
BUDGET_SECTIONS = ["understand", "eat", "drink", "sleep"]
BUDGET_TEMPLATE = "Visiting this place is {}."

# 7 activity sections x 2 chunks = at most 14 chunks per city; 3 would help long articles but at much higher cost
MAX_CHUNKS_PER_SECTION = 2
TAG_THRESHOLD = 0.75
MAX_TAGS = 5

BLOCK_SIZE = 50
CHECKPOINT_DIR = ARTIFACTS / "tag_checkpoints"


def model_short_name() -> str:
    """The part of the model name after the slash, used in file names."""
    return MODEL_NAME.split("/")[-1]


def output_path():
    """Where the final tag table for the current model is saved."""
    return ARTIFACTS / f"city_tags_{model_short_name()}.csv"


def sample_chunks(record, sections) -> list[str]:
    """Take the first MAX_CHUNKS_PER_SECTION chunks from each of the given sections."""
    pieces = []
    for section in sections:
        pieces.extend(chunk(record[section])[:MAX_CHUNKS_PER_SECTION])
    return pieces


def score_block(classifier, records, offset, sections, labels, template, multi_label, agg) -> pd.DataFrame:
    """Score one block of cities and return one row per city, indexed by its position in the full data."""
    rows = [
        (offset + i, piece)
        for i, record in enumerate(records)
        for piece in sample_chunks(record, sections)
    ]
    if not rows:
        return pd.DataFrame(columns=labels).rename_axis("city")

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
    return table.groupby("city").agg(agg)


def city_scores(classifier, df, sections, labels, template, multi_label, agg, task) -> pd.DataFrame:
    """Score all cities block by block, saving each block so an interrupted run can resume."""
    checkpoint_dir = CHECKPOINT_DIR / f"{model_short_name()}_{task}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = df.to_dict("records")

    blocks = []
    for start in tqdm(range(0, len(records), BLOCK_SIZE), desc=task, unit="block"):
        path = checkpoint_dir / f"block_{start:05d}.csv"
        if path.exists():
            blocks.append(pd.read_csv(path, index_col="city"))
            continue
        block = score_block(
            classifier, records[start:start + BLOCK_SIZE], start,
            sections, labels, template, multi_label, agg,
        )
        block.to_csv(path)
        blocks.append(block)

    return pd.concat(blocks).reindex(range(len(df)))


def top_tags(row) -> str:
    """Labels scoring at least TAG_THRESHOLD, strongest first, at most MAX_TAGS."""
    strong = row[row >= TAG_THRESHOLD].sort_values(ascending=False).head(MAX_TAGS)
    return ", ".join(strong.index)


def main():
    df = pd.read_json(DATA_PROCESSED / "city_texts.jsonl", lines=True)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=pick_device())
    print(f"running {model_short_name()} on {classifier.device} for {len(df)} cities")

    activity_scores = city_scores(
        classifier, df, ACTIVITY_SECTIONS, ACTIVITY_LABELS, ACTIVITY_TEMPLATE,
        multi_label=True, agg="max", task="activity",
    )
    budget_scores = city_scores(
        classifier, df, BUDGET_SECTIONS, BUDGET_LABELS, BUDGET_TEMPLATE,
        multi_label=False, agg="mean", task="budget",
    )

    result = pd.concat([df[["geonameid", "title"]], activity_scores, budget_scores], axis=1)
    result["tags"] = activity_scores.apply(top_tags, axis=1)
    result.to_csv(output_path(), index=False)
    print(f"tagged {len(result)} cities, saved to {output_path()}")


if __name__ == "__main__":
    main()
