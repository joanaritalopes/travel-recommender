import json
import re

import pandas as pd

from travel_recsys.config import DATA_PROCESSED, DATA_RAW

KEEP_SECTIONS = ["Intro", "Understand", "See", "Do", "Buy", "Eat", "Drink"]
BUDGET_ONLY_SECTIONS = ["Sleep"]
HEADING = re.compile(r"^(=+)\s*(.+?)\s*=+\s*$")
MIN_WORDS = 300


def split_sections(text) -> dict[str, str]:
    sections = {}
    current = "Intro"
    lines = []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match and len(match.group(1)) == 2:
            sections[current] = "\n".join(lines)
            current = match.group(2)
            lines = []
        elif match:
            lines.append(match.group(2))
        else:
            lines.append(line)
    sections[current] = "\n".join(lines)
    return sections


def normalize(text) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_record(page) -> dict:
    sections = split_sections(page["text"])
    record = {"geonameid": page["geonameid"], "title": page["title"]}
    for name in KEEP_SECTIONS:
        record[name.lower()] = normalize(sections.get(name, ""))
    for name in BUDGET_ONLY_SECTIONS:
        record[name.lower()] = normalize(sections.get(name, ""))
    record["text"] = " ".join(record[name.lower()] for name in KEEP_SECTIONS if record[name.lower()])
    record["n_words"] = len(record["text"].split())
    return record


def main():
    with open(DATA_RAW / "wikivoyage_pages.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    df = pd.DataFrame([build_record(page) for page in pages])
    short = df["n_words"] < MIN_WORDS
    df = df.loc[~short].reset_index(drop=True)

    print(f"{len(df)} cities kept, {short.sum()} dropped for having under {MIN_WORDS} words")
    print(f"median length: {df['n_words'].median():.0f} words")
    print("empty sections:", {
        name: int((df[name.lower()] == "").sum())
        for name in KEEP_SECTIONS + BUDGET_ONLY_SECTIONS
    })

    out_path = DATA_PROCESSED / "city_texts.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
