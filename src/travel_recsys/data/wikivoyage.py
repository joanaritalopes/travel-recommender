import json
import time

import pandas as pd
import requests

from travel_recsys.config import DATA_INTERIM, DATA_RAW

API_URL = "https://en.wikivoyage.org/w/api.php"
HEADERS = {"User-Agent": "travel-recsys/0.1 (https://github.com/joanaritalopes; student project)"}
BATCH_SIZE = 50
N_CITIES = 3680
PAUSE = 0.2


def query(params) -> dict:
    response = requests.get(
        API_URL,
        params={"action": "query", "format": "json", "formatversion": 2, **params},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["query"]


def resolve_titles(names) -> list[dict]:
    result = query({
        "titles": "|".join(names),
        "prop": "info|pageprops",
        "ppprop": "disambiguation",
        "redirects": 1,
    })
    normalized = {n["from"]: n["to"] for n in result.get("normalized", [])}
    redirects = {r["from"]: r["to"] for r in result.get("redirects", [])}
    pages = {p["title"]: p for p in result["pages"] if "title" in p}

    rows = []
    for name in names:
        title = normalized.get(name, name)
        title = redirects.get(title, title)
        page = pages.get(title)
        if page is None or page.get("missing") or page.get("invalid"):
            continue
        if "disambiguation" in page.get("pageprops", {}):
            continue
        rows.append({"name": name, "title": title, "page_length": page.get("length", 0)})
    return rows


def match_pages(cities) -> pd.DataFrame:
    names = cities["name"].unique().tolist()
    rows = []
    for i in range(0, len(names), BATCH_SIZE):
        rows.extend(resolve_titles(names[i:i + BATCH_SIZE]))
        time.sleep(PAUSE)

    return (
        cities.merge(pd.DataFrame(rows), on="name")
        .sort_values("population", ascending=False)
        .drop_duplicates("title")
        .nlargest(N_CITIES, "page_length")
        .reset_index(drop=True)
    )


def fetch_text(title) -> str:
    result = query({"titles": title, "prop": "extracts", "explaintext": 1})
    return result["pages"][0].get("extract", "")


def main():
    cities = pd.read_csv(DATA_INTERIM / "selected_cities.csv", keep_default_na=False)
    matched = match_pages(cities)
    matched.to_csv(DATA_INTERIM / "matched_cities.csv", index=False)
    print(f"{len(matched)} cities matched across {matched['country_code'].nunique()} countries")

    out_path = DATA_RAW / "wikivoyage_pages.jsonl"
    geonameids = matched["geonameid"].tolist()
    titles = matched["title"].tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        for geonameid, title in zip(geonameids, titles):
            record = {"geonameid": geonameid, "title": title, "text": fetch_text(title)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(PAUSE)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
