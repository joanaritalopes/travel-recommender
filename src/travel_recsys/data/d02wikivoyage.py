import json
import time

import numpy as np
import pandas as pd
import requests

from travel_recsys.config import DATA_INTERIM, DATA_RAW

API_URL = "https://en.wikivoyage.org/w/api.php"
HEADERS = {"User-Agent": "travel-recsys/0.1 (https://github.com/joanaritalopes; student project)"}
BATCH_SIZE = 50
PAUSE = 0.2
MAX_DISTANCE_KM = 55
EARTH_RADIUS_KM = 6371


def query(params) -> dict:
    """Send one request to the MediaWiki API and return the 'query' part of the response."""
    response = requests.get(
        API_URL,
        params={"action": "query", "format": "json", "formatversion": 2, **params},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["query"]


def resolve_titles(names) -> list[dict]:
    """Look up a batch of names, following redirects, and return each page's title, length and coordinates."""
    result = query({
        "titles": "|".join(names),
        "prop": "info|pageprops|coordinates",
        "ppprop": "disambiguation",
        "colimit": "max",
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
        coords = page.get("coordinates", [])
        rows.append({
            "name": name,
            "title": title,
            "page_length": page.get("length", 0),
            "page_lat": coords[0]["lat"] if coords else None,
            "page_lon": coords[0]["lon"] if coords else None,
        })
    return rows


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points on the Earth's surface, given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def match_pages(cities) -> pd.DataFrame:
    """Match candidate cities to Wikivoyage pages and compute how far each page is from its city."""
    names = cities["name"].unique().tolist()
    rows = []
    for i in range(0, len(names), BATCH_SIZE):
        rows.extend(resolve_titles(names[i:i + BATCH_SIZE]))
        time.sleep(PAUSE)

    matched = cities.merge(pd.DataFrame(rows), on="name")
    matched["distance_km"] = haversine_km(
        matched["latitude"], matched["longitude"], matched["page_lat"], matched["page_lon"]
    )
    return matched


def verify_matches(matched) -> pd.DataFrame:
    """Keep matches within MAX_DISTANCE_KM, and the closest city when several share a page."""
    close = matched[matched["distance_km"] <= MAX_DISTANCE_KM]
    return close.sort_values("distance_km").drop_duplicates("title").reset_index(drop=True)


def fetch_text(title) -> str:
    """Download the plain-text content of one Wikivoyage page."""
    result = query({"titles": title, "prop": "extracts", "explaintext": 1})
    return result["pages"][0].get("extract", "")


def main():
    cities = pd.read_csv(DATA_INTERIM / "selected_cities.csv", keep_default_na=False)
    matched = match_pages(cities)
    verified = verify_matches(matched)

    no_coords = matched["page_lat"].isna().sum()
    too_far = (matched["distance_km"] > MAX_DISTANCE_KM).sum()
    print(f"{len(matched)} name matches: {no_coords} without coordinates, {too_far} over {MAX_DISTANCE_KM} km")
    print(f"{len(verified)} verified cities across {verified['country_code'].nunique()} countries")

    rejected = matched[matched["distance_km"] > MAX_DISTANCE_KM]
    print(rejected.nlargest(15, "page_length")[["name", "country_code", "title", "distance_km"]].round(0).to_string(index=False))

    verified.to_csv(DATA_INTERIM / "matched_cities.csv", index=False)

    out_path = DATA_RAW / "wikivoyage_pages.jsonl"
    geonameids = verified["geonameid"].tolist()
    titles = verified["title"].tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        for geonameid, title in zip(geonameids, titles):
            record = {"geonameid": geonameid, "title": title, "text": fetch_text(title)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(PAUSE)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
