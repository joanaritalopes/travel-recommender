import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from travel_recsys.config import ARTIFACTS, DATA_PROCESSED

SECTIONS = ["intro", "understand", "see", "do", "buy", "eat", "drink"]
CHUNK_WORDS = 150
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def chunk(text, size=CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def build_chunks(df) -> pd.DataFrame:
    rows = []
    for city, record in enumerate(df.to_dict("records")):
        for section in SECTIONS:
            for piece in chunk(record[section]):
                rows.append(
                    {
                        "city": city,
                        "title": record["title"],
                        "section": section,
                        "text": piece
                    }
                )
    return pd.DataFrame(rows)


def embed_cities(df, model) -> np.ndarray:
    chunks = build_chunks(df)
    vectors = model.encode(
        chunks["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    city_vectors = np.zeros((len(df), vectors.shape[1]), dtype=np.float32)
    for city, city_chunks in chunks.groupby("city"):
        section_means = [
            vectors[rows.to_numpy()].mean(axis=0)
            for rows in city_chunks.groupby("section").groups.values()
        ]
        city_vectors[city] = np.mean(section_means, axis=0)

    return city_vectors / np.linalg.norm(city_vectors, axis=1, keepdims=True)


def main():
    df = pd.read_json(DATA_PROCESSED / "city_texts.jsonl", lines=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=pick_device())

    embeddings = embed_cities(df, model)
    np.save(ARTIFACTS / "city_embeddings.npy", embeddings)
    df[["geonameid", "title"]].to_csv(ARTIFACTS / "city_index.csv", index=False)
    print(f"embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
