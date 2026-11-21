import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from travel_recsys.config import ARTIFACTS
from travel_recsys.features.f04embeddings import MODEL_NAME, pick_device
from travel_recsys.features.f050tags import ACTIVITY_LABELS, ACTIVITY_TEMPLATE


def load_city_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """Load the city vectors and the index that maps each row to a city, checking they align."""
    embeddings = np.load(ARTIFACTS / "city_embeddings.npy")
    index = pd.read_csv(ARTIFACTS / "city_index.csv", keep_default_na=False)
    if len(embeddings) != len(index):
        raise ValueError(f"{len(embeddings)} embeddings but {len(index)} index rows")
    return embeddings, index


def build_prompts(labels, template) -> list[str]:
    """Turn each label into a sentence using the same template as the zero-shot tagger."""
    return [template.format(label) for label in labels]


def embed_labels(model, prompts) -> np.ndarray:
    """Embed the label sentences as unit-length vectors in the same space as the cities."""
    return model.encode(prompts, normalize_embeddings=True)


def similarity_matrix(city_embeddings, label_vectors, labels) -> pd.DataFrame:
    """Compute the cosine similarity between every city and every label."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = city_embeddings @ label_vectors.T
    if not np.isfinite(scores).all():
        raise ValueError("similarity scores contain non-finite values")
    return pd.DataFrame(scores, columns=labels)


def top_cities(scores, index, label, k=10) -> pd.DataFrame:
    """Return the k cities with the highest score for one label."""
    top = scores[label].nlargest(k).index
    return pd.DataFrame({"title": index.loc[top, "title"], label: scores.loc[top, label].round(3)})


def main():
    embeddings, index = load_city_embeddings()
    model = SentenceTransformer(MODEL_NAME, device=pick_device())
    label_vectors = embed_labels(model, build_prompts(ACTIVITY_LABELS, ACTIVITY_TEMPLATE))
    scores = similarity_matrix(embeddings, label_vectors, ACTIVITY_LABELS)

    out_path = ARTIFACTS / "city_tags_embedding.csv"
    pd.concat([index, scores], axis=1).to_csv(out_path, index=False)
    print(f"scored {len(scores)} cities, saved to {out_path}")


if __name__ == "__main__":
    main()