import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from travel_recsys.config import ARTIFACTS
from travel_recsys.features.embeddings04 import MODEL_NAME, pick_device
from travel_recsys.features.tags05 import ACTIVITY_LABELS, ACTIVITY_TEMPLATE


city_embeddings = np.load(ARTIFACTS / "city_embeddings.npy")
model = SentenceTransformer(MODEL_NAME, device=pick_device())
print("city embeddings:", city_embeddings.shape)

prompts = [ACTIVITY_TEMPLATE.format(label) for label in ACTIVITY_LABELS]
print("prompts:", prompts)

label_vectors = model.encode(prompts, normalize_embeddings=True)
print("label vectors:", label_vectors.shape)
print("lengths:", np.linalg.norm(label_vectors, axis=1).round(3))

similarity = city_embeddings @ label_vectors.T
print("similarity:", similarity.shape, "range:", similarity.min().round(3), "to", similarity.max().round(3))

scores = pd.DataFrame(similarity, columns=ACTIVITY_LABELS)
index = pd.read_csv(ARTIFACTS / "city_index.csv", keep_default_na=False)
scores.insert(0, "title", index["title"])
print(scores.nlargest(10, "beaches")[["title", "beaches"]].round(3).to_string())
print(scores.nlargest(10, "museums")[["title", "museums"]].round(3).to_string())