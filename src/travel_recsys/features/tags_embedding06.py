import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from travel_recsys.config import ARTIFACTS
from travel_recsys.features.embeddings04 import MODEL_NAME, pick_device
from travel_recsys.features.tags05 import ACTIVITY_LABELS, ACTIVITY_TEMPLATE
