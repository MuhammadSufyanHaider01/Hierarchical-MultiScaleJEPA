import torch

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


COORD_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
COORD_MAX = np.array([80.0, 60.0, 40.0], dtype=np.float32)


def evaluate_pos(embeddings, labels):
    latent_reps = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    embeds_train, embeds_val, y_train, y_val = train_test_split(latent_reps, labels, test_size=0.2, random_state=0)
    sk_model = make_pipeline(StandardScaler(), LinearRegression())
    sk_model.fit(embeds_train, y_train)
    preds = sk_model.predict(embeds_val)
    perf = mean_pos_error(y_val, preds, COORD_MAX, COORD_MIN)

    return perf


def mean_pos_error(preds, labels, coord_max, coord_min):
    preds = _denorm(preds, coord_max, coord_min)
    labels = _denorm(labels, coord_max, coord_min)
    pos_errors = np.linalg.norm(preds - labels, axis=1)
    return pos_errors.mean()


def _denorm(x: np.ndarray, coord_max: np.ndarray, coord_min: np.ndarray) -> np.ndarray:
    return (x + 1) * 0.5 * (coord_max - coord_min) + coord_min
