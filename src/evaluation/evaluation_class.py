import torch

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


def evaluate_class(embeddings, labels, seeds=(0, 1, 2)):
    latent_reps = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    perf_scores = []
    for seed in seeds:
        embeds_train, embeds_val, y_train, y_val = train_test_split(latent_reps, labels, test_size=0.2, random_state=seed, stratify=labels)
        sk_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, C=5, solver='newton-cg'))
        sk_model.fit(embeds_train, y_train)
        preds = sk_model.predict(embeds_val)
        perf = balanced_accuracy_score(y_val, preds)
        perf_scores.append(perf)

    return np.mean(perf_scores)
