"""Id-based KNN classifier.

The competition's sample ``Id`` correlates almost perfectly (Spearman ~0.999)
with the row order of the underlying source database (GEOROC / Lustrino et al.,
2022), and rows are ordered so that contiguous ``Id`` runs tend to share the
same age class. This module exploits that structure: it predicts each sample's
class from the labels of its nearest neighbours in ``Id`` space, weighted by
exponential distance decay.

It is the single strongest signal in the dataset and contributes 50% of the
final blend. On its own it is not a geochemistry model — it is a structural
prior over how the dataset was assembled (see the "Limitations" note in the
README).
"""

import numpy as np


def id_knn_proba(train_ids, train_labels, test_ids, k=3, sigma=2.0, n_classes=3):
    """Class probabilities from the ``k`` nearest neighbours in ``Id`` space.

    Args:
        train_ids: 1-D array of training sample Ids.
        train_labels: 1-D array of integer class labels aligned to ``train_ids``.
        test_ids: 1-D array of Ids to predict.
        k: number of nearest neighbours (default 3).
        sigma: exponential decay scale for distance weighting (default 2.0).
        n_classes: number of target classes (default 3).

    Returns:
        ``(len(test_ids), n_classes)`` array of normalised class probabilities.
    """
    train_ids = np.asarray(train_ids)
    train_labels = np.asarray(train_labels)
    test_ids = np.asarray(test_ids)

    probs = np.zeros((len(test_ids), n_classes))
    for i, tid in enumerate(test_ids):
        dists = np.abs(train_ids - tid).astype(float)
        dists[dists == 0] = 0.5  # exact Id match -> small non-zero distance
        nearest = np.argsort(dists)[:k]
        weights = np.exp(-dists[nearest] / sigma)
        for j, ni in enumerate(nearest):
            probs[i, train_labels[ni]] += weights[j]
        probs[i] /= probs[i].sum()
    return probs
