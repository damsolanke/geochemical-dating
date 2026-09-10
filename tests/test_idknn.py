"""Tests for the Id-based KNN component (src/idknn.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from idknn import id_knn_proba  # noqa: E402


def test_probabilities_are_normalised():
    rng = np.random.default_rng(0)
    train_ids = np.arange(1, 201)
    train_labels = rng.integers(0, 3, size=200)
    test_ids = np.array([0, 7, 50, 133, 200, 999])

    probs = id_knn_proba(train_ids, train_labels, test_ids)

    assert probs.shape == (len(test_ids), 3)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_nearest_id_run_wins():
    # Three contiguous Id runs, one class each: 1-10 -> 0, 11-20 -> 1, 21-30 -> 2.
    train_ids = np.arange(1, 31)
    train_labels = np.repeat([0, 1, 2], 10)
    test_ids = np.array([5, 15, 25, 12, 29])

    probs = id_knn_proba(train_ids, train_labels, test_ids)

    np.testing.assert_array_equal(probs.argmax(axis=1), [0, 1, 2, 1, 2])
    # Deep inside a run, all k=3 neighbours share the class -> a hard vote.
    np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0])


def test_exact_id_match_uses_small_nonzero_distance():
    # Train Ids 10, 11, 12 with labels 0, 1, 1; the query Id 10 matches exactly.
    train_ids = np.array([10, 11, 12])
    train_labels = np.array([0, 1, 1])

    probs = id_knn_proba(train_ids, train_labels, np.array([10]), k=3, sigma=2.0)

    # An exact match is treated as distance 0.5, not 0, so its weight is
    # exp(-0.25): the heaviest single neighbour, but not an infinite override.
    w_exact, w_1, w_2 = np.exp(-0.5 / 2.0), np.exp(-1.0 / 2.0), np.exp(-2.0 / 2.0)
    expected = np.array([w_exact, w_1 + w_2, 0.0]) / (w_exact + w_1 + w_2)
    np.testing.assert_allclose(probs[0], expected)
    assert w_exact > w_1
    assert probs[0].argmax() == 1  # the two-vote run outweighs the exact match


def test_k_and_sigma_control_the_neighbourhood():
    train_ids = np.array([0, 1, 2, 3])
    train_labels = np.array([0, 1, 2, 2])
    query = np.array([0])

    # k=1: only the nearest neighbour votes -> a one-hot answer.
    np.testing.assert_allclose(id_knn_proba(train_ids, train_labels, query, k=1)[0], [1, 0, 0])

    # k=2: exactly the two nearest classes receive mass.
    p2 = id_knn_proba(train_ids, train_labels, query, k=2)[0]
    assert p2[2] == 0 and p2[0] > p2[1] > 0

    # Large sigma flattens the distance weighting toward a uniform vote ...
    p_flat = id_knn_proba(train_ids, train_labels, query, k=3, sigma=1e9)[0]
    np.testing.assert_allclose(p_flat, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    # ... while a small sigma makes the nearest neighbour dominate.
    p_sharp = id_knn_proba(train_ids, train_labels, query, k=3, sigma=0.1)[0]
    assert p_sharp[0] > 0.99
