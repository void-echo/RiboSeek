"""
RNA structural alphabet: maps per-residue 15-D features to 20 discrete
states via K-means assignment, and carries the substitution score matrix
used by NW / SW alignment.
"""

from __future__ import annotations

import importlib.resources as _resources
from typing import Optional

import numpy as np


class Alphabet:
    """20-letter (RS-20) structural alphabet."""

    def __init__(self, centroids, means, stds, score_matrix):
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float64)
        self.means = np.ascontiguousarray(means, dtype=np.float64)
        self.stds = np.ascontiguousarray(stds, dtype=np.float64)
        # avoid div-by-zero
        self.stds = np.where(self.stds < 1e-10, 1.0, self.stds)
        self.score_matrix = np.ascontiguousarray(score_matrix, dtype=np.float64)
        if self.score_matrix.shape[0] != self.score_matrix.shape[1]:
            raise ValueError("score_matrix must be square")
        if self.centroids.shape[0] != self.score_matrix.shape[0]:
            raise ValueError("centroids and score_matrix sizes disagree")
        if self.centroids.shape[1] != self.means.shape[0]:
            raise ValueError("centroids and means feature dim disagree")

    @property
    def K(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.centroids.shape[1])

    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, name: str = "sa20") -> "Alphabet":
        """Load a bundled alphabet by name. Currently only ``"sa20"`` exists."""
        if name != "sa20":
            raise ValueError(f"Unknown bundled alphabet: {name!r}")
        with _resources.as_file(
                _resources.files("riboseek.data").joinpath("sa20.npz")) as p:
            return cls.from_file(str(p))

    @classmethod
    def from_file(cls, path: str) -> "Alphabet":
        data = np.load(path)
        required = {"centroids", "means", "stds", "score_matrix"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Alphabet file {path!r} is missing keys: {missing}")
        return cls(
            centroids=data["centroids"],
            means=data["means"],
            stds=data["stds"],
            score_matrix=data["score_matrix"],
        )

    # ─────────────────────────────────────────────────────────────────

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Assign each feature row to the closest centroid → integer label.

        Parameters
        ----------
        features : (n, n_features) ndarray
            Output of :func:`riboseek.features.pdb_to_features`.

        Returns
        -------
        (n,) int32 ndarray of labels in [0, K).
        """
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != self.n_features:
            raise ValueError(
                f"features must have shape (n, {self.n_features}), "
                f"got {feats.shape}")
        # Impute NaNs with column means so they map to 0 after standardization
        feats = np.where(np.isnan(feats), self.means, feats)
        x = (feats - self.means) / self.stds
        # centroids are already in standardized space (K-means was trained
        # on standardized features) — compute squared L2 directly.
        d2 = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(np.int32)
