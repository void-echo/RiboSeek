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
        """Load a bundled alphabet by name: ``"sa20"`` (= RS-20 of the paper,
        20 geometric letters) or ``"rs80"`` (RS-20 x nucleotide identity,
        80 letters, the default for database search)."""
        if name in ("sa20", "rs20"):
            with _resources.as_file(
                    _resources.files("riboseek.data").joinpath("sa20.npz")) as p:
                return cls.from_file(str(p))
        if name in ("rs80", "joint80"):
            return RS80Alphabet.from_pretrained()
        raise ValueError(f"Unknown bundled alphabet: {name!r}")

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


# ──────────────────────────────────────────────────────────────────────
#  RS-80: geometric letter x nucleotide identity
# ──────────────────────────────────────────────────────────────────────

BASE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
BASE_ALPHA = 2.0


class RS80Alphabet(Alphabet):
    """80-letter composite alphabet: ``J = RS20 * 4 + base``.

    The substitution matrix is ``M_J[i, j] = M_RS20[i // 4, j // 4] +
    alpha * M_base[i % 4, j % 4]`` with ``M_base`` = +2 on the diagonal and
    -1 off it and ``alpha = 2`` (manuscript, Methods § RS-20 and RS-80
    encodings). Unknown / non-AUGC bases take index 0.
    """

    def __init__(self, base: Alphabet, alpha: float = BASE_ALPHA):
        m20 = base.score_matrix
        K = m20.shape[0]
        mb = -1.0 * np.ones((4, 4)); np.fill_diagonal(mb, 2.0)
        m80 = np.zeros((4 * K, 4 * K))
        for i in range(4 * K):
            for j in range(4 * K):
                m80[i, j] = m20[i // 4, j // 4] + alpha * mb[i % 4, j % 4]
        self.base = base
        self.alpha = float(alpha)
        # centroids of the parent alphabet are kept for feature encoding;
        # K reports 80 letters through the score matrix.
        super().__init__(base.centroids, base.means, base.stds, m20)
        self.score_matrix = np.ascontiguousarray(m80, dtype=np.float64)

    @property
    def K(self) -> int:
        return int(self.score_matrix.shape[0])

    @classmethod
    def from_pretrained(cls) -> "RS80Alphabet":
        return cls(Alphabet.from_pretrained("sa20"))

    @staticmethod
    def join(rs20_labels, sequence: str) -> np.ndarray:
        lab = np.asarray(rs20_labels, dtype=np.int32)
        base = np.zeros(len(lab), dtype=np.int32)
        seq = (sequence or "").upper()
        for i in range(min(len(seq), len(lab))):
            base[i] = BASE_INDEX.get(seq[i], 0)
        return (lab * 4 + base).astype(np.int32)

    def encode(self, features: np.ndarray, sequence: str = "") -> np.ndarray:
        """RS-20 assignment of ``features`` combined with ``sequence``."""
        return self.join(self.base.encode(features), sequence)
