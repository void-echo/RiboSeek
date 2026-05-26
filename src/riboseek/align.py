"""
Python wrapper around the compiled NW/SW alignment kernels in
``riboseek._nw_align``. Loaded via ctypes so the data path stays
on raw C buffers — no PyObject overhead per cell.
"""

from __future__ import annotations

import ctypes
import importlib
from ctypes import POINTER, c_double, c_int

import numpy as np


def _resolve_lib():
    """Locate the compiled C extension and load it as a plain DLL/.so/.dylib."""
    mod = importlib.import_module("riboseek._nw_align")
    if not getattr(mod, "__file__", None):
        raise RuntimeError(
            "riboseek._nw_align has no __file__; the C extension may not be built"
        )
    lib = ctypes.CDLL(mod.__file__)

    lib.nw_align.argtypes = [
        POINTER(c_int), c_int,
        POINTER(c_int), c_int,
        POINTER(c_double), c_int,
        c_double,
    ]
    lib.nw_align.restype = c_double

    lib.sw_align.argtypes = lib.nw_align.argtypes
    lib.sw_align.restype = c_double

    lib.batch_nw_align.argtypes = [
        POINTER(c_int), c_int,
        POINTER(c_int), POINTER(c_int),
        POINTER(c_int), c_int,
        POINTER(c_double), c_int,
        c_double,
        POINTER(c_double),
    ]
    lib.batch_nw_align.restype = None

    lib.batch_pairwise_align.argtypes = [
        POINTER(c_int), c_int,
        POINTER(c_int), POINTER(c_int),
        POINTER(c_int),
        POINTER(c_double), c_int,
        c_double,
        POINTER(c_double),
    ]
    lib.batch_pairwise_align.restype = None
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _resolve_lib()
    return _LIB


class NWAligner:
    """Compiled NW (global) + SW (local) alignment over a fixed score matrix."""

    def __init__(self, score_matrix: np.ndarray, gap_penalty: float = -2.0):
        self._lib = _lib()
        sm = np.ascontiguousarray(score_matrix, dtype=np.float64)
        if sm.ndim != 2 or sm.shape[0] != sm.shape[1]:
            raise ValueError("score_matrix must be square 2-D")
        self.K = int(sm.shape[0])
        self.gap_penalty = float(gap_penalty)
        self._score = sm
        self._score_ptr = self._score.ctypes.data_as(POINTER(c_double))

    # ─────────────────────────────────────────────────────────────────

    def align(self, seq1, seq2, local: bool = False) -> float:
        s1 = np.ascontiguousarray(seq1, dtype=np.int32)
        s2 = np.ascontiguousarray(seq2, dtype=np.int32)
        fn = self._lib.sw_align if local else self._lib.nw_align
        return float(fn(
            s1.ctypes.data_as(POINTER(c_int)), len(s1),
            s2.ctypes.data_as(POINTER(c_int)), len(s2),
            self._score_ptr, self.K, self.gap_penalty,
        ))

    def align_batch(self, query, targets) -> np.ndarray:
        """Align ``query`` against many ``targets``; returns NW scores."""
        q = np.ascontiguousarray(query, dtype=np.int32)
        n = len(targets)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        concat = np.concatenate([np.asarray(t, dtype=np.int32) for t in targets])
        offsets = np.zeros(n, dtype=np.int32)
        lengths = np.zeros(n, dtype=np.int32)
        pos = 0
        for i, t in enumerate(targets):
            offsets[i] = pos
            lengths[i] = len(t)
            pos += len(t)

        scores = np.zeros(n, dtype=np.float64)
        self._lib.batch_nw_align(
            q.ctypes.data_as(POINTER(c_int)), len(q),
            concat.ctypes.data_as(POINTER(c_int)),
            offsets.ctypes.data_as(POINTER(c_int)),
            lengths.ctypes.data_as(POINTER(c_int)), n,
            self._score_ptr, self.K, self.gap_penalty,
            scores.ctypes.data_as(POINTER(c_double)),
        )
        return scores

    def align_pairs(self, pairs, all_sequences) -> np.ndarray:
        """Align a list of (i, j) pairs from a sequence collection."""
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros(0, dtype=np.float64)
        pairs_flat = np.ascontiguousarray(
            np.asarray(pairs, dtype=np.int32).flatten())
        concat = np.concatenate(
            [np.asarray(s, dtype=np.int32) for s in all_sequences])
        n_seqs = len(all_sequences)
        offsets = np.zeros(n_seqs, dtype=np.int32)
        lengths = np.zeros(n_seqs, dtype=np.int32)
        pos = 0
        for i, s in enumerate(all_sequences):
            offsets[i] = pos
            lengths[i] = len(s)
            pos += len(s)
        scores = np.zeros(n_pairs, dtype=np.float64)
        self._lib.batch_pairwise_align(
            pairs_flat.ctypes.data_as(POINTER(c_int)), n_pairs,
            concat.ctypes.data_as(POINTER(c_int)),
            offsets.ctypes.data_as(POINTER(c_int)),
            lengths.ctypes.data_as(POINTER(c_int)),
            self._score_ptr, self.K, self.gap_penalty,
            scores.ctypes.data_as(POINTER(c_double)),
        )
        return scores
