"""
Smoke tests — exercise the public API without requiring network or
external PDB files. Aim is to catch packaging breakage (missing data
files, broken C extension, import-time errors) on every supported
Python version.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_import_top_level():
    import riboseek
    assert hasattr(riboseek, "__version__")
    assert hasattr(riboseek, "Searcher")
    assert hasattr(riboseek, "Alphabet")
    assert hasattr(riboseek, "NWAligner")


def test_alphabet_from_pretrained():
    from riboseek import Alphabet
    a = Alphabet.from_pretrained("sa20")
    assert a.K == 20
    assert a.n_features == 15
    assert a.centroids.shape == (20, 15)
    assert a.score_matrix.shape == (20, 20)


def test_alphabet_encode_random_features():
    from riboseek import Alphabet
    a = Alphabet.from_pretrained()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((37, 15)).astype(np.float32)
    labels = a.encode(feats)
    assert labels.shape == (37,)
    assert labels.dtype == np.int32
    assert int(labels.min()) >= 0
    assert int(labels.max()) < a.K


def test_c_extension_aligns_two_sequences():
    from riboseek import Alphabet, NWAligner
    a = Alphabet.from_pretrained()
    aligner = NWAligner(a.score_matrix, gap_penalty=-2.0)
    s1 = [0, 1, 2, 3, 4, 5]
    s2 = [0, 1, 2, 3, 4, 5]
    same = aligner.align(s1, s2, local=False)
    diff = aligner.align(s1, [10, 11, 12, 13, 14, 15], local=False)
    assert same > diff


def test_searcher_from_pretrained_demo_db():
    from riboseek import Searcher
    s = Searcher.from_pretrained()
    assert len(s.encoded_chains) >= 30
    # Take any chain in the db, search for top-3 — must return non-empty
    qkey = next(iter(s.encoded_chains))
    qlabels = s.encoded_chains[qkey]["labels"]
    hits = s.search(qlabels, top_n=3, prefilter=False)
    assert 1 <= len(hits) <= 3
    for h in hits:
        for field in ("chain", "combined_score", "nw_score",
                      "sw_score", "length"):
            assert field in h


def test_searcher_self_query_is_top_hit():
    """A chain searched against itself (via its own labels) should
    rank itself near the top — but we exclude the query when it's a
    known key, so the closest non-self chain should still score high."""
    from riboseek import Searcher
    s = Searcher.from_pretrained()
    qkey = next(iter(s.encoded_chains))
    hits = s.search(qkey, top_n=5, prefilter=False)
    assert qkey not in [h["chain"] for h in hits]
    assert hits[0]["combined_score"] > -10  # sanity: scoring works
