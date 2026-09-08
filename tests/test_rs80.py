"""RS-80 alphabet, E-values and the v0.2 CLI paths (demo database only)."""
import numpy as np


def test_rs80_alphabet_shapes():
    from riboseek import Alphabet
    from riboseek.alphabet import RS80Alphabet
    a = Alphabet.from_pretrained("rs80")
    assert isinstance(a, RS80Alphabet)
    assert a.K == 80 and a.score_matrix.shape == (80, 80)
    # diagonal of the base channel: same RS-20 letter, same base scores +2*alpha over different base
    m = a.score_matrix
    assert np.isclose(m[0, 0] - m[0, 1], 3 * a.alpha)


def test_rs80_join_and_encode():
    from riboseek import Alphabet
    a = Alphabet.from_pretrained("rs80")
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((12, 15)).astype(np.float32)
    lab = a.encode(feats, "ACGUACGUACGU")
    assert lab.shape == (12,) and lab.max() < 80 and lab.min() >= 0
    assert set(lab % 4) <= {0, 1, 2, 3}


def test_search_demo_db_rs80_with_evalues():
    from riboseek import Searcher
    from riboseek.search import _resolve_demo_db
    s = Searcher.from_pretrained("rs80", db=_resolve_demo_db())
    assert s.is_rs80
    q = next(iter(s.encoded_chains))
    hits = s.search(q, top_n=3, prefilter=False)
    assert hits and all("evalue" in h for h in hits)
    assert all(0.0 <= h["evalue"] <= len(s.encoded_chains) for h in hits)


def test_evalue_monotone_in_score():
    from riboseek import stats
    e1 = stats.evalue(1.5, 80, 80, 15391)
    e2 = stats.evalue(2.5, 80, 80, 15391)
    assert e2 < e1
