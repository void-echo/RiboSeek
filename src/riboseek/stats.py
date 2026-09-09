"""
Significance of a hit: P-value and E-value for RS-80 local-alignment scores.

Karlin–Altschul statistics do not apply to this scoring scheme (the score of
unrelated chains grows linearly with the shorter chain, not with log(mn)), so
the null is empirical: 168,000 alignments of database chains against shuffled
database chains, expressed as the local score per residue of the shorter
chain (exactly the ``sw_score`` the search reports), with the upper 5 % tail
modelled as an exponential whose threshold ``t`` and scale ``beta`` are smooth
functions of ``L = min(n, m)`` and ``r = max(n, m) / L``::

    t    = a0 + a1 ln L + a2 (ln L)^2 + a3 ln r + a4 ln L ln r
    beta = exp(b0 + b1 ln L + b2 ln r)
    P(score >= s) = 0.05 * exp(-(s - t) / beta)      (clipped to [0, 1])
    E = P * N_db

The coefficients ship in ``riboseek/data/evalue_rs80.json``.
"""

from __future__ import annotations

import importlib.resources as _resources
import json
from typing import Optional

import numpy as np

_CAL = None


def load_calibration(alphabet: str = "rs80") -> dict:
    global _CAL
    if _CAL is None:
        with _resources.as_file(
                _resources.files("riboseek.data").joinpath(f"evalue_{alphabet}.json")) as p:
            _CAL = json.load(open(p))
    return _CAL


def tail_params(L, r, theta):
    L = np.asarray(L, dtype=float)
    r = np.asarray(r, dtype=float)
    lL, lr = np.log(L), np.log(r)
    t = theta[0] + theta[1] * lL + theta[2] * lL ** 2 + theta[3] * lr + theta[4] * lL * lr
    beta = np.exp(theta[5] + theta[6] * lL + theta[7] * lr)
    return t, beta


def pvalue(sw_score, n_query: int, n_target, calibration: Optional[dict] = None):
    """P(random chain pair of these lengths scores >= ``sw_score``)."""
    cal = calibration or load_calibration()
    theta = np.asarray(cal["theta"], dtype=float)
    n_t = np.asarray(n_target, dtype=float)
    L = np.minimum(n_query, n_t)
    r = np.maximum(n_query, n_t) / L
    t, beta = tail_params(L, r, theta)
    p = cal.get("tail_fraction", 0.05) * np.exp(-(np.asarray(sw_score, dtype=float) - t) / beta)
    return np.clip(p, 0.0, 1.0)


def evalue(sw_score, n_query: int, n_target, n_db: int, calibration: Optional[dict] = None):
    """Expected number of database chains scoring this high by chance."""
    return pvalue(sw_score, n_query, n_target, calibration) * float(n_db)
