"""
The high-level :class:`Searcher` ties the alphabet, the C aligner, an
optional k-mer prefilter, and an in-memory chain database together. Given
a PDB / mmCIF file (or a pre-computed label sequence) it returns the
top-N most similar entries in the database under the combined NW + SW
z-score.
"""

from __future__ import annotations

import gzip
import importlib.resources as _resources
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .align import NWAligner
from .alphabet import Alphabet
from .features import pdb_to_features
from .prefilter import KmerPrefilter

PathLike = Union[str, os.PathLike]


def _load_db(path: PathLike) -> Dict[str, Dict]:
    """Load an encoded-chain database from JSON or gzipped JSON."""
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _default_db_search_paths() -> List[str]:
    """Where ``Searcher.from_pretrained`` looks for the bundled database."""
    paths: List[str] = []
    home_cache = os.path.expanduser("~/.cache/riboseek/encoded_chains.json")
    home_cache_gz = home_cache + ".gz"
    paths += [home_cache, home_cache_gz]
    return paths


def _resolve_demo_db() -> Optional[str]:
    """Try to locate the bundled demo database inside the installed package."""
    try:
        with _resources.as_file(
                _resources.files("riboseek.data").joinpath("demo_db.json")) as p:
            if p.exists():
                return str(p)
    except (FileNotFoundError, AttributeError):
        pass
    return None


class Searcher:
    """High-level RNA structural search engine."""

    def __init__(self, alphabet: Alphabet, encoded_chains: Dict[str, Dict],
                 gap_penalty: float = -2.0, prefilter_k: int = 6):
        self.alphabet = alphabet
        self.encoded_chains = encoded_chains
        self.gap_penalty = float(gap_penalty)
        self.aligner = NWAligner(alphabet.score_matrix, gap_penalty=gap_penalty)
        self._chain_keys = list(encoded_chains.keys())
        self._prefilter_k = prefilter_k
        self._prefilter: Optional[KmerPrefilter] = None

    # ─────────────────────────────────────────────────────────────────
    #  Construction helpers
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        alphabet: str = "sa20",
        db: Optional[PathLike] = None,
        gap_penalty: float = -2.0,
    ) -> "Searcher":
        """
        Load the bundled alphabet and database.

        Parameters
        ----------
        alphabet : str
            Which bundled alphabet to use. Currently ``"sa20"``.
        db : path-like, optional
            Path to an encoded-chain JSON file. If omitted, looks for a
            full database under ``~/.cache/riboseek/encoded_chains.json``
            (downloaded via ``riboseek download-db``) and otherwise falls
            back to the small demo database bundled with the package.
        """
        alpha = Alphabet.from_pretrained(alphabet)

        chosen: Optional[str] = None
        if db is not None:
            chosen = str(db)
        else:
            for p in _default_db_search_paths():
                if os.path.exists(p):
                    chosen = p
                    break
            if chosen is None:
                chosen = _resolve_demo_db()

        if chosen is None or not os.path.exists(chosen):
            raise FileNotFoundError(
                "No database found. Either run `riboseek download-db` "
                "to fetch the full 16K-chain database, or pass db=<path>."
            )
        return cls(alpha, _load_db(chosen), gap_penalty=gap_penalty)

    @classmethod
    def from_files(cls, alphabet_path: PathLike, db_path: PathLike,
                   gap_penalty: float = -2.0) -> "Searcher":
        return cls(
            Alphabet.from_file(str(alphabet_path)),
            _load_db(db_path),
            gap_penalty=gap_penalty,
        )

    # ─────────────────────────────────────────────────────────────────
    #  Encoding
    # ─────────────────────────────────────────────────────────────────

    def encode(self, pdb_or_features, chain_id: Optional[str] = None
               ) -> np.ndarray:
        """
        Map an RNA structure to its SA-20 label sequence.

        Accepts either a path to a PDB / mmCIF file, or a pre-computed
        ``(n, 15)`` feature matrix.
        """
        if isinstance(pdb_or_features, (str, os.PathLike, Path)):
            f = pdb_to_features(str(pdb_or_features), chain_id=chain_id)
            features = f["features"]
        else:
            features = np.asarray(pdb_or_features)
        return self.alphabet.encode(features)

    # ─────────────────────────────────────────────────────────────────
    #  Search
    # ─────────────────────────────────────────────────────────────────

    def _ensure_prefilter(self) -> KmerPrefilter:
        if self._prefilter is None:
            self._prefilter = KmerPrefilter(self.encoded_chains,
                                            k=self._prefilter_k)
        return self._prefilter

    def search(
        self,
        query,
        top_n: int = 20,
        prefilter: bool = True,
        prefilter_candidates: int = 500,
        chain_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search the database for entries similar to ``query``.

        Parameters
        ----------
        query : path-like, label sequence, or str
            Either a path to a PDB / mmCIF file, an array of labels, or
            the key of an entry already in the database.
        top_n : int
            Number of top hits to return.
        prefilter : bool
            Use k-mer prefilter (much faster on large databases). Off by
            default for small (< 200) databases — turn on for 16K+.
        prefilter_candidates : int
            How many candidates the prefilter forwards to alignment.

        Returns
        -------
        list of dict, sorted by combined NW+SW score, each with keys
        ``chain``, ``combined_score``, ``nw_score``, ``sw_score``,
        ``length``.
        """
        query_key: Optional[str] = None
        if isinstance(query, str) and query in self.encoded_chains:
            query_key = query
            q_labels = np.asarray(
                self.encoded_chains[query]["labels"], dtype=np.int32)
        elif isinstance(query, (str, os.PathLike, Path)):
            q_labels = self.encode(query, chain_id=chain_id).astype(np.int32)
        else:
            q_labels = np.asarray(query, dtype=np.int32)

        if prefilter:
            self._ensure_prefilter()
            cand = self._prefilter.candidates(
                q_labels, top_n=prefilter_candidates, exclude_key=query_key)
            target_keys = [k for k, _ in cand]
        else:
            target_keys = [k for k in self._chain_keys if k != query_key]

        if not target_keys:
            return []

        nw_scores: List[float] = []
        sw_scores: List[float] = []
        for key in target_keys:
            t_labels = np.asarray(
                self.encoded_chains[key]["labels"], dtype=np.int32)
            nw_scores.append(self.aligner.align(q_labels, t_labels, local=False))
            sw_scores.append(self.aligner.align(q_labels, t_labels, local=True))

        nw = np.asarray(nw_scores)
        sw = np.asarray(sw_scores)
        nw_z = (nw - nw.mean()) / (nw.std() + 1e-10)
        sw_z = (sw - sw.mean()) / (sw.std() + 1e-10)
        combined = (nw_z + sw_z) / 2.0

        results = [
            {
                "chain": target_keys[i],
                "combined_score": float(combined[i]),
                "nw_score": float(nw[i]),
                "sw_score": float(sw[i]),
                "length": int(self.encoded_chains[target_keys[i]]["length"]),
            }
            for i in range(len(target_keys))
        ]
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results[:top_n]
