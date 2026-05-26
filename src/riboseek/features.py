"""
PDB / mmCIF structure → 15-D structural feature matrix.

The 15 features per nucleotide are:
    [0]  dist_PP                 - sequential P-P distance
    [1]  dist_C4C4               - sequential C4'-C4' distance
    [2]  dist_C1C1               - sequential C1'-C1' distance
    [3]  stacking_angle          - base normal angle with next residue
    [4]  glycosidic_angle        - backbone-to-base orientation
    [5]  neighbor1_dist          - distance to 1st spatial neighbor
    [6]  neighbor1_normal_angle  - base normal angle with 1st neighbor
    [7]  neighbor1_c1_angle      - C1'-centroid angle with 1st neighbor
    [8]  neighbor2_dist          - distance to 2nd spatial neighbor
    [9]  neighbor2_normal_angle  - base normal angle with 2nd neighbor
    [10] neighbor2_c1_angle      - C1'-centroid angle with 2nd neighbor
    [11] neighbor3_dist          - distance to 3rd spatial neighbor
    [12] neighbor3_normal_angle  - base normal angle with 3rd neighbor
    [13] neighbor3_c1_angle      - C1'-centroid angle with 3rd neighbor
    [14] contact_count           - number of spatial contacts within 10A

The pseudo-torsion (η, θ) features used by classical RNA alphabets like
iPARTS are intentionally dropped — feature-ablation experiments showed
they add noise in the presence of multi-neighbor spatial features.
"""

from __future__ import annotations

import gzip
import os
import warnings
from typing import Optional

import numpy as np

N_FEATURES = 15
TOP_K_NEIGHBORS = 3
CONTACT_DIST = 10.0
SEQ_EXCLUDE = 2

_BASE_ATOMS = {
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"],
    "G": ["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}
_THREE_TO_ONE = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
}
_RNA_RESIDUES = set(_THREE_TO_ONE.keys())


# ──────────────────────────────────────────────────────────────────────
#  Step 1: parse PDB / mmCIF → per-residue coordinate arrays
# ──────────────────────────────────────────────────────────────────────

def _load_structure(path: str):
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as e:
        raise ImportError(
            "BioPython is required for PDB/mmCIF parsing. "
            "Install with: pip install biopython"
        ) from e

    pdb_id = os.path.basename(path)
    for suffix in (".gz", ".cif", ".mmcif", ".pdb", ".ent"):
        pdb_id = pdb_id.replace(suffix, "")

    lower = path.lower()
    is_cif = ".cif" in lower or ".mmcif" in lower
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                return parser.get_structure(pdb_id, f)
        return parser.get_structure(path, path)


def _is_rna_residue(residue) -> bool:
    return residue.get_resname().strip() in _RNA_RESIDUES


def _extract_chain_coords(chain_residues):
    """Pull P, C4', C1', base centroid, base normal arrays from RNA residues."""
    n = len(chain_residues)
    P = np.full((n, 3), np.nan)
    C4 = np.full((n, 3), np.nan)
    C1 = np.full((n, 3), np.nan)
    base_centroids = np.full((n, 3), np.nan)
    base_normals = np.full((n, 3), np.nan)

    for i, res in enumerate(chain_residues):
        atom_xyz = {a.get_name(): a.get_vector().get_array() for a in res}
        if "P" in atom_xyz:
            P[i] = atom_xyz["P"]
        if "C4'" in atom_xyz:
            C4[i] = atom_xyz["C4'"]
        if "C1'" in atom_xyz:
            C1[i] = atom_xyz["C1'"]

        resname = _THREE_TO_ONE.get(res.get_resname().strip())
        if resname and resname in _BASE_ATOMS:
            pts = [atom_xyz[a] for a in _BASE_ATOMS[resname] if a in atom_xyz]
            if len(pts) >= 3:
                pts = np.asarray(pts)
                centroid = pts.mean(axis=0)
                base_centroids[i] = centroid
                try:
                    _, _, vh = np.linalg.svd(pts - centroid)
                    base_normals[i] = vh[-1]
                except np.linalg.LinAlgError:
                    pass

    return P, C4, C1, base_centroids, base_normals


# ──────────────────────────────────────────────────────────────────────
#  Step 2: coords → per-residue 15-D feature vectors
# ──────────────────────────────────────────────────────────────────────

def _sequential_distances(P, C4, C1):
    n = P.shape[0]
    d_PP = np.full(n, np.nan)
    d_C4 = np.full(n, np.nan)
    d_C1 = np.full(n, np.nan)
    for i in range(n - 1):
        if not (np.any(np.isnan(P[i])) or np.any(np.isnan(P[i + 1]))):
            d_PP[i] = np.linalg.norm(P[i + 1] - P[i])
        if not (np.any(np.isnan(C4[i])) or np.any(np.isnan(C4[i + 1]))):
            d_C4[i] = np.linalg.norm(C4[i + 1] - C4[i])
        if not (np.any(np.isnan(C1[i])) or np.any(np.isnan(C1[i + 1]))):
            d_C1[i] = np.linalg.norm(C1[i + 1] - C1[i])
    return d_PP, d_C4, d_C1


def _stacking_angles(base_normals):
    n = base_normals.shape[0]
    out = np.full(n, np.nan)
    for i in range(n - 1):
        a, b = base_normals[i], base_normals[i + 1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            continue
        cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        out[i] = np.arccos(abs(cos))
    return out


def _glycosidic_angles(C4, C1, base_centroids):
    n = C4.shape[0]
    out = np.full(n, np.nan)
    for i in range(n - 1):
        if (np.any(np.isnan(C4[i])) or np.any(np.isnan(C4[i + 1]))
                or np.any(np.isnan(C1[i])) or np.any(np.isnan(base_centroids[i]))):
            continue
        bb = C4[i + 1] - C4[i]
        bs = base_centroids[i] - C1[i]
        n1, n2 = np.linalg.norm(bb), np.linalg.norm(bs)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        out[i] = np.arccos(np.clip(np.dot(bb, bs) / (n1 * n2), -1.0, 1.0))
    return out


def _topk_spatial_neighbors(C1, top_k=TOP_K_NEIGHBORS, max_dist=12.0,
                            seq_exclude=SEQ_EXCLUDE):
    from scipy.spatial import cKDTree
    n = C1.shape[0]
    nb_idx = np.full((n, top_k), -1, dtype=int)
    nb_dist = np.full((n, top_k), np.nan)
    valid = ~np.any(np.isnan(C1), axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < seq_exclude * 2 + top_k + 1:
        return nb_idx, nb_dist

    valid_C1 = C1[valid_idx]
    tree = cKDTree(valid_C1)
    k_query = min(seq_exclude * 2 + top_k + 10, len(valid_idx))
    dists, idxs = tree.query(valid_C1, k=k_query)

    for pos, vi in enumerate(valid_idx):
        found = 0
        for q in range(1, k_query):
            if dists[pos, q] > max_dist:
                break
            nb_vi = valid_idx[idxs[pos, q]]
            if abs(vi - nb_vi) > seq_exclude:
                nb_idx[vi, found] = nb_vi
                nb_dist[vi, found] = dists[pos, q]
                found += 1
                if found >= top_k:
                    break
    return nb_idx, nb_dist


def _neighbor_orientations(base_normals, base_centroids, C1, nb_idx,
                           top_k=TOP_K_NEIGHBORS):
    n = base_normals.shape[0]
    normal_a = np.full((n, top_k), np.nan)
    c1_a = np.full((n, top_k), np.nan)
    for i in range(n):
        for r in range(top_k):
            j = nb_idx[i, r]
            if j < 0:
                continue
            ni, nj = base_normals[i], base_normals[j]
            if not (np.any(np.isnan(ni)) or np.any(np.isnan(nj))):
                k1, k2 = np.linalg.norm(ni), np.linalg.norm(nj)
                if k1 > 1e-10 and k2 > 1e-10:
                    normal_a[i, r] = np.arccos(
                        abs(np.clip(np.dot(ni, nj) / (k1 * k2), -1.0, 1.0)))
            if not (np.any(np.isnan(C1[i])) or np.any(np.isnan(C1[j]))
                    or np.any(np.isnan(base_centroids[i]))
                    or np.any(np.isnan(base_centroids[j]))):
                v1 = base_centroids[i] - C1[i]
                v2 = base_centroids[j] - C1[j]
                k1, k2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if k1 > 1e-10 and k2 > 1e-10:
                    c1_a[i, r] = np.arccos(
                        np.clip(np.dot(v1, v2) / (k1 * k2), -1.0, 1.0))
    return normal_a, c1_a


def _contact_count(C1, contact_dist=CONTACT_DIST, seq_exclude=SEQ_EXCLUDE):
    from scipy.spatial import cKDTree
    n = C1.shape[0]
    out = np.zeros(n, dtype=np.float64)
    valid = ~np.any(np.isnan(C1), axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < seq_exclude * 2 + 2:
        return out
    tree = cKDTree(C1[valid_idx])
    pairs = tree.query_pairs(contact_dist, output_type="ndarray")
    for p in range(pairs.shape[0]):
        vi = valid_idx[pairs[p, 0]]
        vj = valid_idx[pairs[p, 1]]
        if abs(vi - vj) > seq_exclude:
            out[vi] += 1
            out[vj] += 1
    return out


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def pdb_to_features(path: str, chain_id: Optional[str] = None) -> dict:
    """
    Parse a PDB / mmCIF file and compute per-residue 15-D structural features.

    Parameters
    ----------
    path : str
        Path to a .pdb / .cif / .mmcif file (optionally .gz-compressed).
    chain_id : str, optional
        Chain identifier (e.g. ``"A"``). If omitted, returns features for
        the longest RNA chain in the structure.

    Returns
    -------
    dict with keys:
        ``features`` (n, 15) float32 ndarray
        ``chain_id`` str
        ``length`` int
        ``sequence`` str (1-letter A/C/G/U, ``N`` for unknown)

    Raises
    ------
    ValueError if no RNA chain is found.
    """
    structure = _load_structure(path)

    chains = []
    for model in structure:
        for chain in model:
            rna = [r for r in chain if r.get_id()[0].strip() == ""
                   and _is_rna_residue(r)]
            if len(rna) >= 5:
                chains.append((chain.get_id(), rna))
        break  # only first model

    if not chains:
        raise ValueError(f"No RNA chain (>=5 residues) found in {path!r}")

    if chain_id is not None:
        match = [c for c in chains if c[0] == chain_id]
        if not match:
            raise ValueError(
                f"Chain {chain_id!r} not found in {path!r}. "
                f"Available: {[c[0] for c in chains]}"
            )
        cid, residues = match[0]
    else:
        cid, residues = max(chains, key=lambda c: len(c[1]))

    P, C4, C1, centroids, normals = _extract_chain_coords(residues)

    d_PP, d_C4, d_C1 = _sequential_distances(P, C4, C1)
    stack = _stacking_angles(normals)
    glyc = _glycosidic_angles(C4, C1, centroids)
    nb_idx, nb_dist = _topk_spatial_neighbors(C1)
    nb_norm_a, nb_c1_a = _neighbor_orientations(normals, centroids, C1, nb_idx)
    contacts = _contact_count(C1)

    cols = [d_PP, d_C4, d_C1, stack, glyc]
    for r in range(TOP_K_NEIGHBORS):
        cols += [nb_dist[:, r], nb_norm_a[:, r], nb_c1_a[:, r]]
    cols.append(contacts)
    feats = np.column_stack(cols).astype(np.float32)
    assert feats.shape[1] == N_FEATURES, feats.shape

    seq = "".join(_THREE_TO_ONE.get(r.get_resname().strip(), "N")
                  for r in residues)

    return {
        "features": feats,
        "chain_id": cid,
        "length": len(residues),
        "sequence": seq,
    }
