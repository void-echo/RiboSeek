"""
Command-line interface for riboseek.

Sub-commands:
    riboseek encode <pdb>             - print the SA-20 label sequence
    riboseek search <pdb>             - search the bundled / cached database
    riboseek build-db <dir> -o <out>  - encode a directory of PDBs into a db
    riboseek download-db              - fetch the full 16K-chain database
    riboseek info                     - show version / cache / db status
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

import numpy as np


# Default location of the full database (populated by `riboseek download-db`)
DEFAULT_DB_DIR = Path.home() / ".cache" / "riboseek"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "encoded_chains.json"

# GitHub release asset for the full database. Pinned to v0.1.0.
FULL_DB_URL = (
    "https://github.com/void-echo/RiboSeek/releases/download/v0.1.0/"
    "encoded_chains.json.gz"
)


# ──────────────────────────────────────────────────────────────────────
#  Commands
# ──────────────────────────────────────────────────────────────────────

def cmd_encode(args) -> int:
    from .features import pdb_to_features
    from .alphabet import Alphabet

    f = pdb_to_features(args.pdb, chain_id=args.chain)
    alpha = Alphabet.from_pretrained(args.alphabet)
    labels = alpha.encode(f["features"])
    if args.format == "labels":
        print(",".join(str(int(x)) for x in labels))
    else:  # letters: A..T (20 letters)
        letters = "ABCDEFGHIJKLMNOPQRST"
        print("".join(letters[int(x)] for x in labels))
    print(
        f"# chain={f['chain_id']} length={f['length']} sequence={f['sequence'][:60]}"
        f"{'...' if len(f['sequence']) > 60 else ''}",
        file=sys.stderr,
    )
    return 0


def cmd_search(args) -> int:
    from .search import Searcher

    print("Loading riboseek search engine...", flush=True)
    t0 = time.time()
    searcher = Searcher.from_pretrained(
        alphabet=args.alphabet,
        db=args.db if args.db != "default" else None,
    )
    n_db = len(searcher.encoded_chains)
    print(f"  Database: {n_db} chains  (loaded in {time.time() - t0:.2f}s)",
          flush=True)

    print(f"\nQuery: {args.pdb}", flush=True)
    t0 = time.time()
    hits = searcher.search(
        args.pdb,
        top_n=args.top_n,
        prefilter=not args.no_prefilter and n_db > 200,
        prefilter_candidates=args.prefilter_candidates,
        chain_id=args.chain,
    )
    print(f"Search time: {time.time() - t0:.3f}s\n", flush=True)

    if not hits:
        print("(no hits)")
        return 0

    header = f"{'rank':>4}  {'chain':>20}  {'combined':>9}  {'nw':>7}  {'sw':>7}  {'length':>6}"
    print(header)
    print("-" * len(header))
    for i, h in enumerate(hits, 1):
        print(f"{i:>4}  {h['chain']:>20}  {h['combined_score']:+9.4f}  "
              f"{h['nw_score']:+7.4f}  {h['sw_score']:+7.4f}  {h['length']:>6d}")
    return 0


def cmd_build_db(args) -> int:
    from .alphabet import Alphabet
    from .features import pdb_to_features

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"Not a directory: {in_dir}", file=sys.stderr)
        return 2

    paths: List[Path] = []
    for suffix in ("*.pdb", "*.cif", "*.mmcif", "*.pdb.gz", "*.cif.gz"):
        paths.extend(in_dir.glob(suffix))
    paths.sort()
    if not paths:
        print(f"No structure files found in {in_dir}", file=sys.stderr)
        return 2

    print(f"Encoding {len(paths)} structures with alphabet '{args.alphabet}'...")
    alpha = Alphabet.from_pretrained(args.alphabet)
    encoded = {}
    failures = 0
    for i, p in enumerate(paths, 1):
        try:
            f = pdb_to_features(str(p))
            labels = alpha.encode(f["features"])
            key = p.stem.replace(".pdb", "").replace(".cif", "")
            if f["chain_id"]:
                key = f"{key}_{f['chain_id']}"
            encoded[key] = {"labels": labels.tolist(),
                            "length": int(len(labels))}
        except Exception as e:
            failures += 1
            print(f"  [WARN] {p.name}: {e}", file=sys.stderr)
        if i % 100 == 0:
            print(f"  {i}/{len(paths)} done", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).endswith(".gz"):
        with gzip.open(out_path, "wt") as f:
            json.dump(encoded, f)
    else:
        with open(out_path, "w") as f:
            json.dump(encoded, f)
    print(f"\nWrote {len(encoded)} chains to {out_path}  "
          f"({failures} failures)")
    return 0


def cmd_download_db(args) -> int:
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DEFAULT_DB_DIR / "encoded_chains.json.gz.part"
    final = DEFAULT_DB_PATH

    if final.exists() and not args.force:
        print(f"Database already present: {final}")
        print("Re-run with --force to redownload.")
        return 0

    print(f"Downloading {FULL_DB_URL} ...")
    try:
        with urllib.request.urlopen(FULL_DB_URL) as r, open(tmp, "wb") as out:
            total = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
            print(f"  fetched {total / 1e6:.1f} MB")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print("Decompressing ...")
    with gzip.open(tmp, "rb") as fin, open(final, "wb") as fout:
        while True:
            chunk = fin.read(1 << 16)
            if not chunk:
                break
            fout.write(chunk)
    tmp.unlink()
    print(f"Database ready: {final}")
    return 0


def cmd_info(args) -> int:
    from . import __version__

    print(f"riboseek {__version__}")
    print(f"  bundled alphabets : sa20")
    cached = DEFAULT_DB_PATH.exists()
    print(f"  cached full db    : {DEFAULT_DB_PATH}  "
          f"({'present' if cached else 'not downloaded'})")
    try:
        from .align import _resolve_lib  # noqa: F401
        _resolve_lib()
        print("  C extension       : ok")
    except Exception as e:
        print(f"  C extension       : FAILED ({e})")
    return 0


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(
        prog="riboseek",
        description="Ultra-fast RNA 3D structure search via spatial-neighbor "
                    "structural alphabets.",
    )
    p.add_argument("--version", action="version",
                   version=f"riboseek {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="encode a PDB / mmCIF into SA-20")
    pe.add_argument("pdb", help="path to .pdb / .cif / .mmcif (optionally .gz)")
    pe.add_argument("--chain", default=None,
                    help="chain id (defaults to longest)")
    pe.add_argument("--alphabet", default="sa20")
    pe.add_argument("--format", choices=("labels", "letters"), default="letters")
    pe.set_defaults(func=cmd_encode)

    ps = sub.add_parser("search", help="search the database")
    ps.add_argument("pdb", help="query PDB / mmCIF file")
    ps.add_argument("--chain", default=None)
    ps.add_argument("--alphabet", default="sa20")
    ps.add_argument("--db", default="default",
                    help="path to encoded-chain JSON (defaults to bundled / cached)")
    ps.add_argument("--top-n", type=int, default=10)
    ps.add_argument("--no-prefilter", action="store_true")
    ps.add_argument("--prefilter-candidates", type=int, default=500)
    ps.set_defaults(func=cmd_search)

    pb = sub.add_parser("build-db", help="encode a directory of structures")
    pb.add_argument("input_dir", help="directory containing .pdb / .cif files")
    pb.add_argument("-o", "--output", required=True,
                    help="output JSON path (.json or .json.gz)")
    pb.add_argument("--alphabet", default="sa20")
    pb.set_defaults(func=cmd_build_db)

    pd = sub.add_parser("download-db", help="fetch the full 16K-chain database")
    pd.add_argument("--force", action="store_true",
                    help="re-download even if cache exists")
    pd.set_defaults(func=cmd_download_db)

    pi = sub.add_parser("info", help="show version / status")
    pi.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
