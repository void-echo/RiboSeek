# RiboSeek

**Spatial-neighbour encoding enables fast RNA 3D structure search.**

RiboSeek encodes each RNA 3D structure as a string over a 20-letter
structural alphabet whose features describe each nucleotide's spatial
neighborhood (top-3 nearest non-sequential neighbors), then runs
C-accelerated Needleman–Wunsch / Smith–Waterman alignment to retrieve
similar structures from a database.

Full-database search against 16K RNA chains takes ~200 ms per query, vs
~46 h for US-align.

## Installation

```bash
pip install riboseek
```

A C compiler is required (gcc/clang on Linux/macOS, MSVC on Windows). On
most systems pip will compile the small C extension automatically.

## Quickstart

```python
from riboseek import Searcher

# Load the default 20-letter alphabet + bundled demo database
searcher = Searcher.from_pretrained()

# Encode an RNA structure from a PDB or mmCIF file
labels = searcher.encode("my_rna.pdb")
print(f"SA-20 sequence: {labels[:50]}...")

# Search the database for similar structures
hits = searcher.search("my_rna.pdb", top_n=10)
for h in hits:
    print(f"  {h['chain']:>10s}  combined={h['combined_score']:+.3f}")
```

Or use the CLI:

```bash
# Encode a single structure
riboseek encode my_rna.pdb

# Search against the bundled demo database
riboseek search my_rna.pdb --top-n 10

# Build a custom database from a directory of PDB / mmCIF files
riboseek build-db ./my_pdbs/ -o ./my_db/

# Search against your custom database
riboseek search my_rna.pdb --db ./my_db/ --top-n 20
```

## Full 16K-chain database

The PyPI package ships with a ~50-chain demo subset so installs stay small.
For the full 16,641-chain experimental RNA database used in the paper:

```bash
riboseek download-db
```

This fetches the full SA-20 encoded chain set (~10 MB compressed) from
the GitHub release into `~/.cache/riboseek/`. Subsequent `riboseek search`
calls will use it automatically when `--db default` (the default) is set.

## What this package does NOT include

This is a minimal release. It does **not** ship the research scripts
behind the paper (feature-ablation studies, alternative discretisation
methods such as VQ-VAE, learned prefilters, RhoFold+-predicted dark-family
pipeline, figure-rendering code, etc.). Those live in the private
research repository.

## Citation

If you use RiboSeek in academic work, please cite the preprint:

> Wang D, Jin J, Qiao J, Wei L, Wu S, Liu Q.
> Spatial-neighbour encoding enables fast RNA 3D structure search.
> *bioRxiv* 2026.04.19.719441 (2026). doi:
> [10.64898/2026.04.19.719441](https://doi.org/10.64898/2026.04.19.719441)

## License

MIT — see [LICENSE](LICENSE).
