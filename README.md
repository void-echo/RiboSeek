# RiboSeek

**Spatial-neighbour encoding enables fast RNA 3D structure search.**

RiboSeek encodes each RNA 3D structure as a string over a structural
alphabet whose letters describe each nucleotide's spatial neighbourhood
(its three nearest non-sequential neighbours), then runs C-accelerated
Needleman–Wunsch / Smith–Waterman alignment to retrieve similar structures
from a database. Two alphabets are provided: **RS-80** (structure × base
identity, the default for database search) and **RS-20** (geometry only,
internal name `sa20`).

Searching the 15,391-chain experimental RNA database takes about 0.1 s
(median) per query on one CPU core.

## Installation

```bash
pip install riboseek
```

A C compiler is required (gcc/clang on Linux/macOS, MSVC on Windows); pip
compiles the small C extension automatically.

## Quickstart

```bash
pip install riboseek
riboseek download-db                      # 8 MB, once
riboseek search my_rna.cif --top-n 10
```

Both mmCIF and legacy PDB files are accepted (`.cif`, `.mmcif`, `.pdb`,
`.ent`, optionally gzip-compressed). The longest RNA chain is used unless
`--chain` is given. For yeast tRNA-Phe (PDB 1EHZ) the output reads:

```
$ riboseek search 1ehz.cif --top-n 5
Query: 1ehz.cif

rank                 chain   combined       nw       sw     evalue  length
--------------------------------------------------------------------------
   1                1ehz_A    +0.9082  +5.5569  +5.5569    1.4e-12      62
   2                1evv_A    +0.8341  +5.4729  +5.4729    3.5e-12      62
   3                6tna_A    +0.8285  +5.4665  +5.4665    3.7e-12      62
   4                1tn1_A    +0.7927  +5.4259  +5.4259    5.8e-12      62
   5                1tn2_A    +0.7927  +5.4259  +5.4259    5.8e-12      62
```

The columns are the database chain (PDB id and author chain id), the
composite score the list is sorted by (mean of the per-query z-scored
global and local alignment scores), the two length-normalised alignment
scores, the E-value of the hit (E < 0.01 is a practical threshold) and the
chain length in encoded residues.

More options:

```bash
riboseek search ./structures/ --tsv hits.tsv      # a directory: one ranked list per file, all hits to one TSV
riboseek search my_rna.cif --keep-modified        # fold modified nucleotides onto their parent base instead of dropping them
riboseek search my_rna.cif --alphabet sa20        # search with RS-20 (geometry only)
riboseek encode my_rna.cif                        # print the RS-20 letter string
riboseek build-db ./my_pdbs/ -o my_db.json.gz     # encode your own structures into a database
riboseek build-db ./new_pdbs/ -o my_db.json.gz --append my_db.json.gz   # add newly deposited structures
riboseek search my_rna.cif --db my_db.json.gz
```

A Colab notebook with the same steps is in
[`examples/riboseek_colab.ipynb`](examples/riboseek_colab.ipynb).

Python API:

```python
from riboseek import Searcher

searcher = Searcher.from_pretrained()          # RS-80 by default
hits = searcher.search("my_rna.cif", top_n=10)
for h in hits:
    print(f"{h['chain']:>10s}  combined={h['combined_score']:+.3f}  E={h['evalue']:.1e}")
```

## Full 15,391-chain database

The PyPI package ships with a ~50-chain demo subset so installs stay small.
`riboseek download-db` fetches the full 15,391-chain encoded database used in
the paper (8 MB compressed, with sequences for RS-80 and E-values) from the
GitHub release into `~/.cache/riboseek/`; subsequent `riboseek search` calls
use it automatically. The database is refreshed with the PDB on each release;
the release page states the PDB snapshot date of each asset.

The benchmark pair lists of the paper are in [`benchmark/`](benchmark/).

## Citation

If you use RiboSeek in academic work, please cite the preprint (Nucleic Acids Research, in revision):

> Wang D, Jin J, Qiao J, Wei L, Wu S, Liu Q.
> Spatial-neighbour encoding enables fast RNA 3D structure search.
> *bioRxiv* 2026.04.19.719441 (2026). doi:
> [10.64898/2026.04.19.719441](https://doi.org/10.64898/2026.04.19.719441)

## License

MIT — see [LICENSE](LICENSE).
