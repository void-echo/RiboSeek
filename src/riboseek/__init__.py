"""
RiboSeek: Ultra-fast RNA 3D structure search via spatial-neighbor
structural alphabets.

Quickstart:

    from riboseek import Searcher
    searcher = Searcher.from_pretrained()
    labels = searcher.encode("my_rna.pdb")
    hits = searcher.search("my_rna.pdb", top_n=10)

See https://github.com/void-echo/RiboSeek for full documentation.
"""

from ._version import __version__
from .alphabet import Alphabet
from .align import NWAligner
from .features import pdb_to_features
from .search import Searcher

__all__ = [
    "__version__",
    "Alphabet",
    "NWAligner",
    "Searcher",
    "pdb_to_features",
]
