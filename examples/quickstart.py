"""
End-to-end demo of RiboSeek's Python API.

This script picks an arbitrary chain from the bundled demo database,
re-encodes the query side using its own SA-20 labels, and searches the
database for similar structures. Use it as a template for your own
queries from PDB / mmCIF files.

Run with:
    python examples/quickstart.py
"""

from riboseek import Searcher


def main() -> None:
    print("Loading RiboSeek with the bundled demo database ...")
    s = Searcher.from_pretrained()
    print(f"  alphabet K = {s.alphabet.K}")
    print(f"  database  : {len(s.encoded_chains)} chains")

    # Pick the first chain as a self-query example. In a real workflow
    # you would call s.search("my_rna.pdb", top_n=10) instead. Passing
    # the chain key (a string) lets Searcher exclude the query itself
    # from the hits.
    query_key = next(iter(s.encoded_chains))
    query_length = s.encoded_chains[query_key]["length"]
    print(f"\nQuerying with database entry: {query_key} "
          f"(length {query_length})")

    hits = s.search(query_key, top_n=5, prefilter=False)
    print(f"\nTop {len(hits)} hits:")
    print(f"  {'chain':>15}  {'combined':>9}  {'nw':>7}  {'sw':>7}  "
          f"{'length':>6}")
    for h in hits:
        print(f"  {h['chain']:>15}  {h['combined_score']:+9.4f}  "
              f"{h['nw_score']:+7.4f}  {h['sw_score']:+7.4f}  "
              f"{h['length']:>6d}")


if __name__ == "__main__":
    main()
