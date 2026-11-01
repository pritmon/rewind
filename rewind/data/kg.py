"""PrimeKG loading, the typed knowledge-graph container, and the held-out split."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

GENE_TYPE = "gene/protein"


@dataclass
class KnowledgeGraph:
    names: List[str]
    types: List[str]
    relations: List[str]
    edge_index: torch.Tensor  # [2, E] long; every edge is present in both directions
    edge_type: torch.Tensor   # [E] long

    @property
    def num_nodes(self) -> int:
        return len(self.names)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @cached_property
    def node_type_ids(self) -> Tuple[List[str], torch.Tensor]:
        vocab = sorted(set(self.types))
        index = {t: i for i, t in enumerate(vocab)}
        return vocab, torch.tensor([index[t] for t in self.types], dtype=torch.long)

    @cached_property
    def _name_index(self) -> Dict[str, List[int]]:
        index: Dict[str, List[int]] = {}
        for i, name in enumerate(self.names):
            index.setdefault(name.lower(), []).append(i)
        return index

    @cached_property
    def csr(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(rowptr, neighbour, relation), sorted by source node."""
        order = torch.argsort(self.edge_index[0], stable=True)
        counts = torch.bincount(self.edge_index[0], minlength=self.num_nodes)
        rowptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
        rowptr[1:] = torch.cumsum(counts, 0)
        return rowptr, self.edge_index[1, order], self.edge_type[order]

    def lookup(self, name: str, node_type: Optional[str] = None) -> int:
        hits = [i for i in self._name_index.get(name.lower(), []) if node_type is None or self.types[i] == node_type]
        if not hits:
            raise KeyError(f"no node named {name!r}" + (f" of type {node_type!r}" if node_type else ""))
        return hits[0]

    def resolve(self, names: Sequence[str], node_type: Optional[str] = None) -> Tuple[List[int], List[str]]:
        """Look up many names at once; returns (found ids, missing names)."""
        found, missing = [], []
        for name in names:
            try:
                found.append(self.lookup(name, node_type))
            except KeyError:
                missing.append(name)
        return found, missing

    def search(self, query: str, node_type: Optional[str] = None, limit: int = 10) -> List[Tuple[int, str, str]]:
        q = query.lower()
        hits = [
            (i, n, t)
            for i, (n, t) in enumerate(zip(self.names, self.types))
            if q in n.lower() and (node_type is None or t == node_type)
        ]
        hits.sort(key=lambda hit: (len(hit[1]), hit[1]))  # exact and short matches first
        return hits[:limit]

    def nodes_of_type(self, node_type: str) -> torch.Tensor:
        return torch.tensor([i for i, t in enumerate(self.types) if t == node_type], dtype=torch.long)

    def relation_id(self, name: str) -> int:
        return self.relations.index(name)

    def neighbors(self, node: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rowptr, col, rel = self.csr
        start, end = rowptr[node].item(), rowptr[node + 1].item()
        return col[start:end], rel[start:end]

    def degree(self) -> torch.Tensor:
        return torch.bincount(self.edge_index[0], minlength=self.num_nodes)

    def filter_edges(self, keep: torch.Tensor) -> "KnowledgeGraph":
        return KnowledgeGraph(self.names, self.types, self.relations, self.edge_index[:, keep], self.edge_type[keep])

    def save(self, path: str) -> None:
        torch.save(
            {"names": self.names, "types": self.types, "relations": self.relations,
             "edge_index": self.edge_index, "edge_type": self.edge_type},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        return cls(**torch.load(path, weights_only=False))


def from_primekg(df: pd.DataFrame) -> KnowledgeGraph:
    """Build a graph from PrimeKG's kg.csv columns (relation, x_index, x_name, x_type, y_index, ...)."""
    num_nodes = int(max(df.x_index.max(), df.y_index.max())) + 1
    names, types = [""] * num_nodes, [""] * num_nodes
    for idx_col, name_col, type_col in (("x_index", "x_name", "x_type"), ("y_index", "y_name", "y_type")):
        for i, name, node_type in df[[idx_col, name_col, type_col]].drop_duplicates(idx_col).itertuples(index=False):
            names[i], types[i] = str(name), node_type

    relations = sorted(df.relation.unique())
    rel_id = {r: i for i, r in enumerate(relations)}
    src = df.x_index.to_numpy(np.int64)
    dst = df.y_index.to_numpy(np.int64)
    rel = df.relation.map(rel_id).to_numpy(np.int64)

    # PrimeKG lists most edges in both directions already; make that guaranteed, then dedupe via a packed key.
    src, dst, rel = np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([rel, rel])
    keys = np.unique((src * len(relations) + rel) * num_nodes + dst)
    dst = keys % num_nodes
    rel = (keys // num_nodes) % len(relations)
    src = keys // num_nodes // len(relations)

    return KnowledgeGraph(
        names=names,
        types=types,
        relations=relations,
        edge_index=torch.from_numpy(np.stack([src, dst])),
        edge_type=torch.from_numpy(rel),
    )


ONTOLOGY_RELATIONS = ("bioprocess_bioprocess", "pathway_pathway", "molfunc_molfunc")


@dataclass
class Holdout:
    """The blinded graph plus everything needed to audit what was hidden."""

    graph: KnowledgeGraph
    triples: torch.Tensor      # [H, 3] removed gene->seed-target edges (src, rel, dst)
    related: torch.Tensor      # [R, 3] removed gene->related-term edges (the leak paths)
    seed_ids: List[int]        # the target terms named in the config
    related_ids: List[int]     # terms reached from the seeds along ontology edges
    missing: List[str]

    @property
    def gene_ids(self) -> List[int]:
        return sorted(set(self.triples[:, 0].tolist()))


def expand_terms(
    kg: KnowledgeGraph, seed_ids: Sequence[int], hops: int = 1, relations: Sequence[str] = ONTOLOGY_RELATIONS
) -> List[int]:
    """Terms reachable from the seeds along ontology edges, e.g. "neuronal stem cell population
    maintenance" from "stem cell population maintenance". Seeds themselves are excluded."""
    rel_ids = torch.tensor([kg.relation_id(r) for r in relations if r in kg.relations], dtype=torch.long)
    if rel_ids.numel() == 0:
        return []
    src, dst = kg.edge_index
    on_ontology = torch.isin(kg.edge_type, rel_ids)
    seen = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(hops):
        mask = on_ontology & torch.isin(src, torch.tensor(sorted(frontier), dtype=torch.long))
        frontier = set(dst[mask].tolist()) - seen
        if not frontier:
            break
        seen |= frontier
    return sorted(seen - set(seed_ids))


def hold_out(
    kg: KnowledgeGraph,
    genes: Sequence[str],
    targets: Sequence[str],
    expand_hops: int = 0,
) -> Holdout:
    """Delete every edge between `genes` and the target terms, in both directions.

    With `expand_hops > 0` the targets are widened along ontology edges first, so a held-out gene
    cannot be recovered through a near-synonym of the target (the "related term" leak). Those extra
    removals are reported separately in `Holdout.related`, since they are a stricter test, not the
    fact being rediscovered.
    """
    gene_ids, missing_genes = kg.resolve(genes, GENE_TYPE)
    seed_ids, missing_targets = kg.resolve(targets)
    related_ids = expand_terms(kg, seed_ids, expand_hops) if expand_hops else []

    g = torch.tensor(gene_ids, dtype=torch.long)
    src, dst = kg.edge_index

    def between(term_ids: List[int]) -> torch.Tensor:
        if not term_ids:
            return torch.zeros(kg.num_edges, dtype=torch.bool)
        t = torch.tensor(term_ids, dtype=torch.long)
        return (torch.isin(src, g) & torch.isin(dst, t)) | (torch.isin(src, t) & torch.isin(dst, g))

    on_seeds, on_related = between(seed_ids), between(related_ids)
    forward = on_seeds & torch.isin(src, g)
    related_forward = on_related & torch.isin(src, g)
    return Holdout(
        graph=kg.filter_edges(~(on_seeds | on_related)),
        triples=torch.stack([src[forward], kg.edge_type[forward], dst[forward]], dim=1),
        related=torch.stack([src[related_forward], kg.edge_type[related_forward], dst[related_forward]], dim=1),
        seed_ids=seed_ids,
        related_ids=related_ids,
        missing=missing_genes + missing_targets,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PrimeKG into a tensor graph.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("csv")
    build.add_argument("out")
    args = parser.parse_args()

    kg = from_primekg(pd.read_csv(args.csv, low_memory=False))
    kg.save(args.out)
    print(f"{kg.num_nodes:,} nodes, {kg.num_edges:,} directed edges, {kg.num_relations} relations -> {args.out}")


if __name__ == "__main__":
    main()
