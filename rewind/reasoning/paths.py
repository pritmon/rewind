"""Multi-hop explanations: the highest-scoring paths from a gene to a target term."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from rewind.data.kg import KnowledgeGraph
from rewind.models.gnn import RewindGNN


@dataclass
class Path:
    nodes: List[int]
    relations: List[int]
    log_prob: float

    @property
    def confidence(self) -> float:
        """Geometric mean of edge probabilities, so longer paths aren't penalised just for length."""
        return float(torch.tensor(self.log_prob / len(self.relations)).exp())


UNREACHABLE = 127


def hops_to(kg: KnowledgeGraph, target: int, max_hops: int) -> torch.Tensor:
    """Breadth-first hop count from every node to `target` (the graph is symmetric), capped at max_hops.

    This is what makes the forward search reliable: without it, a hub gene's hundreds of neighbours
    swamp the beam and the one neighbour that reaches the target gets dropped.
    """
    rowptr, col, _ = kg.csr
    dist = torch.full((kg.num_nodes,), UNREACHABLE, dtype=torch.uint8)
    dist[target] = 0
    frontier = torch.tensor([target])
    for d in range(1, max_hops + 1):
        if frontier.numel() == 0:
            break
        spans = [col[rowptr[n] : rowptr[n + 1]] for n in frontier.tolist()]
        nbr = torch.unique(torch.cat(spans)) if spans else torch.empty(0, dtype=torch.long)
        frontier = nbr[dist[nbr] == UNREACHABLE]
        dist[frontier] = d
    return dist


@torch.no_grad()
def explain(
    model: RewindGNN,
    h: torch.Tensor,
    kg: KnowledgeGraph,
    source: int,
    target: int,
    max_hops: int = 3,
    beam_width: int = 32,
    top_k: int = 5,
) -> List[Path]:
    """Find the highest-confidence paths from `source` to `target`, scoring each hop with the GNN.

    Candidates are first restricted to nodes that can still reach the target within the hops that
    remain, then ranked by how likely the model thinks each step is. So the ranking is the model's
    judgement, while the reachability test is exact.
    """
    dist = hops_to(kg, target, max_hops)
    if dist[source] == UNREACHABLE:
        return []

    beams = [Path([source], [], 0.0)]
    found: List[Path] = []

    for depth in range(max_hops):
        remaining = max_hops - depth
        candidates = []
        for path in beams:
            node = path.nodes[-1]
            nbr, rel = kg.neighbors(node)
            if nbr.numel() == 0:
                continue
            nbr, rel = nbr.to(h.device), rel.to(h.device)
            edge_lp = F.logsigmoid(model.score(h, torch.full_like(nbr, node), rel, nbr))

            hit = nbr == target
            for lp, r in zip(edge_lp[hit].tolist(), rel[hit].tolist()):
                found.append(Path(path.nodes + [target], path.relations + [r], path.log_prob + lp))

            visited = torch.tensor(path.nodes, device=h.device)
            # Keep only neighbours that can still reach the target in the hops left after this one.
            open_ = ~hit & ~torch.isin(nbr, visited) & (dist[nbr.cpu()].to(h.device) <= remaining - 1)
            if not open_.any():
                continue
            priority = path.log_prob + edge_lp
            priority[~open_] = float("-inf")
            k = min(beam_width, int(open_.sum()))
            for i in priority.topk(k).indices.tolist():
                candidates.append((
                    float(priority[i]),
                    Path(path.nodes + [int(nbr[i])], path.relations + [int(rel[i])], path.log_prob + float(edge_lp[i])),
                ))
        candidates.sort(key=lambda c: c[0], reverse=True)
        beams = [step for _, step in candidates[:beam_width]]
        if not beams:
            break

    found.sort(key=lambda p: p.confidence, reverse=True)
    return found[:top_k]


def format_path(kg: KnowledgeGraph, path: Path) -> str:
    parts = [kg.names[path.nodes[0]]]
    for rel, node in zip(path.relations, path.nodes[1:]):
        parts.append(f"--[{kg.relations[rel]}]--> {kg.names[node]}")
    return " ".join(parts) + f"  (confidence {path.confidence:.2f})"
