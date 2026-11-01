"""The magic test: after training on the blinded graph, where do the held-out genes rank?"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import torch

from rewind.data.kg import GENE_TYPE, KnowledgeGraph, hold_out
from rewind.models.gnn import RewindGNN
from rewind.reasoning.paths import explain, format_path


def load_checkpoint(path: str, device: torch.device) -> Tuple[RewindGNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RewindGNN(ckpt["num_nodes"], ckpt["num_relations"], **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def encode(model: RewindGNN, kg: KnowledgeGraph, device: torch.device) -> torch.Tensor:
    model.eval()
    return model.encode(kg.edge_index.to(device), kg.edge_type.to(device))


@torch.no_grad()
def score_genes(
    model: RewindGNN, h: torch.Tensor, kg: KnowledgeGraph, target_ids: Sequence[int], relation: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean link probability from every gene to the target terms. Returns (gene ids, scores) on CPU."""
    genes = kg.nodes_of_type(GENE_TYPE).to(h.device)
    rel = torch.full_like(genes, kg.relation_id(relation))
    probs = torch.stack([torch.sigmoid(model.score(h, genes, rel, torch.full_like(genes, t))) for t in target_ids])
    return genes.cpu(), probs.mean(0).cpu()


def already_linked(kg: KnowledgeGraph, genes: torch.Tensor, target_ids: Sequence[int]) -> torch.Tensor:
    """Genes that the graph already connects to a target. They are excluded from ranking ("filtered" setting)."""
    src, dst = kg.edge_index
    linked = src[torch.isin(dst, torch.tensor(list(target_ids)))]
    return torch.isin(genes, linked)


def rank_among(scores: torch.Tensor, candidates: torch.Tensor, index: int) -> int:
    """1-based rank of scores[index] among the candidate genes (ties count against it)."""
    return int((scores[candidates] >= scores[index]).sum())


def evaluate(
    model: RewindGNN, kg: KnowledgeGraph, cfg: dict, device: torch.device, with_paths: bool = False
) -> Dict:
    hc = cfg["holdout"]
    ho = hold_out(kg, hc["genes"], hc["targets"], hc.get("expand_hops", 0))
    blinded = ho.graph
    h = encode(model, blinded, device)
    genes, scores = score_genes(model, h, blinded, ho.seed_ids, hc["relation"])

    candidates = ~already_linked(blinded, genes, ho.seed_ids)
    position = {int(g): i for i, g in enumerate(genes)}
    degree = blinded.degree()[genes].float()

    rows: List[Dict] = []
    for gene_id in ho.gene_ids:
        i = position[gene_id]
        row = {
            "gene": kg.names[gene_id],
            "model_rank": rank_among(scores, candidates, i),
            "degree_rank": rank_among(degree, candidates, i),
            "held_edges": int((ho.triples[:, 0] == gene_id).sum()),
        }
        if with_paths:
            paths = explain(model, h, blinded, gene_id, ho.seed_ids[0], top_k=1)
            row["path"] = format_path(blinded, paths[0]) if paths else "no path within 3 hops"
        rows.append(row)
    return {
        "rows": rows,
        "num_candidates": int(candidates.sum()),
        "num_held_edges": len(ho.triples),
        "num_related_removed": len(ho.related),
        "num_related_terms": len(ho.related_ids),
        "expand_hops": hc.get("expand_hops", 0),
        "missing": ho.missing,
    }


def hits_at(rows: List[Dict], key: str, k: int) -> int:
    return sum(r[key] <= k for r in rows)


def report(result: Dict) -> str:
    rows, n = result["rows"], result["num_candidates"]
    lines = []
    if result["missing"]:
        lines.append(f"warning: not found in graph: {', '.join(result['missing'])}")
    if not rows:
        lines.append("No edges were held out, so check holdout.genes and holdout.targets against the graph's node names.")
        return "\n".join(lines)
    lines.append(f"Rediscovery: {len(rows)} held-out genes ({result['num_held_edges']} edges), "
                 f"ranked among {n:,} genes with no known link to the targets\n")
    lines.append(f"{'gene':<10}{'model rank':>18}{'degree baseline':>20}{'random':>10}")
    for r in rows:
        pct = 100 * r["model_rank"] / n
        lines.append(f"{r['gene']:<10}{r['model_rank']:>9,} (top {pct:4.1f}%){r['degree_rank']:>20,}{(n + 1) // 2:>10,}")
    lines.append("")
    for k in (10, 50, 100):
        lines.append(f"hits@{k:<4} model {hits_at(rows, 'model_rank', k)}/{len(rows)}   "
                     f"degree {hits_at(rows, 'degree_rank', k)}/{len(rows)}")

    hops = result["expand_hops"]
    lines.append("")
    if hops:
        lines.append(f"Leak control: targets widened {hops} hop(s) to {result['num_related_terms']} related terms; "
                     f"{result['num_related_removed']} extra gene->term edges removed.")
    else:
        lines.append("Leak control: OFF (expand_hops: 0) - a near-synonym of a target may still reveal the answer.")
    if any("path" in r for r in rows):
        lines.append("Supporting path for each gene (check none is a restatement of the hidden fact):")
        for r in rows:
            lines.append(f"  {r['gene']:<8} {r['path']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank the held-out genes after training on the blinded graph.")
    parser.add_argument("--checkpoint", default="checkpoints/rewind.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--paths", action="store_true", help="also print the best supporting path per gene")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    kg = KnowledgeGraph.load(cfg["data"]["processed"])
    print(report(evaluate(model, kg, cfg, device, with_paths=args.paths)))


if __name__ == "__main__":
    main()
