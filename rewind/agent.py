"""Claude as the reasoning layer: it turns questions into graph queries and explains what the GNN finds.

    python -m rewind.agent "Which genes could restore stemness in aged fibroblasts? Explain the top 3."
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List

import anthropic
import torch
from anthropic import beta_tool

from rewind.data.kg import GENE_TYPE, KnowledgeGraph, hold_out
from rewind.eval.rediscovery import already_linked, encode, load_checkpoint, score_genes
from rewind.models.gnn import RewindGNN
from rewind.reasoning.paths import explain, format_path

SYSTEM = """You are Rewind, a research assistant for cell-rejuvenation biology.

You reason over a biomedical knowledge graph (PrimeKG) through a graph neural network trained to predict missing links. Use the tools to ground every claim:
- search_nodes to find the exact node names for genes, biological processes, diseases or drugs.
- rank_genes to get the model's top candidate genes for one or more target terms.
- explain_link to get the multi-hop paths that support a candidate.

Present predictions as hypotheses with their model scores and supporting paths, not as established facts. Say when a path is weak (low confidence) or runs through a generic hub node. Keep answers concise."""


@dataclass
class Context:
    kg: KnowledgeGraph
    model: RewindGNN
    h: torch.Tensor
    relation: str


def build_tools(ctx: Context) -> list:
    @beta_tool
    def search_nodes(query: str, node_type: str = "") -> str:
        """Find knowledge-graph nodes whose name contains the query.

        Args:
            query: Text to match, e.g. "senescence" or "SOX".
            node_type: Optional filter: gene/protein, biological_process, disease, drug, pathway, anatomy.
        """
        hits = ctx.kg.search(query, node_type or None, limit=15)
        return json.dumps([{"name": n, "type": t} for _, n, t in hits])

    @beta_tool
    def rank_genes(target_terms: List[str], top_k: int = 15, exclude_known: bool = True) -> str:
        """Rank all genes by predicted link probability to the target terms (averaged across terms).

        Args:
            target_terms: Exact node names from search_nodes, e.g. ["stem cell population maintenance"].
            top_k: Number of genes to return.
            exclude_known: Drop genes the graph already links to these terms, leaving only new predictions.
        """
        target_ids, missing = ctx.kg.resolve(target_terms)
        if not target_ids:
            return json.dumps({"error": f"no matching nodes for {missing}; use search_nodes first"})
        genes, scores = score_genes(ctx.model, ctx.h, ctx.kg, target_ids, ctx.relation)
        if exclude_known:
            scores = scores.masked_fill(already_linked(ctx.kg, genes, target_ids), float("-inf"))
        top = scores.topk(min(top_k, len(scores)))
        ranked = [{"gene": ctx.kg.names[int(genes[i])], "score": round(float(s), 4)}
                  for s, i in zip(top.values, top.indices) if s > float("-inf")]
        return json.dumps({"ranked": ranked, "missing_terms": missing})

    @beta_tool
    def explain_link(gene: str, target: str, max_paths: int = 3) -> str:
        """Find the highest-confidence multi-hop paths (up to 3 hops) from a gene to a target node.

        Args:
            gene: Gene symbol, e.g. "SOX2".
            target: Exact target node name.
            max_paths: Number of paths to return.
        """
        try:
            source = ctx.kg.lookup(gene, GENE_TYPE)
            dest = ctx.kg.lookup(target)
        except KeyError as e:
            return json.dumps({"error": str(e)})
        paths = explain(ctx.model, ctx.h, ctx.kg, source, dest, top_k=max_paths)
        return json.dumps({"paths": [format_path(ctx.kg, p) for p in paths] or ["no path within 3 hops"]})

    return [search_nodes, rank_genes, explain_link]


def ask(question: str, ctx: Context, model: str, effort: str) -> str:
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        tools=build_tools(ctx),
        messages=[{"role": "user", "content": question}],
        output_config={"effort": effort},
        # Biology questions can trip safety classifiers; let the API retry on a fallback model instead of stopping.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    final = None
    for message in runner:
        final = message
    if final is None:
        return "(no response)"
    if final.stop_reason == "refusal":
        return "The model declined this request."
    return "".join(block.text for block in final.content if block.type == "text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--checkpoint", default="checkpoints/rewind.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-graph", action="store_true",
                        help="Reason over the full graph instead of the blinded one the model was trained on.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    kg = KnowledgeGraph.load(cfg["data"]["processed"])
    if not args.full_graph:
        hc = cfg["holdout"]
        kg = hold_out(kg, hc["genes"], hc["targets"], hc.get("expand_hops", 0)).graph
    ctx = Context(kg, model, encode(model, kg, device), cfg["holdout"]["relation"])
    print(ask(args.question, ctx, cfg["agent"]["model"], cfg["agent"]["effort"]))


if __name__ == "__main__":
    main()
