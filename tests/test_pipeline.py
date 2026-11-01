"""End-to-end checks on a tiny synthetic PrimeKG-shaped graph (CPU, a few seconds)."""
import random

import pandas as pd
import torch

from rewind.data.kg import GENE_TYPE, expand_terms, from_primekg, hold_out
from rewind.eval.rediscovery import evaluate, report
from rewind.reasoning.paths import explain, format_path
from rewind.train import NegativeSampler, train

HELD_GENES = ["POU5F1", "SOX2"]
TARGETS = ["stem cell population maintenance"]
RELATED = "neuronal stem cell population maintenance"  # ontology child of TARGETS[0]
CONFIG = {
    "holdout": {"genes": HELD_GENES, "targets": TARGETS, "relation": "bioprocess_protein",
                "expand_hops": 1},
    "model": {"dim": 16, "num_layers": 2, "num_bases": 2, "dropout": 0.0},
    "train": {"steps": 150, "batch_size": 64, "neg_per_pos": 2, "message_edges": 400,
              "lr": 0.01, "seed": 0, "fsdp": False, "log_every": 1000},
}


def toy_primekg() -> pd.DataFrame:
    """A stemness module (OSKM-like genes + partners, all wired together) and unrelated background genes."""
    rng = random.Random(0)
    nodes = {}

    def node(name, node_type):
        return nodes.setdefault((name, node_type), len(nodes))

    stem = HELD_GENES + ["NANOG", "LIN28A", "SALL4", "DPPA4"]
    background = [f"G{i}" for i in range(40)]
    rows = []

    def edge(rel, a, at, b, bt):
        rows.append({"relation": rel, "display_relation": rel,
                     "x_index": node(a, at), "x_name": a, "x_type": at,
                     "y_index": node(b, bt), "y_name": b, "y_type": bt})

    for i, a in enumerate(stem):
        for b in stem[i + 1:]:
            edge("protein_protein", a, GENE_TYPE, b, GENE_TYPE)
        edge("bioprocess_protein", a, GENE_TYPE, TARGETS[0], "biological_process")
    # The leak: a child term that restates the target, with SOX2 attached to it.
    edge("bioprocess_bioprocess", RELATED, "biological_process", TARGETS[0], "biological_process")
    edge("bioprocess_protein", "SOX2", GENE_TYPE, RELATED, "biological_process")
    for g in background:
        for h in rng.sample(background, 3):
            if h != g:
                edge("protein_protein", g, GENE_TYPE, h, GENE_TYPE)
        edge("bioprocess_protein", g, GENE_TYPE, f"process {rng.randint(0, 4)}", "biological_process")
    return pd.DataFrame(rows)


def test_graph_is_symmetric_and_deduplicated():
    kg = from_primekg(toy_primekg())
    triples = set(zip(kg.edge_index[0].tolist(), kg.edge_type.tolist(), kg.edge_index[1].tolist()))
    assert len(triples) == kg.num_edges
    assert all((d, r, s) in triples for s, r, d in triples)


def test_hold_out_removes_both_directions():
    kg = from_primekg(toy_primekg())
    ho = hold_out(kg, HELD_GENES + ["NOT_A_GENE"], TARGETS)
    assert ho.missing == ["NOT_A_GENE"]
    assert len(ho.triples) == 2
    target = kg.lookup(TARGETS[0])
    genes = torch.tensor([kg.lookup(g) for g in HELD_GENES])
    src, dst = ho.graph.edge_index
    assert not ((torch.isin(src, genes) & (dst == target)) | ((src == target) & torch.isin(dst, genes))).any()
    assert ho.graph.num_edges == kg.num_edges - 4


def test_expansion_closes_the_related_term_leak():
    kg = from_primekg(toy_primekg())
    related, target = kg.lookup(RELATED), kg.lookup(TARGETS[0])
    assert expand_terms(kg, [target], hops=1) == [related]

    plain = hold_out(kg, HELD_GENES, TARGETS, expand_hops=0)
    src, dst = plain.graph.edge_index
    sox2 = kg.lookup("SOX2", GENE_TYPE)
    assert bool(((src == sox2) & (dst == related)).any()), "leak edge should survive without expansion"

    strict = hold_out(kg, HELD_GENES, TARGETS, expand_hops=1)
    src, dst = strict.graph.edge_index
    assert not bool(((src == sox2) & (dst == related)).any()), "expansion must remove the leak edge"
    assert strict.related_ids == [related]
    assert len(strict.related) == 1                      # SOX2 -> related
    assert len(strict.triples) == len(plain.triples)     # the rediscovery target is unchanged
    assert strict.graph.num_edges == plain.graph.num_edges - 2


def test_negative_sampler_keeps_node_type():
    kg = from_primekg(toy_primekg())
    _, type_ids = kg.node_type_ids
    dst = kg.edge_index[1, :50]
    neg = NegativeSampler(kg)(dst, 3, torch.Generator().manual_seed(0))
    assert torch.equal(type_ids[neg], type_ids[dst].repeat_interleave(3))


def test_train_rediscover_and_explain():
    kg = from_primekg(toy_primekg())
    blinded = hold_out(kg, HELD_GENES, TARGETS, CONFIG["holdout"]["expand_hops"]).graph
    device = torch.device("cpu")
    _, model = train(blinded, CONFIG, device, log=lambda _: None)

    result = evaluate(model, kg, CONFIG, device, with_paths=True)
    print(report(result))
    assert {r["gene"] for r in result["rows"]} == set(HELD_GENES)
    # The held-out genes sit inside the stemness module, so they should beat random among 40 background genes.
    assert all(r["model_rank"] <= result["num_candidates"] // 2 for r in result["rows"])
    assert result["num_related_removed"] == 1 and result["expand_hops"] == 1
    assert all(r["path"] for r in result["rows"])

    h = model.encode(blinded.edge_index, blinded.edge_type)
    paths = explain(model, h, blinded, kg.lookup("SOX2", GENE_TYPE), kg.lookup(TARGETS[0]))
    assert paths, "expected a path through the stemness module"
    assert format_path(blinded, paths[0]).startswith("SOX2 --[")
