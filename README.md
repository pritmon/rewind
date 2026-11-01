# Rewind

**Can an AI rediscover a Nobel Prize on its own?**

In 2006 Shinya Yamanaka showed that four genes (POU5F1/OCT4, SOX2, KLF4, MYC) can turn an adult cell back into a young, stem-like one. Rewind deletes everything a biomedical knowledge graph knows about how those four genes connect to stemness and aging, trains a graph neural network on what is left, and asks: *which genes would rejuvenate a cell?* If the held-out Yamanaka factors come back near the top, the model's other, untested suggestions are worth a closer look.

An LLM agent (Claude) sits on top. You ask questions in plain English, and it calls the GNN to rank candidates and pull out the multi-hop paths that explain each prediction.

```
"What could push an old cell back toward a stem-like state?"
        │
        ▼
  Claude agent ── search_nodes / rank_genes / explain_link ──►  R-GCN over PrimeKG
        │                                                        (129k nodes, ~8M edges)
        ▼
  ranked genes + mechanistic paths, e.g.
  SOX2 ─protein_protein→ NANOG ─bioprocess_protein→ stem cell population maintenance
```

## How it works

| Piece | File | What it does |
|---|---|---|
| Knowledge graph | `rewind/data/kg.py` | Loads [PrimeKG](https://github.com/mims-harvard/PrimeKG) and builds the held-out split |
| GNN | `rewind/models/gnn.py` | Relational GCN (basis decomposition) + DistMult link decoder |
| Training | `rewind/train.py` | Link prediction with type-aware negatives; single GPU, DDP or FSDP via `torchrun` |
| The magic test | `rewind/eval/rediscovery.py` | Ranks the held-out genes against every other gene, and against baselines |
| Explanations | `rewind/reasoning/paths.py` | Beam search for the highest-scoring multi-hop paths between a gene and a term |
| Agent | `rewind/agent.py` | Claude tool-use loop over the three graph tools |

## Quickstart

```bash
pip install -e ".[dev]"
bash scripts/download_primekg.sh                         # ~1 GB download + preprocessing
python -m rewind.train                                   # 1 GPU
torchrun --nproc_per_node=4 -m rewind.train              # 4 GPUs (DDP; set train.fsdp: true for FSDP)
python -m rewind.eval.rediscovery                        # the magic test
python -m rewind.agent "Which genes could restore stemness in aged fibroblasts? Explain the top 3."
```

The agent needs Anthropic credentials (`ANTHROPIC_API_KEY` or `ant auth login`).

## Results so far

A laptop-scale run (CPU, 2,000 steps, 64-dim model: roughly 1% of the compute the GPU config uses). Three of the four Yamanaka factors have edges to the target terms in PrimeKG and are held out; MYC has none, so it is not part of the test.

| Gene | Model rank | Degree baseline | Random |
|---|---|---|---|
| KLF4 | **194** (top 0.7%) | 1,099 | ~13,766 |
| SOX2 | **438** (top 1.6%) | 1,033 | ~13,766 |
| POU5F1 | **564** (top 2.0%) | 8,287 | ~13,766 |

Ranked among 27,531 genes with no known link to the targets, with the leak control on. All three beat the degree baseline, so the model is not just reciting famous genes; three independent genes all landing in the top 2% by chance is roughly a 1-in-60,000 event. None reaches hits@100 yet, and training loss was still falling when the run ended (1.11 → 0.56), so these numbers are a floor, not a ceiling.

The model's own top new predictions (genes with no recorded link to the targets) include SMAD3, TGFBR2 and CCND1. TGF-β inhibition is a known booster of reprogramming efficiency, which is a reassuring sanity check rather than a discovery.

Reproduce with:

```bash
python -m rewind.train --config configs/laptop.yaml     # ~45 min on an M1 Pro
python -m rewind.eval.rediscovery --checkpoint checkpoints/laptop.pt --paths
```

### Known limitations

- **Explanation paths favour hubs.** Paths are ranked by the model's confidence, and generic high-degree nodes score well. SOX2's top paths route through brain-tumour nodes to CTNNB1 (β-catenin, a genuine Wnt/pluripotency hub), while the more interpretable `SOX2 → NANOG → target` route ranks lower. A degree penalty or a path-diversity constraint is the obvious next step.
- **Annotation, not biology.** Recovering a Gene Ontology edge is a sanity check on the reasoning, not evidence that an intervention would rejuvenate a cell. That is what phase 2 is for.
- **One seed, no error bars.** Every number above is a single run at seed 0.

## Keeping the result honest

A rediscovery claim is only as good as its controls:

- **Filtered ranking.** Genes that already link to the target terms in the training graph are taken out of the ranking. Only genes the model could not simply look up are compared.
- **Degree baseline.** Yamanaka factors are well-studied hub genes. Rewind reports how a "rank by number of connections" baseline does, so you can see whether the GNN beats popularity.
- **Related-term control (`holdout.expand_hops`).** A near-synonym of the target can hand over the answer: before this control, the model explained SOX2 as `SOX2 → neuronal stem cell population maintenance → stem cell population maintenance`, which is the hidden fact in different words. The holdout now widens the targets one hop along ontology edges (`bioprocess_bioprocess` and friends) and removes those edges too. On PrimeKG that reaches 13 related terms and removes exactly that one leak edge. The report prints what was removed, so the control is visible rather than claimed.
- **The control is deliberately one-sided.** Only the held-out genes lose their related-term edges; the 27,000 competing genes keep theirs. That handicaps the held-out genes, so a good rank under this setting understates the model rather than flattering it.
- **Read the paths, don't just trust the rank.** `--paths` prints the best supporting path per gene so a reviewer can check that no explanation is a restatement of the deleted fact.

## Roadmap

1. **Phase 1 (this repo):** GNN + held-out rediscovery + agent.
2. **Phase 2, cell-level check:** embed old and young cells from Tabula Muris Senis with a single-cell foundation model (Geneformer or scGPT). Simulate perturbing the top candidates in silico, and measure whether old cells move toward the young cluster.
3. **Phase 3, LLM ↔ GNN fusion:** replace tool calls with soft graph tokens, so a fine-tuned LLM attends to GNN embeddings directly (multi-node FSDP).
4. **Write-up:** 4-page workshop paper (for example MLGenX at ICLR or LMRL at NeurIPS).

## Tests

```bash
pytest
```

The tests run the whole pipeline on a tiny synthetic graph on CPU in a few seconds.
