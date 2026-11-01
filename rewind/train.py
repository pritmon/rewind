"""Link-prediction training on the blinded graph.

    python -m rewind.train                              # one GPU (or CPU)
    torchrun --nproc_per_node=4 -m rewind.train         # DDP; set train.fsdp: true to shard with FSDP
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn

from rewind.data.kg import KnowledgeGraph, hold_out
from rewind.models.gnn import RewindGNN


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class NegativeSampler:
    """Replaces each triple's tail with a random node of the same type (gene for gene, disease for disease).

    Same-type negatives are much harder than uniform ones, which is what forces the model to learn biology
    rather than node types.
    """

    def __init__(self, kg: KnowledgeGraph):
        vocab, self.type_ids = kg.node_type_ids
        self.sorted_nodes = torch.argsort(self.type_ids, stable=True)
        self.count = torch.bincount(self.type_ids, minlength=len(vocab))
        self.start = torch.cumsum(self.count, 0) - self.count

    def __call__(self, dst: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
        t = self.type_ids[dst].repeat_interleave(k)
        offset = (torch.rand(t.numel(), generator=generator) * self.count[t]).long()
        return self.sorted_nodes[self.start[t] + offset]


def wrap(model: nn.Module, tc: dict, device: torch.device, world_size: int) -> nn.Module:
    if world_size == 1:
        return model
    if tc.get("fsdp"):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, device_id=device)
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)


def full_state_dict(wrapped: nn.Module, model: nn.Module) -> dict:
    """Gathers sharded FSDP weights into one state dict. Every rank must call this."""
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    if isinstance(wrapped, FSDP):
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, config):
            return wrapped.state_dict()
    return model.state_dict()


def train(
    kg: KnowledgeGraph,
    cfg: dict,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    log: Callable[[str], None] = print,
) -> Tuple[nn.Module, RewindGNN]:
    """Returns (wrapped model, underlying RewindGNN)."""
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])  # identical initial weights on every rank
    model = RewindGNN(kg.num_nodes, kg.num_relations, **cfg["model"]).to(device)
    wrapped = wrap(model, tc, device, world_size)
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=tc["lr"])
    generator = torch.Generator().manual_seed(tc["seed"] + rank)  # different batches per rank
    sampler = NegativeSampler(kg)
    k, batch = tc["neg_per_pos"], tc["batch_size"]
    pos_weight = torch.tensor(float(k), device=device)

    wrapped.train()
    started = time.time()
    for step in range(1, tc["steps"] + 1):
        perm = torch.randperm(kg.num_edges, generator=generator)
        sup = perm[:batch]
        # Supervision edges are kept out of the message graph, otherwise the model just reads the answer.
        msg = perm[batch : batch + tc["message_edges"]]

        s, r, d = kg.edge_index[0, sup], kg.edge_type[sup], kg.edge_index[1, sup]
        neg = sampler(d, k, generator)
        src = torch.cat([s, s.repeat_interleave(k)])
        rel = torch.cat([r, r.repeat_interleave(k)])
        dst = torch.cat([d, neg])
        labels = torch.cat([torch.ones(len(s)), torch.zeros(len(neg))])

        logits = wrapped(
            kg.edge_index[:, msg].to(device), kg.edge_type[msg].to(device),
            src.to(device), rel.to(device), dst.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(device), pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if rank == 0 and (step % tc["log_every"] == 0 or step == tc["steps"]):
            elapsed = time.time() - started
            triples = step * len(labels) * world_size
            log(f"step {step:>6}  loss {loss.item():.4f}  {triples / elapsed:,.0f} triples/s")
    return wrapped, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")

    kg = KnowledgeGraph.load(cfg["data"]["processed"])
    hc = cfg["holdout"]
    ho = hold_out(kg, hc["genes"], hc["targets"], hc.get("expand_hops", 0))
    blinded = ho.graph
    if rank == 0:
        if ho.missing:
            print(f"warning: not found in graph: {', '.join(ho.missing)}")
        print(f"held out {len(ho.triples)} gene->target edges "
              f"(+{len(ho.related)} to {len(ho.related_ids)} related terms); "
              f"training on {blinded.num_edges:,} edges, world size {world_size}")

    wrapped, model = train(blinded, cfg, device, rank, world_size)
    state = full_state_dict(wrapped, model)
    if rank == 0:
        out = Path(cfg["train"]["checkpoint"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state, "config": cfg, "num_nodes": kg.num_nodes, "num_relations": kg.num_relations}, out)
        print(f"saved {out}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
