"""Relational GCN encoder with a DistMult link decoder, in plain PyTorch."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RelGraphConv(nn.Module):
    """R-GCN layer (Schlichtkrull et al. 2018) with basis decomposition: W_r = sum_b coef[r, b] * bases[b].

    Messages are aggregated per basis before projecting, so peak memory is O(E * dim)
    instead of O(E * num_bases * dim). That matters on PrimeKG's ~8M edges.
    """

    def __init__(self, dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.bases = nn.Parameter(torch.empty(num_bases, dim, dim))
        self.coef = nn.Parameter(torch.empty(num_relations, num_bases))
        self.self_loop = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coef)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        deg = torch.zeros(h.size(0), device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        norm = deg.clamp(min=1).unsqueeze(1)
        h_src = h[src]
        out = self.self_loop(h)
        for b in range(self.bases.size(0)):
            agg = torch.zeros_like(h).index_add_(0, dst, h_src * self.coef[edge_type, b : b + 1])
            out = out + (agg / norm) @ self.bases[b]
        return out


class RewindGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, dim)
        self.layers = nn.ModuleList(RelGraphConv(dim, num_relations, num_bases) for _ in range(num_layers))
        self.rel_diag = nn.Parameter(torch.ones(num_relations, dim) + 0.1 * torch.randn(num_relations, dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.embedding.weight)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.embedding.weight
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_type)
            if i < len(self.layers) - 1:
                h = F.dropout(F.relu(h), self.dropout, self.training)
        return h

    def score(self, h: torch.Tensor, src: torch.Tensor, rel: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """DistMult logit for each (src, rel, dst) triple."""
        return (h[src] * self.rel_diag[rel] * h[dst]).sum(-1)

    def forward(self, edge_index, edge_type, src, rel, dst) -> torch.Tensor:
        # One entry point for training, so DDP/FSDP hooks fire on every parameter access.
        return self.score(self.encode(edge_index, edge_type), src, rel, dst)
