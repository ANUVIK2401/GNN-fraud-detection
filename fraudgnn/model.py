"""Temporal heterogeneous GNN with relation-level attention.

For a target transaction:
  1. Encode its own features:                    h_t = MLP(x_t)
  2. For each relation r (card1, device, email...), build the entity state
     from the K most recent prior transactions on that entity:
        m = ReLU(W_r h_nbr + w_r * log(1 + dt))     time-aware message
        h_e = ReLU(emb_r[entity] + mean(m) + c_r * log(1 + history_count))
  3. Attention over relations (HAN-style semantic attention) gives one
     context vector and per-relation weights, which double as an explanation.
  4. Classifier on [h_t, context].

Step 1 is shared between targets and neighbors, so at serving time neighbor
encodings are computed once at ingest and cached (see serve/scorer.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalHeteroGNN(nn.Module):
    def __init__(self, in_dim, vocab_sizes, hidden=64, dropout=0.2):
        super().__init__()
        self.r = len(vocab_sizes)
        self.hidden = hidden
        self.txn_enc = nn.Sequential(
            nn.Linear(in_dim, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
        )
        self.ent_emb = nn.ModuleList(nn.Embedding(n, hidden, padding_idx=0) for n in vocab_sizes)
        r, h = self.r, hidden
        self.w_msg = nn.Parameter(torch.randn(r, h, h) / h ** 0.5)
        self.w_dt = nn.Parameter(torch.zeros(r, h))
        self.b_msg = nn.Parameter(torch.zeros(r, h))
        self.w_cnt = nn.Parameter(torch.zeros(r, h))
        self.w_rel = nn.Parameter(torch.randn(r, h, h) / h ** 0.5)
        self.att = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 1),
        )

    def encode(self, x):
        return self.txn_enc(x)

    def forward(self, x, h_nbr, nbr_mask, dt, ent_ids, rel_mask, hist_count):
        """
        x [B,D]; h_nbr [B,R,K,H] (already encoded); nbr_mask [B,R,K] bool;
        dt [B,R,K] seconds since each neighbor; ent_ids [B,R] embedding ids;
        rel_mask [B,R] bool, entity present; hist_count [B,R] prior txn count.
        Returns logits [B], relation attention [B,R], embedding [B,2H].
        """
        h_t = self.encode(x)
        m = torch.einsum("brkh,rhg->brkg", h_nbr, self.w_msg)
        m = m + torch.log1p(dt.clamp(min=0)).unsqueeze(-1) * self.w_dt[None, :, None, :] / 10.0
        m = F.relu(m + self.b_msg[None, :, None, :])
        w = nbr_mask.float().unsqueeze(-1)
        agg = (m * w).sum(2) / w.sum(2).clamp(min=1.0)                       # [B,R,H]

        emb = torch.stack([self.ent_emb[j](ent_ids[:, j]) for j in range(self.r)], dim=1)
        cnt = torch.log1p(hist_count.float()).unsqueeze(-1) * self.w_cnt[None] / 5.0
        h_e = self.drop(F.relu(emb + agg + cnt))
        z = torch.einsum("brh,rhg->brg", h_e, self.w_rel)                    # [B,R,H]

        scores = self.att(z).squeeze(-1).masked_fill(~rel_mask, -1e9)
        alpha = torch.softmax(scores, dim=1) * rel_mask.float()
        alpha = alpha / alpha.sum(1, keepdim=True).clamp(min=1e-9)
        ctx = (alpha.unsqueeze(-1) * z).sum(1)

        out = torch.cat([h_t, ctx], dim=1)
        return self.cls(out).squeeze(-1), alpha, out
