"""Time-respecting neighbor sampling over the transaction-entity graph.

The graph is bipartite: transaction -- relation r --> entity (a card, a
device, an email domain...). For a target transaction at time t, its 2-hop
neighborhood through relation r is "the K most recent OTHER transactions that
used the same entity, strictly before t". Using only the past is what makes
offline evaluation match what a live system can actually see, and it removes
the future-neighbor leakage that a static full-graph setup has.

Implementation: per relation, sort rows by (entity, time) once, then answer
every query with one vectorized np.searchsorted. No graph library needed.
"""
import numpy as np

_SHIFT = np.int64(1) << np.int64(34)   # times fit in 34 bits (~544 years of seconds)


class TemporalNeighborIndex:
    def __init__(self, entity_ids, times):
        """
        entity_ids: int64 [N, R] graph entity id per row and relation (0 = none)
        times:      int64 [N] event time of each row
        """
        self.n, self.r = entity_ids.shape
        self.times = times.astype(np.int64)
        self.keys, self.rows, self.starts = [], [], []
        for j in range(self.r):
            e = entity_ids[:, j].astype(np.int64)
            key = e * _SHIFT + self.times
            order = np.argsort(key, kind="stable")
            self.keys.append(key[order])
            self.rows.append(order.astype(np.int64))
            self.starts.append(e[order])

    def query(self, entity_ids, times, k):
        """
        entity_ids: [B, R] graph ids of the targets, times: [B]
        returns nbr_rows [B, R, k] (-1 = padding), nbr_mask [B, R, k] bool,
                hist_count [B, R] total prior txns on that entity (uncapped)
        """
        b = entity_ids.shape[0]
        nbr = np.full((b, self.r, k), -1, dtype=np.int64)
        count = np.zeros((b, self.r), dtype=np.int64)
        offs = np.arange(k, dtype=np.int64)[None, :] - k       # -k..-1
        t = times.astype(np.int64)
        for j in range(self.r):
            e = entity_ids[:, j].astype(np.int64)
            keys = self.keys[j]
            lo = np.searchsorted(keys, e * _SHIFT, side="left")        # first row of entity
            hi = np.searchsorted(keys, e * _SHIFT + t, side="left")    # strictly before t
            valid_e = e > 0
            count[:, j] = np.where(valid_e, hi - lo, 0)
            pos = hi[:, None] + offs                                   # last k before t
            ok = (pos >= lo[:, None]) & valid_e[:, None]
            pos = np.clip(pos, 0, max(len(keys) - 1, 0))
            nbr[:, j, :] = np.where(ok, self.rows[j][pos], -1)
        return nbr, nbr >= 0, count
