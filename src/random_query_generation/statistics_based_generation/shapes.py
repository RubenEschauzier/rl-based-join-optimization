"""Query shapes as abstract graphs: nodes (later entities or variables) and directed edges
(later triple patterns, subject -> object). Each builder returns exactly `n` edges.

The eight shapes of FICE / G-CARE, sized 4..16 patterns:

    star       one center, n spokes (mostly outgoing, as subject stars; some incoming)
    path       a chain of n edges, each edge in a random direction (WatDiv "linear")
    tree       a random tree: every new edge hangs a new node off an existing one
    snowflake  a center with 2-4 arms, each arm node carrying a small star (WatDiv "snowflake")
    cycle      a ring plus max(1, n // 4) stems
    diamond    two endpoints joined by parallel 2-paths (G-CARE "petal") plus max(1, n // 4) stems
    flower     a center with triangle petals through it plus max(1, n // 4) stems (G-CARE "flower")

Cycle nodes never become constants (instantiate.binding_plan): a constant is no join variable,
so binding a ring node would turn the cycle into a path. The stems are where constants go.
    path_star  a chain whose nodes carry small stars ("path+star" in FICE)

Directions are random where the shape does not fix them, so joins mix subject-subject,
subject-object and object-object positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SHAPES = ("star", "path", "tree", "snowflake", "cycle", "diamond", "flower", "path_star")


@dataclass
class QueryShape:
    kind: str
    n_nodes: int
    edges: list = field(default_factory=list)     # (subject node, object node)
    root: int = 0                                 # where instantiation starts

    @property
    def n_edges(self):
        return len(self.edges)

    def degree(self):
        degree = np.zeros(self.n_nodes, dtype=int)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return degree


class _Builder:
    def __init__(self, kind, rng, outgoing_bias=0.5):
        self.kind, self.rng, self.outgoing_bias = kind, rng, outgoing_bias
        self.n_nodes, self.edges = 0, []

    def node(self):
        self.n_nodes += 1
        return self.n_nodes - 1

    def edge(self, u, v, outgoing=None):
        """Edge between u and v; direction u -> v with probability outgoing_bias (or forced)."""
        forward = self.rng.random() < self.outgoing_bias if outgoing is None else outgoing
        self.edges.append((u, v) if forward else (v, u))

    def spoke(self, u, outgoing=None):
        v = self.node()
        self.edge(u, v, outgoing)
        return v

    def chain(self, start, length, end=None):
        """`length` edges from start; to `end` if given (closing a cycle), else to new nodes."""
        current = start
        for i in range(length):
            last = i == length - 1
            nxt = end if (last and end is not None) else self.node()
            self.edge(current, nxt)
            current = nxt
        return current

    def build(self, root=0):
        return QueryShape(self.kind, self.n_nodes, self.edges, root)


def build_shape(kind, n, rng):
    if n < 2:
        raise ValueError("shapes need at least 2 edges")
    b = _Builder(kind, rng)
    if kind == "star":
        center = b.node()
        for _ in range(n):
            b.spoke(center, outgoing=rng.random() < 0.8)
        return b.build(center)
    if kind == "path":
        b.chain(b.node(), n)
        return b.build(int(rng.integers(0, n + 1)))
    if kind == "tree":
        b.node()
        for _ in range(n):
            b.spoke(int(rng.integers(0, b.n_nodes)))
        return b.build(int(rng.integers(0, b.n_nodes)))
    if kind == "snowflake":
        center = b.node()
        arms = int(min(max(2, n // 3), 4))
        remaining = n - arms
        arm_nodes = [b.spoke(center) for _ in range(arms)]
        for k in range(remaining):
            b.spoke(arm_nodes[k % arms], outgoing=rng.random() < 0.8)
        return b.build(center)
    stems = max(1, n // 4)          # cycle nodes stay variables: stems carry the constants
    if kind == "cycle":
        start = b.node()
        b.chain(start, max(3, n - stems), end=start)
        while len(b.edges) < n:
            b.spoke(int(rng.integers(0, b.n_nodes)))
        return b.build(start)
    if kind == "diamond":
        a, z = b.node(), b.node()
        for _ in range(max(2, (n - stems) // 2)):
            middle = b.node()
            b.edge(a, middle)
            b.edge(middle, z)
        while len(b.edges) < n:
            b.spoke(a if rng.random() < 0.5 else z)
        return b.build(a)
    if kind == "flower":
        center = b.node()
        petals = max(1, (n - stems) // 3)      # every petal is a triangle: 3 edges
        for _ in range(petals):
            b.chain(center, 3, end=center)
        while len(b.edges) < n:
            b.chain(center, min(int(rng.integers(1, 3)), n - len(b.edges)))
        return b.build(center)
    if kind == "path_star":
        length = max(2, n // 2)
        chain_nodes = [b.node()]
        for _ in range(length):
            chain_nodes.append(b.spoke(chain_nodes[-1]))
        while len(b.edges) < n:
            b.spoke(chain_nodes[int(rng.integers(0, len(chain_nodes)))], outgoing=rng.random() < 0.8)
        return b.build(chain_nodes[0])
    raise ValueError(f"unknown shape {kind!r}; known: {SHAPES}")
