import math
import random
from typing import Dict, List, Set, Tuple, Optional, Callable
import re


class JoinPlan:
    """Represents a join plan with cost estimation"""

    def __init__(self, left: Optional['JoinPlan'], right: Optional['JoinPlan'],
                 entries: Set[int], estimated_size: float, left_deep: bool, cardinality_is_log_scale: bool):
        self.left = left
        self.right = right
        self.entries = entries
        self.estimated_size = estimated_size

        # Turn back to non-log scale
        if cardinality_is_log_scale:
            self.estimated_size = math.exp(self.estimated_size)

        self.left_deep = left_deep
        self.is_leaf = self.left is None or self.right is None
        self.cost = self._calculate_cost()
        self.cardinality_is_log_scale = cardinality_is_log_scale

    def _calculate_cost(self) -> float:
        """
        Calculate cost using a simple cost function
        Returns: calculated cost
        """
        # If either left or right is undefined, we are triple pattern
        if self.is_leaf:
            return self.estimated_size

        if self.left_deep:
            # Left deep plan cost is equivalent to Q-learning reward currently.
            return self.left.cost + self.estimated_size
        else:
            if self.left.is_leaf:
                return self.right.cost + self.estimated_size
            elif self.right.is_leaf:
                return self.left.cost + self.estimated_size
            else:
                return self.left.cost + self.right.cost + self.estimated_size
            # join_cost = (self.left.estimated_size * self.right.estimated_size) + self.estimated_size
            # return join_cost + self.left.cost + self.right.cost

    def get_order(self) -> list[int]:
        """
        Return a list of entry indices in the order they are joined (left-deep plan).
        """
        if self.is_leaf:
            # Leaf node: just return its entries
            return list(self.entries)

        if self.left_deep:
            # Left-deep: get order from left subtree, then append right leaf
            order = self.left.get_order() if self.left else []
            if self.right:
                order += list(self.right.entries)
            return order
        else:
            raise ValueError("Get order not defined for non-left deep")

    def __str__(self) -> str:
        return self._build_tree_string()

    def _build_tree_string(self, indent: int = 0, prefix: str = "") -> str:
        """
        Build tree string representation recursively
        """
        # Current node representation
        if not self.left or not self.right:
            # Leaf node (triple pattern)
            node_str = f"TP{sorted(self.entries)} (size: {self.estimated_size:.2f}, cost: {self.cost:.2f})"
        else:
            # Internal join node
            join_type = "LD" if self.left_deep else "BT"  # Left-Deep or Bushy-Tree
            node_str = f"Join[{join_type}] {sorted(self.entries)} (size: {self.estimated_size:.2f}, cost: {self.cost:.2f})"

        result = " " * indent + prefix + node_str

        # Add children if they exist
        if self.left and self.right:
            result += "\n" + self.left._build_tree_string(indent + 4, "├── L: ")
            result += "\n" + self.right._build_tree_string(indent + 4, "└── R: ")

        return result


def _bits(mask: int) -> List[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def neighbour_masks(adjacency_list: Dict[int, List[int]], n: int) -> List[int]:
    """Per vertex, the bitmask of its neighbours."""
    masks = []
    for v in range(n):
        mask = 0
        for u in adjacency_list.get(v, ()):
            mask |= 1 << u
        masks.append(mask & ~(1 << v))
    return masks


def connected_subgraphs(n: int, nbr: List[int]) -> Dict[int, int]:
    """Every connected subgraph (csg, as a bitmask) -> its neighbourhood, by EnumerateCsg
    (Moerkotte & Neumann, VLDB 2006): from each vertex v (descending), grow connected sets with
    vertices > v only (B_v = {0..v} excluded), extending by non-empty subsets of the neighbourhood.
    Each csg is produced exactly once."""
    hood: Dict[int, int] = {}

    def neighbourhood(mask):
        reach = 0
        for v in _bits(mask):
            reach |= nbr[v]
        return reach & ~mask

    stack = []
    for v in range(n - 1, -1, -1):
        vb = 1 << v
        hood[vb] = nbr[v] & ~vb
        stack.append((vb, (vb << 1) - 1))
        while stack:
            s, x = stack.pop()
            frontier = hood[s] & ~x
            sub = 0
            while True:                                  # all non-empty subsets of the frontier
                sub = (sub - frontier) & frontier
                if not sub:
                    break
                grown = s | sub
                hood[grown] = neighbourhood(grown)
                stack.append((grown, x | frontier))
    return hood


def csg_cmp_pairs(n: int, nbr: List[int], hood: Optional[Dict[int, int]] = None):
    """All csg-cmp pairs (S1, S2): S1, S2 connected, disjoint and adjacent, each unordered pair
    once (min(S1) < min(S2)), by EnumerateCmp (Moerkotte & Neumann 2006) -- only valid splits are
    ever generated. The csgs S1 are visited by descending minimum vertex, then ascending size, so
    when a pair is yielded all pairs that build S1 or S2 were yielded before (a valid DP order):
    the pairs building S2 have a csg with minimum min(S2) > min(S1), those building S1 a proper
    sub-csg of S1 with the same minimum."""
    hood = connected_subgraphs(n, nbr) if hood is None else hood
    for s1 in sorted(hood, key=lambda m: (-(m & -m), m.bit_count())):
        low = s1 & -s1
        x = ((low << 1) - 1) | s1                        # B_min(S1) and S1 itself
        frontier = hood[s1] & ~x
        stack = []
        for v in reversed(_bits(frontier)):
            vb = 1 << v
            yield s1, vb
            stack.append((vb, x | (frontier & ((vb << 1) - 1))))
            while stack:                                 # EnumerateCsgRec from {v}
                s, xs = stack.pop()
                grow = hood[s] & ~xs
                sub = 0
                while True:
                    sub = (sub - grow) & grow
                    if not sub:
                        break
                    yield s1, s | sub
                    stack.append((s | sub, xs | grow))


class JoinOrderEnumerator:
    """
    Enumerates join orders using dynamic programming with connected complement pairs (DPccp).
    This algorithm finds optimal join orders for query optimization by exploring all possible
    connected sub graphs and their complements.
    """

    def __init__(self, adjacency_list: Dict[int, List[int]],
                 estimated_cardinality: Callable[[tuple], float],
                 n_entries: int,
                 cardinality_is_log_scale=False):
        self.adjacency_list = adjacency_list
        self.estimated_cardinality = estimated_cardinality
        self.n_entries = n_entries
        self.cardinality_is_log_scale = cardinality_is_log_scale

    def search(self) -> tuple[JoinPlan, JoinPlan]:
        """Optimal bushy and left-deep plans (DPccp over bitmasks; one cardinality estimate per
        connected subgraph). Costs as JoinPlan computes them; ties may pick a different plan of
        equal cost than the earlier set-based version."""
        n = self.n_entries
        nbr = neighbour_masks(self.adjacency_list, n)
        hood = connected_subgraphs(n, nbr)
        raw = {m: self.estimated_cardinality(tuple(_bits(m))) for m in hood}
        size = {m: math.exp(v) for m, v in raw.items()} if self.cardinality_is_log_scale else dict(raw)
        cost = {1 << i: size[1 << i] for i in range(n)}  # bushy, per subset: cost and (left, right)
        cost_ld = dict(cost)                             # left-deep
        split, split_ld = {}, {}

        for s1, s2 in csg_cmp_pairs(n, nbr, hood):
            union = s1 | s2
            card = size[union]
            leaf1, leaf2 = not s1 & (s1 - 1), not s2 & (s2 - 1)
            if leaf1 and leaf2:
                # Two scans: bushy counts the right one, left-deep the left one; the cheaper side wins.
                a, b = size[s2] + card, size[s1] + card
                value, children = (b, (s2, s1)) if a > b else (a, (s1, s2))
                a, b = size[s1] + card, size[s2] + card
                value_ld, children_ld = (b, (s2, s1)) if a > b else (a, (s1, s2))
            elif leaf1:
                value, children = cost[s2] + card, (s1, s2)
                value_ld, children_ld = cost_ld[s2] + card, (s2, s1)
            elif leaf2:
                value, children = cost[s1] + card, (s1, s2)
                value_ld, children_ld = cost_ld[s1] + card, (s1, s2)
            else:
                value, children = cost[s1] + cost[s2] + card, (s1, s2)
                value_ld = None
            if union not in cost or cost[union] > value:
                cost[union], split[union] = value, children
            if value_ld is not None and (union not in cost_ld or cost_ld[union] > value_ld):
                cost_ld[union], split_ld[union] = value_ld, children_ld

        full = (1 << n) - 1
        if full not in cost:
            raise KeyError(tuple(range(n)))              # no plan without a cartesian product
        return self._plan(full, split, raw, False), self._plan(full, split_ld, raw, True)

    def _plan(self, mask, split, raw, left_deep) -> JoinPlan:
        if mask & (mask - 1):
            left, right = split[mask]
            return JoinPlan(left=self._plan(left, split, raw, left_deep),
                            right=self._plan(right, split, raw, left_deep),
                            entries=set(_bits(mask)), estimated_size=raw[mask], left_deep=left_deep,
                            cardinality_is_log_scale=self.cardinality_is_log_scale)
        return JoinPlan(left=None, right=None, entries=set(_bits(mask)), estimated_size=raw[mask],
                        left_deep=left_deep, cardinality_is_log_scale=self.cardinality_is_log_scale)

    def enumerate_left_deep_plans(self):
        cardinality_cache = {}
        for i in range(self.n_entries):
            cardinality_cache[(i,)] = self.estimated_cardinality((i,))
        plans = []
        for j in range(self.n_entries):
            singleton_plan = JoinPlan(left=None, right=None,
                                      entries={j},
                                      estimated_size=cardinality_cache[(j,)],
                                      left_deep=True,
                                      cardinality_is_log_scale=self.cardinality_is_log_scale)
            self.recurse_left_deep(plans, [j], singleton_plan, cardinality_cache, self.n_entries)
        return plans

    def recurse_left_deep(self, plans, current_order, current_plan: JoinPlan, cardinality_cache, total_entries):
        current_entries = current_plan.entries
        if len(current_entries) == total_entries:
            plans.append(current_plan)
        neighbours = list(self._get_neighbours(current_entries))
        for neighbour in neighbours:
            if neighbour not in current_entries:
                estimated_cardinality_singleton = cardinality_cache[(neighbour,)]
                right_leave_plan = JoinPlan(left=None, right=None, entries={neighbour},
                                            estimated_size=estimated_cardinality_singleton,
                                            left_deep=True,
                                            cardinality_is_log_scale=self.cardinality_is_log_scale)
                new_entries = current_entries | {neighbour}

                estimate_key = tuple(self.sort_array_asc(list(new_entries)))

                # Use cached cardinality for sub queries as the same sub query can occur in different plans.
                if estimate_key not in cardinality_cache:
                    estimate_new_plan = self.estimated_cardinality(estimate_key)
                    cardinality_cache[estimate_key] = estimate_new_plan
                else:
                    estimate_new_plan = cardinality_cache[estimate_key]

                new_plan = JoinPlan(left=current_plan, right=right_leave_plan,
                                    entries=new_entries,
                                    estimated_size=estimate_new_plan,
                                    left_deep=True,
                                    cardinality_is_log_scale=self.cardinality_is_log_scale)
                self.recurse_left_deep(plans, current_order + [neighbour], new_plan, cardinality_cache,
                                       total_entries)

    def sample_left_deep_plans(
            self,
            max_samples_per_start_relation,
    ):
        """
        Depth-first sampling of left-deep plans.

        Args:
            :param max_samples_per_start_relation: Number of samples to get for each start relation (triple)
        """
        max_samples = min(math.factorial(self.n_entries - 1), max_samples_per_start_relation)
        cardinality_cache = {}
        for i in range(self.n_entries):
            cardinality_cache[(i,)] = self.estimated_cardinality((i,))
        plans = []
        for j in range(self.n_entries):
            # self.recurse_left_deep(plans, [j], singleton_plan, cardinality_cache, self.n_entries)

            plans.extend(self._sample_plans(j, cardinality_cache, self.n_entries,
                                            max_samples))
        return plans

    def _sample_plans(self, start_entry, cardinality_cache, total_entries, max_samples):
        """
        Randomly generate up to max_samples left-deep join plans.

        Args:
            start_entry: base relation ID to start from
            cardinality_cache: dict for caching subplan cardinalities
            total_entries: total number of base relations
            max_samples: number of random plans to produce
        """
        plans = []

        for _ in range(max_samples):
            # Start a new plan from the same root each time
            start_key = (start_entry,)
            if start_key not in cardinality_cache:
                cardinality_cache[start_key] = self.estimated_cardinality(start_key)
            start_card = cardinality_cache[start_key]

            current_plan = JoinPlan(left=None,
                                    right=None,
                                    entries={start_entry},
                                    estimated_size=start_card,
                                    left_deep=True,
                                    cardinality_is_log_scale=self.cardinality_is_log_scale)
            current_entries = {start_entry}

            # Build one complete plan
            while len(current_entries) < total_entries:
                neighbours = list(self._get_neighbours(current_entries))
                # Filter out neighbours already in the plan
                available = [n for n in neighbours if n not in current_entries]

                if not available:
                    # Dead-end — restart this plan attempt
                    break

                next_neighbour = random.choice(available)
                right_key = (next_neighbour,)

                if right_key not in cardinality_cache:
                    cardinality_cache[right_key] = self.estimated_cardinality(right_key)
                right_card = cardinality_cache[right_key]

                right_leaf = JoinPlan(left=None,
                                      right=None,
                                      entries={next_neighbour},
                                      estimated_size=right_card,
                                      left_deep=True,
                                      cardinality_is_log_scale=self.cardinality_is_log_scale)
                new_entries = current_entries | {next_neighbour}

                estimate_key = tuple(self.sort_array_asc(list(new_entries)))
                if estimate_key not in cardinality_cache:
                    cardinality_cache[estimate_key] = self.estimated_cardinality(estimate_key)
                new_card = cardinality_cache[estimate_key]

                current_plan = JoinPlan(left=current_plan,
                                        right=right_leaf,
                                        entries=new_entries,
                                        estimated_size=new_card,
                                        left_deep=True,
                                        cardinality_is_log_scale=self.cardinality_is_log_scale)
                current_entries = new_entries

            if len(current_entries) == total_entries:
                plans.append(current_plan)

        return plans

    def enumerate_csg_cmp_pairs(self, n_tps: int) -> List[Tuple[Set[int], Set[int]]]:
        """Enumerate all connected subgraph-complement pairs"""
        return [(set(_bits(a)), set(_bits(b)))
                for a, b in csg_cmp_pairs(n_tps, neighbour_masks(self.adjacency_list, n_tps))]

    def enumerate_csg(self, n_tps: int) -> List[Set[int]]:
        """Enumerate all connected sub graphs"""
        return [set(_bits(m)) for m in connected_subgraphs(n_tps, neighbour_masks(self.adjacency_list, n_tps))]

    def _get_neighbours(self, s: Set[int]) -> Set[int]:
        """Get all neighbours of vertices in set S"""
        neighbours: Set[int] = set()
        for vertex in s:
            if vertex in self.adjacency_list:
                neighbours.update(self.adjacency_list[vertex])
        return neighbours

    @staticmethod
    def sort_array_asc(arr: List[int]) -> List[int]:
        """Sort array in ascending order"""
        return sorted(arr)


def build_adj_list(query):
    term_patterns = re.compile(
        r'\?[a-zA-Z0-9_]+'  # variable, e.g. ?o1
        r'|<[^>]+>'  # IRI, e.g. <http://example.com/433045>
        r'|"(?:[^"\\]|\\.)*"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?'  # literal, incl. optional lang tag/datatype
    )

    def _extract_terms(triple_pattern):
        """Extract the ordered terms (subject, predicate, object) of a triple pattern."""
        terms = term_patterns.findall(triple_pattern)
        if len(terms) < 3:
            raise ValueError(f"Could not parse triple pattern: {triple_pattern!r}")
        return terms[:3]

    # Parse each triple pattern to extract variables
    triple_patterns = query.triple_patterns
    parsed_patterns = []

    for i, pattern in enumerate(triple_patterns):
        # Create a simple SPARQL query to parse the pattern
        # variables = set(re.findall(r'\?[a-zA-Z0-9_]+', pattern))
        join_terms = set(_extract_terms(pattern))

        parsed_patterns.append({
            'index': i,
            'pattern': pattern,
            'join_terms': join_terms
        })
    # Create adjacency dictionary - regular dict since we're initializing all keys
    adjacency_dict = {i: set() for i in range(len(triple_patterns))}

    # Check each pair for shared variables
    for i in range(len(parsed_patterns)):
        for j in range(i + 1, len(parsed_patterns)):
            vars_i = parsed_patterns[i]['join_terms']
            vars_j = parsed_patterns[j]['join_terms']

            if vars_i.intersection(vars_j):
                adjacency_dict[i].add(j)
                adjacency_dict[j].add(i)
    # Convert to a list
    adjacency_dict = {key: sorted(list(value)) for key, value in adjacency_dict.items()}
    return adjacency_dict
