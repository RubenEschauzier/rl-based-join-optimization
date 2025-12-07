import math
import random
from typing import Dict, List, Set, Tuple, Optional, Callable
import re


class JoinPlan:
    """Represents a join plan with cost estimation"""

    def __init__(self, left: Optional['JoinPlan'], right: Optional['JoinPlan'],
                 entries: Set[int], estimated_size: float, left_deep: bool):
        self.left = left
        self.right = right
        self.entries = entries
        self.estimated_size = estimated_size
        self.left_deep = left_deep
        self.is_leaf = self.left is None or self.right is None
        self.cost = self._calculate_cost()

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


class JoinOrderEnumerator:
    """
    Enumerates join orders using dynamic programming with connected complement pairs (DPccp).
    This algorithm finds optimal join orders for query optimization by exploring all possible
    connected sub graphs and their complements.
    """

    def __init__(self, adjacency_list: Dict[int, List[int]],
                 estimated_cardinality: Callable[[tuple], float], n_entries: int):
        self.adjacency_list = adjacency_list
        self.estimated_cardinality = estimated_cardinality
        self.n_entries = n_entries

    def search(self) -> [JoinPlan, JoinPlan]:
        """Main search method that finds the optimal join plan"""
        best_plan: Dict[tuple, JoinPlan] = {}
        best_plan_left_deep: Dict[tuple, JoinPlan] = {}

        # Initialize singleton plans
        for i in range(self.n_entries):
            singleton = {i}
            singleton_key = (i,)
            estimated_cardinality = self.estimated_cardinality(singleton_key)
            best_plan[singleton_key] = JoinPlan(None, None, singleton, estimated_cardinality, False)
            best_plan_left_deep[singleton_key] = JoinPlan(None, None, singleton, estimated_cardinality, True)

        # Enumerate all connected subgraph-complement pairs
        csg_cmp_pairs = self.enumerate_csg_cmp_pairs(self.n_entries)
        for csg_cmp_pair in csg_cmp_pairs:
            csg, cmp = csg_cmp_pair[0], csg_cmp_pair[1]
            tree1_key = tuple(self.sort_array_asc(list(csg)))
            tree2_key = tuple(self.sort_array_asc(list(cmp)))

            tree1 = best_plan[tree1_key]
            tree2 = best_plan[tree2_key]
            new_entries = tree1.entries | tree2.entries

            estimate_key = tuple(self.sort_array_asc(list(new_entries)))
            estimate = self.estimated_cardinality(estimate_key)

            self.update_best_plan(tree1, tree2, new_entries, estimate, best_plan, best_plan_left_deep, False)

            # If tree2 key is size 1 we are working with left-deep plan
            if len(tree2_key) == 1 or len(tree1_key) == 1:
                tree1_left_deep = best_plan_left_deep[tree1_key]
                tree2_left_deep = best_plan_left_deep[tree2_key]
                self.update_best_plan(tree1_left_deep, tree2_left_deep,
                                      new_entries, estimate, best_plan, best_plan_left_deep, True)

        all_entries_key = tuple(list(range(self.n_entries)))
        return best_plan[all_entries_key], best_plan_left_deep[all_entries_key]


    def enumerate_left_deep_plans(self):
        cardinality_cache = {}
        for i in range(self.n_entries):
            cardinality_cache[(i,)] = self.estimated_cardinality((i,))
        plans = []
        for j in range(self.n_entries):
            singleton_plan = JoinPlan(None, None, {j}, cardinality_cache[(j,)], True)
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
                right_leave_plan = JoinPlan(None, None, {neighbour}, estimated_cardinality_singleton, True)
                new_entries = current_entries | {neighbour}

                estimate_key = tuple(self.sort_array_asc(list(new_entries)))

                # Use cached cardinality for sub queries as the same sub query can occur in different plans.
                if estimate_key not in cardinality_cache:
                    estimate_new_plan = self.estimated_cardinality(estimate_key)
                    cardinality_cache[estimate_key] = estimate_new_plan
                else:
                    estimate_new_plan = cardinality_cache[estimate_key]

                new_plan = JoinPlan(current_plan, right_leave_plan,
                                    new_entries,
                                    estimate_new_plan, True)
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
        max_samples = min(math.factorial(self.n_entries-1), max_samples_per_start_relation)
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

            current_plan = JoinPlan(None, None, {start_entry}, start_card, True)
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

                right_leaf = JoinPlan(None, None, {next_neighbour}, right_card, True)
                new_entries = current_entries | {next_neighbour}

                estimate_key = tuple(self.sort_array_asc(list(new_entries)))
                if estimate_key not in cardinality_cache:
                    cardinality_cache[estimate_key] = self.estimated_cardinality(estimate_key)
                new_card = cardinality_cache[estimate_key]

                current_plan = JoinPlan(current_plan, right_leaf, new_entries, new_card, True)
                current_entries = new_entries

            if len(current_entries) == total_entries:
                plans.append(current_plan)

        return plans

    def enumerate_csg_cmp_pairs(self, n_tps: int) -> List[Tuple[Set[int], Set[int]]]:
        """Enumerate all connected subgraph-complement pairs"""
        all_tps = set(range(n_tps))
        csgs = self.enumerate_csg(n_tps)
        csg_cmp_pairs: List[Tuple[Set[int], Set[int]]] = []

        for csg in csgs:
            cmps = self.enumerate_cmp(csg, all_tps)
            for cmp in cmps:
                csg_cmp_pairs.append((csg, cmp))

        return csg_cmp_pairs

    def enumerate_csg(self, n_tps: int) -> List[Set[int]]:
        """Enumerate all connected sub graphs"""
        csgs: List[Set[int]] = []
        for i in range(n_tps - 1, -1, -1):
            v_i = {i}
            csgs.append(v_i)
            self._enumerate_csg_recursive(csgs, v_i, set(range(i + 1)))
        return csgs

    def enumerate_cmp(self, tps_subset: Set[int], all_tps: Set[int]) -> List[Set[int]]:
        """Enumerate complements for a given connected sub graph"""
        cmps: List[Set[int]] = []
        min_vertex = self._set_minimum(tps_subset)
        x = self._reduce_set(min_vertex, all_tps) | tps_subset
        neighbours = self._get_neighbours(tps_subset)

        # Remove vertices that are in X from neighbours
        for vertex in x:
            neighbours.discard(vertex)

        for vertex in self.sort_set_desc(neighbours):
            cmps.append({vertex})
            reduced_neighbours = self._reduce_set(vertex, neighbours)
            self._enumerate_csg_recursive(
                cmps,
                {vertex},
                x | reduced_neighbours
            )
        return cmps

    def _enumerate_csg_recursive(self, csgs: List[Set[int]], s: Set[int], x: Set[int]):
        """Recursively enumerate connected subgraphs"""
        neighbours = self._get_neighbours(s)
        # Remove vertices that are in X from neighbours
        for vertex in x:
            neighbours.discard(vertex)

        subsets = self._get_all_subsets(neighbours)
        for subset in subsets:
            csgs.append(s | subset)

        for subset in subsets:
            self._enumerate_csg_recursive(csgs, s | subset, x | neighbours)

    def update_best_plan(self, tree1, tree2, new_entries, estimate_size,
                         best_plan, best_plan_left_deep,
                         left_deep: bool):
        if left_deep:
            curr_plan_left = JoinPlan(tree1, tree2, new_entries, estimate_size, True)
            curr_plan_right = JoinPlan(tree2, tree1, new_entries, estimate_size, True)
            if len(tree1.entries) == 1 and len(tree2.entries) == 1:
                curr_plan = (curr_plan_right if curr_plan_left.cost > curr_plan_right.cost
                             else curr_plan_left)
            elif len(tree1.entries) == 1:
                curr_plan = curr_plan_right
            elif len(tree2.entries) == 1:
                curr_plan = curr_plan_left
            else:
                raise ValueError("Bushy plan passed to left-deep update")
            curr_plan_key = tuple(self.sort_array_asc(list(curr_plan.entries)))
            if (curr_plan_key not in best_plan_left_deep or
                    best_plan_left_deep[curr_plan_key].cost > curr_plan.cost):
                best_plan_left_deep[curr_plan_key] = curr_plan

        else:
            curr_plan_left = JoinPlan(tree1, tree2, new_entries, estimate_size, False)
            curr_plan_right = JoinPlan(tree2, tree1, new_entries, estimate_size, False)
            curr_plan = (curr_plan_right if curr_plan_left.cost > curr_plan_right.cost
                         else curr_plan_left)

            curr_plan_key = tuple(self.sort_array_asc(list(curr_plan.entries)))

            if (curr_plan_key not in best_plan or
                    best_plan[curr_plan_key].cost > curr_plan.cost):
                best_plan[curr_plan_key] = curr_plan
        return curr_plan, curr_plan_key

    def store_plan(self, trees1, trees2, new_entries, plan_key, estimate_size, stored_plans):
        for tree1 in trees1:
            for tree2 in trees2:
                left_plan = JoinPlan(tree1, tree2, new_entries, estimate_size, False)
                right_plan = JoinPlan(tree2, tree1, new_entries, estimate_size, False)
                if plan_key not in stored_plans:
                    stored_plans[plan_key] = []
                stored_plans[plan_key].append(left_plan)
                stored_plans[plan_key].append(right_plan)

    def _get_neighbours(self, s: Set[int]) -> Set[int]:
        """Get all neighbours of vertices in set S"""
        neighbours: Set[int] = set()
        for vertex in s:
            if vertex in self.adjacency_list:
                neighbours.update(self.adjacency_list[vertex])
        return neighbours

    @staticmethod
    def _get_all_subsets(s: Set[int]) -> List[Set[int]]:
        """Get all non-empty subsets of a set"""
        subsets: List[Set[int]] = [set()]
        for el in s:
            last = len(subsets) - 1
            for i in range(last + 1):
                subsets.append(subsets[i] | {el})
        # Remove empty set
        return subsets[1:]

    @staticmethod
    def _reduce_set(i: int, s: Set[int]) -> Set[int]:
        """Reduce set to elements <= i"""
        return {vertex for vertex in s if vertex <= i}

    @staticmethod
    def _set_minimum(s: Set[int]) -> int:
        """Get minimum element from set"""
        return min(s)

    @staticmethod
    def sort_set_desc(s: Set[int]) -> List[int]:
        """Sort set in descending order"""
        return sorted(s, reverse=True)

    @staticmethod
    def sort_array_asc(arr: List[int]) -> List[int]:
        """Sort array in ascending order"""
        return sorted(arr)


def build_adj_list(query):
    # Parse each triple pattern to extract variables
    triple_patterns = query.triple_patterns
    parsed_patterns = []

    for i, pattern in enumerate(triple_patterns):
        # Create a simple SPARQL query to parse the pattern
        variables = set(re.findall(r'\?[a-zA-Z0-9_]+', pattern))

        parsed_patterns.append({
            'index': i,
            'pattern': pattern,
            'variables': variables
        })
    # Create adjacency dictionary - regular dict since we're initializing all keys
    adjacency_dict = {i: set() for i in range(len(triple_patterns))}

    # Check each pair for shared variables
    for i in range(len(parsed_patterns)):
        for j in range(i + 1, len(parsed_patterns)):
            vars_i = parsed_patterns[i]['variables']
            vars_j = parsed_patterns[j]['variables']

            if vars_i.intersection(vars_j):
                adjacency_dict[i].add(j)
                adjacency_dict[j].add(i)
    # Convert to a list
    adjacency_dict = {key: sorted(list(value)) for key, value in adjacency_dict.items()}
    return adjacency_dict
