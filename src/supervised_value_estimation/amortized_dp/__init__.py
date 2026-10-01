"""Amortised dynamic programming for join ordering.

A value network over the *contracted join graph* predicts, for a set S of already-joined
triple patterns, both log card(S) and the optimal log cost-to-go G(S). Greedy (or narrow
beam) decoding on  card(S u a) + G(S u a)  then approximates exact DP at a fraction of its
planning cost. Exact DP is only used offline, as the teacher that produces the targets.

Modules:
    labels               oracle cardinalities for every connected subset + exact DP targets
    model                ContractedJoinGraphValueNet
    agents               AbstractCostAgent implementations and builders for the comparison
    simulated_execution  execution strategy that scores plans by their true (oracle) cost
    train_amortized_dp   training entry point
    compare_agents       runs every agent through the existing validation runner
"""
