"""Statistics-based generation of join-ordering-hard SPARQL BGP workloads.

Queries are generated from exact statistics of the data (no optimizer estimates, no learned
model in the loop) and kept only if their PLAN SPACE is hard: some left-deep order is much
cheaper than a typical one (spread), while the cheap order itself stays cheap (guardrails).
See generate.py for the pipeline.
"""
