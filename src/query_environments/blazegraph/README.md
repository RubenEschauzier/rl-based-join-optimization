## Blazegraph Query Environment

Query environment for Blazegraph that allows running full queries, 
count queries for cardinality estimation, and can process query output to obtain a reward signal.

Blazegraph supports turning off the join optimizer using query hints, thus only left-deep
plans can be passed to blazegraph. 

Possible rewards signals are: intermediate results generated and total execution time (WIP).

