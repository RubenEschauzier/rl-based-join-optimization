Query environment for virtuoso that allows running full queries, 
count queries for cardinality estimation (WIP), and can process query output to obtain a reward signal.

Virtuoso supports turning off the join optimizer using query hints, thus only left-deep
plans can be passed to virtuoso using query hints. 

Possible rewards signals are: intermediate results generated 
(WIP depending on how this works with the SQL backend of virtuoso) and total execution time (WIP).


