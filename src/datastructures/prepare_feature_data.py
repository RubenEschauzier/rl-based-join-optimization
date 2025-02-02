import json
import os

import rdflib
from tqdm import tqdm

from src.datastructures.query import ProcessQuery
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment


def read_queries(query_dir):
    dir_files = os.listdir(query_dir)
    queries_total = []
    for file in tqdm(dir_files):
        queries = []
        path = os.path.join(query_dir, file)
        with open(path, 'r') as f:
            raw_data = json.load(f)
        for i, data in enumerate(raw_data):
            tp_str, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])
            queries.append({
                "query": data['query'],
                "cardinality": data['y'],
                "triple_patterns": tp_str,
                "rdflib_patterns": tp_rdflib
            })
        queries_total.extend(queries)
    return queries_total

def get_occurrences(queries, env):
    occurrences = {}
    for query in tqdm(queries):
        for tp_rdflib in query['rdflib_patterns']:
            for entity in tp_rdflib:
                if type(entity) != rdflib.term.Variable and entity.n3() not in occurrences:
                    occurrence_entity = env.cardinality_term(entity.n3())
                    occurrences[entity.n3()] = occurrence_entity
    return occurrences
    pass

# Encodes dict {tp_str: cardinality}
def get_query_triple_pattern_cardinalities(queries, env):
    triple_pattern_cardinalities = {}
    for query in tqdm(queries):
        for tp_str in query['triple_patterns']:
            if tp_str not in triple_pattern_cardinalities:
                tp_cardinality = env.cardinality_triple_pattern(tp_str)
                triple_pattern_cardinalities[tp_str] = tp_cardinality
    return triple_pattern_cardinalities

if __name__ == '__main__':
    query_location = (
        r"C:\Users\ruben\projects\rl-based-join-optimization\data\pretrain_data\generated_queries"
        r"\sub_sampled_full"
    )
    output_location = (
        r"C:\Users\ruben\projects\rl-based-join-optimization\data\pretrain_data\pattern_term_cardinalities"
        r"\full"
    )
    query_env = BlazeGraphQueryEnvironment("http://localhost:9999/blazegraph/namespace/watdiv/sparql")
    loaded_queries = read_queries(query_location)
    loaded_occurrences = get_occurrences(loaded_queries, query_env)
    loaded_tp_cardinalities = get_query_triple_pattern_cardinalities(loaded_queries, query_env)
    with open(os.path.join(output_location, 'occurrences.json'), 'w') as f0:
        json.dump(loaded_occurrences, f0)
    with open(os.path.join(output_location, 'tp_cardinalities.json'), 'w') as f1:
        json.dump(loaded_tp_cardinalities, f1)
