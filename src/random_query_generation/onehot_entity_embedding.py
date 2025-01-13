import numpy as np
from rdf2vec_entity_embedding import load_graph


def get_all_predicate(g):
    all_pred = set()
    for (_, pred, _) in g:
        all_pred.add(pred)
    return all_pred


def one_hot(predicates):
    n = len(predicates)
    encodings = []
    for i, pred in enumerate(predicates):
        encoding = np.zeros(n)
        encoding[i] = 1
        encodings.append(encoding)
    return encodings


def save_one_hot(vectors, predicates, location):
    with open(location, 'w') as f:
        # First write predicates to file
        for predicate, vector in zip(predicates, vectors):
            to_write = str(predicate) + '[sep]' + ' '.join([str(x) for x in vector])
            f.write(to_write)
            f.write('\n')


def main_onehot_embedding(instantiation_benchmark_location, vector_save_location):

    g = load_graph(instantiation_benchmark_location)
    p = get_all_predicate(g)
    v = one_hot(p)

    save_one_hot(v, p, vector_save_location)