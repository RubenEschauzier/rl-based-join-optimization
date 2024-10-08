from time import sleep
from typing import Literal

import rdflib.term
import requests
import random
from tqdm import tqdm

from src.random_query_generation.generation_utils import create_variable_dictionary, filter_isomorphic_queries, \
    query_all_terms

ENDPOINT_LIMIT = 10000


def main_generate_star(endpoint_url,
                       n_stars, star_sizes, p_object_star, start_star_type, n_literal_stars,
                       default_graph_uri=None):
    generated_queries_no_literal = []
    generated_queries_with_literal = []
    for size in star_sizes:
        star_queries_no_literal, star_queries_literal = generate_star_queries(endpoint_url=endpoint_url,
                                                                              n_stars=n_stars,
                                                                              star_size=size,
                                                                              p_object_star=p_object_star,
                                                                              star_start_type=start_star_type,
                                                                              n_literal_stars=n_literal_stars,
                                                                              default_graph_uri=default_graph_uri
                                                                              )
        generated_queries_with_literal.extend(star_queries_no_literal)
        generated_queries_with_literal.extend(star_queries_literal)
    print("Generated star queries with literal: {}, without literal: {}".format(len(generated_queries_with_literal),
                                                                                len(generated_queries_no_literal)))


def generate_star_queries(endpoint_url,
                          n_stars, star_size, p_object_star, star_start_type, n_literal_stars,
                          default_graph_uri=None):
    non_literal_star_queries = []
    literal_star_queries = []

    subj_stars, object_stars = sample_all_stars(endpoint_url=endpoint_url,
                                                n_stars=n_stars,
                                                star_size=star_size,
                                                p_object_star=p_object_star,
                                                star_start_type=star_start_type,
                                                default_graph_uri=default_graph_uri
                                                )

    non_literal_star_queries.extend(filter_isomorphic_queries(triples_to_query(subj_stars, "subject", False)))
    non_literal_star_queries.extend(filter_isomorphic_queries(triples_to_query(object_stars, "object", False)))

    for i in range(n_literal_stars):
        # Might be very inefficient, can also just use already generated queries
        subj_stars_literal, obj_stars_literals = sample_all_stars(endpoint_url=endpoint_url,
                                                                  n_stars=n_stars,
                                                                  star_size=star_size,
                                                                  p_object_star=p_object_star,
                                                                  star_start_type=star_start_type,
                                                                  default_graph_uri=[
                                                                      'http://localhost:8890/watdiv-default'
                                                                      '-instantiation']
                                                                  )
        literal_star_queries.extend(filter_isomorphic_queries(triples_to_query(subj_stars_literal, "subject", True)))
        literal_star_queries.extend(filter_isomorphic_queries(triples_to_query(obj_stars_literals, "object", True)))

    return non_literal_star_queries, literal_star_queries


def triples_to_query(star_triples, type_star: Literal['subject', 'object'], literal_in_star):
    queries = []
    for star in star_triples:
        # Create dictionary mapping terms to variable names
        variable_dict = create_variable_dictionary(star)
        # Randomly choose a triple to add literal / uri in query (if literal_in_star = True)
        literal_index = random.randint(0, len(star))

        query = "SELECT * WHERE { \n "
        for i, triple in enumerate(star):
            triple_string = "{} {} {} . \n "
            if i == literal_index and literal_in_star:
                if type_star == "subject":
                    triple_string = triple_string.format(variable_dict[triple[0]],
                                                         triple[1].n3(),
                                                         triple[2].n3())
                if type_star == 'object':
                    triple_string = triple_string.format(triple[0].n3(),
                                                         triple[1].n3(),
                                                         variable_dict[triple[2]])
            else:
                triple_string = triple_string.format(variable_dict[triple[0]],
                                                     triple[1].n3(),
                                                     variable_dict[triple[2]])
            query += triple_string

        query += "}"
        queries.append(query)
    return queries


# Adapted from https://github.com/DE-TUM/rdf-subgraph-sampler/
def query_triple(endpoint_url,
                 limit,
                 s=rdflib.term.Variable("?s"), p=rdflib.term.Variable("?p"), o=rdflib.term.Variable("?o"),
                 default_graph_uri=None):
    sleep(.001)
    # Get the query and select statement based on supplied values for s p o
    spo = "{} {} {}".format(s.n3(), p.n3(), o.n3())
    select = ""
    select += (" ?s" if s.n3() == "?s" else "")
    select += (" ?p" if p.n3() == "?p" else "")
    select += (" ?o" if o.n3() == "?o" else "")

    # Adapt query string to use SPARQL 1.1 RAND()
    query_string = "SELECT DISTINCT " + select + " WHERE {" + spo + " } " + \
                   "ORDER BY RAND() LIMIT " + str(limit)
    # SELECT DISTINCT  ?p ?o WHERE {<http://db.uwaterloo.ca/~galuc/wsdbm/Offer555> ?p ?o } ORDER BY RAND() LIMIT 10000
    # Triple: 5298
    # SELECT DISTINCT  ?p ?o WHERE {<http://db.uwaterloo.ca/~galuc/wsdbm/Offer5538> ?p ?o } ORDER BY RAND() LIMIT 10000
    # Triple: 5284
    r = requests.get(endpoint_url,
                     params={'query': query_string,
                             'format': 'json',
                             'default-graph-uri': default_graph_uri}
                     )
    # Sometimes gives error (service unavailable etc) if that is the case we have to retry the query
    if r.status_code == 503:
        print(r)
        print("503 found!!!!")
    res = r.json()["results"]["bindings"]

    # Get bindings from results, if any of the queried triple terms is not a variable we repeat the term in the
    # triple pattern
    subjects = [rdflib.term.URIRef(binding['s']['value']) if s.n3() == "?s" else s for binding in res]
    predicates = [rdflib.term.URIRef(binding['p']['value']) if p.n3() == "?p" else p for binding in res]

    objects = []
    for binding in res:
        if o.n3() == "?o":
            if binding['o']['type'] == "literal":
                objects.append(rdflib.term.Literal(binding['o']['value']))
            else:
                objects.append(rdflib.term.URIRef(binding['o']['value']))
        else:
            objects.append(o)

    return subjects, predicates, objects


def sample_start_triples(endpoint_url, default_graph_uri, limit, samples,
                         s=rdflib.term.Variable("?s"), p=rdflib.term.Variable("?p")):
    if samples > limit:
        raise ValueError("Sampling more triples than available through limit")
    # Randomly sample triples with predicate
    subjects, predicates, objects = query_triple(endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 s=s, p=p,
                                                 limit=limit)
    # Form triples from the result by merging the lists and predicates into tuples
    triples_from_pred = list(map(lambda e: (e[0], e[1], e[2]), zip(subjects, predicates, objects)))
    # If we find less triples than we want to sample we just return all found triples
    if len(triples_from_pred) <= samples:
        return triples_from_pred

    # Return sample of triples
    return random.sample(triples_from_pred, k=samples)


def sample_star(endpoint_url, default_graph_uri, seed_triple, subject_star, size):
    if subject_star:
        # Sample triples with the given subject
        _, pred, obj = query_triple(endpoint_url, ENDPOINT_LIMIT, s=seed_triple[0],
                                    default_graph_uri=default_graph_uri)
        star_triples = list(map(lambda e: (seed_triple[0], e[0], e[1]), zip(pred, obj)))
    else:
        subj, pred, _ = query_triple(endpoint_url, ENDPOINT_LIMIT, o=seed_triple[2],
                                     default_graph_uri=default_graph_uri)
        # Sample triples with the given object
        star_triples = list(map(lambda e: (e[0], e[1], seed_triple[2]), zip(subj, pred)))

    # Filter out seed_triple
    star_triples = [triple for triple in star_triples if triple[1] != seed_triple[1]]
    if len(star_triples) == 0:
        return []
    # Create dictionary to allow for sampling according to possible predicates and avoid duplicates
    pred_dict_star_triples = create_predicate_dictionary(star_triples)

    star = [seed_triple]
    while len(star) < size:
        # Sample a predicate
        sampled_predicate = random.choice(list(pred_dict_star_triples.keys()))
        # Sample a triple with that predicate
        sampled_triple = random.choice(pred_dict_star_triples[sampled_predicate])

        # Add sampled triple to star
        star.append(sampled_triple)
        # Remove predicate to avoid duplicate predicate usage in query
        pred_dict_star_triples.pop(sampled_predicate)

        # If we don't have any predicates to add to star we stop our loop
        if len(list(pred_dict_star_triples.keys())) == 0:
            break
    return star


def sample_all_stars(endpoint_url,
                     n_stars, star_size, p_object_star,
                     star_start_type: Literal["?s", "?p"],
                     default_graph_uri=None):
    start_terms = query_all_terms(endpoint_url=endpoint_url,
                                  term_type=star_start_type,
                                  default_graph_uri=default_graph_uri)
    print("Generating stars, maximum number of stars: {}".format(n_stars * len(start_terms)))
    subject_stars = []
    object_stars = []
    for term in tqdm(start_terms):
        start_triples = []
        # Query start triples according to the star seed type in the function
        if star_start_type == "?s":
            start_triples = sample_start_triples(endpoint_url=endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 limit=ENDPOINT_LIMIT,
                                                 samples=n_stars,
                                                 s=term)
        if star_start_type == "?p":
            start_triples = sample_start_triples(endpoint_url=endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 limit=ENDPOINT_LIMIT,
                                                 samples=n_stars,
                                                 p=term)
        # sample stars around the triple with given size
        for seed_triple in start_triples:
            # Random choose subject or object star according to probability
            subject_star = True

            if random.random() < p_object_star:
                subject_star = False

            star = sample_star(endpoint_url, default_graph_uri, seed_triple, subject_star, star_size)

            if len(star) == star_size:
                if subject_star:
                    subject_stars.append(star)
                else:
                    object_stars.append(star)
    return subject_stars, object_stars


def create_predicate_dictionary(triples):
    pred_dict = {}
    for triple in triples:
        if triple[1] not in pred_dict:
            pred_dict[triple[1]] = [triple]
        else:
            pred_dict[triple[1]].append(triple)
    return pred_dict


if __name__ == "__main__":
    main_generate_star(endpoint_url="http://localhost:8890/sparql",
                       default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
                       n_stars=1,
                       star_sizes=[2, 3, 4, 5, 6],
                       p_object_star=.1,
                       start_star_type="?s",
                       n_literal_stars=1
                       )
    # generated_stars = sample_all_stars(endpoint_url="http://localhost:8890/sparql",
    #                                    n_stars=1,
    #                                    star_size=3,
    #                                    p_object_star=.2,
    #                                    star_start_type="?p",
    #                                    default_graph_uri=['http://localhost:8890/watdiv-default-instantiation']
    #                                    )
    # triples_to_query(generated_stars, "subject", False)
    # start_triples = sample_start_triples(endpoint_url="http://localhost:8890/sparql",
    #                                      default_graph=['http://localhost:8890/watdiv-default-instantiation'],
    #                                      limit=10,
    #                                      samples=5,
    #                                      p="<http://purl.org/stuff/rev#text>"
    #                                      )
    # print("Seed triple: {}".format(start_triples[1]))
    # sample_star(endpoint_url="http://localhost:8890/sparql",
    #             default_graph=['http://localhost:8890/watdiv-default-instantiation'],
    #             seed_triple=start_triples[0],
    #             subject_star=True,
    #             size=3)
