from pyrdf2vec.graphs import KG
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker

from SPARQLWrapper import SPARQLWrapper, JSON

# ========== CONFIG ==========
dataset = "lubm"
endpoint_location = f"http://localhost:9999/blazegraph/namespace/{dataset}/sparql"
MAX_DEPTH = 3
N_WALKS = 10
WORKERS = 4
SEED = 0

VECTOR_SIZE = 128
WINDOW = 5
MIN_COUNT = 1
EPOCHS = 5
# ============================


def get_unique_entities_and_predicates(endpoint: str):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    # Get distinct subjects and objects
    sparql.setQuery("""
        SELECT DISTINCT ?x WHERE {
          { ?x ?p ?o }
          UNION
          { ?s ?p ?x }
        }
    """)
    res = sparql.query().convert()
    entities = [b["x"]["value"] for b in res["results"]["bindings"]]

    # Get distinct predicates
    sparql.setQuery("""
        SELECT DISTINCT ?p WHERE {
          ?s ?p ?o
        }
    """)
    res = sparql.query().convert()
    predicates = [b["p"]["value"] for b in res["results"]["bindings"]]

    return entities, predicates

if __name__ == "__main__":
    # Think about what to do with literals
    terms, pred = get_unique_entities_and_predicates(endpoint_location)
    all_terms = terms + pred
    print(f"Discovered {len(terms)} terms and {len(pred)} predicates.")
    kg = KG(endpoint_location)
    transformer = RDF2VecTransformer(
        embedder=Word2Vec(
            vector_size=VECTOR_SIZE,
            epochs=EPOCHS,
            window=WINDOW,
            min_count=MIN_COUNT,
        ),
        walkers=[
            RandomWalker(
                max_depth=MAX_DEPTH,
                max_walks=N_WALKS,
                with_reverse=True,
                n_jobs=WORKERS,
                random_state=SEED,)
        ],
        verbose=1
    )

    # Step 4: train embeddings
    print("Starting transformer")
    embeddings, literals = transformer.fit_transform(kg, all_terms)
    print(embeddings)
    entity_to_vec = {e: embeddings[i] for i, e in enumerate(transformer._entities)}
    print("Embeddings shape:", embeddings.shape)
