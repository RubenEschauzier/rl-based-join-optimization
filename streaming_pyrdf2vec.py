import argparse
import os
import json
import glob

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from gensim.models import Word2Vec as GensimWord2Vec
from rdflib import URIRef
from tqdm import tqdm

from src.datastructures.query import ProcessQuery


class WalksCorpus:
    """Stream walks from a file line by line for gensim Word2Vec."""

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r") as f:
            for line in f:
                yield line.strip().split()


def main():
    parser = argparse.ArgumentParser(
        description="Train pyrdf2vec using disk-based walk storage with a SPARQL endpoint.")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint URL.")
    parser.add_argument("--queries_to_embed",
                        required=True,
                        nargs="+",
                        help="File location of queries to embed entities of.")
    parser.add_argument("--output", required=True, help="Output folder for walks and model.")
    parser.add_argument("--model_file_name", type=str, default="model.json",
                        help="File to where the model should be written")
    parser.add_argument("--num_walks", type=int, default=100, help="Number of walks per entity.")
    parser.add_argument("--depth", type=int, default=4, help="Depth of walks.")
    parser.add_argument("--dimensions", type=int, default=128, help="Embedding size.")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size.")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec epochs.")
    parser.add_argument("--workers", type=int, default=4, help="Word2Vec epochs.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    walks_file = os.path.join(args.output, "walks.txt")

    kg = KG(args.endpoint, is_remote=True)

    # Walk generator
    walker = RandomWalker(max_depth=args.depth, max_walks=args.num_walks, with_reverse=True, md5_bytes=None)
    # transformer = RDF2VecTransformer(walkers=walker)
    # Read entities to embed
    entities = set()
    for query_path in args.queries_to_embed:
        with open(query_path, 'r') as f:
            raw_data = json.load(f)
        for i, data in tqdm(enumerate(raw_data), total=len(raw_data)):
            _, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])
            for tp in tp_rdflib:
                for entity in tp:
                    if isinstance(entity, URIRef):
                        entities.add(str(entity))

    entities = list(entities)

    print("Generating walks from SPARQL endpoint...")
    with open(walks_file, "w") as f:
        for entity in tqdm(entities):
            try:
                walks = walker.extract(kg, [entity])[0]
                for walk in walks:
                    f.write(" ".join(walk) + "\n")
            except Exception as e:
                print(f"Failed to generate walks for {entity}: {e}")

    print("Training Word2Vec on walks...")
    sentences = WalksCorpus(walks_file)
    model = GensimWord2Vec(
        sentences=sentences,
        vector_size=args.dimensions,
        window=args.window,
        sg=1,
        workers=args.workers,
        epochs=args.epochs,
    )

    model.save(os.path.join(args.output, "embeddings.model"))
    print(f"Model saved to {os.path.join(args.output, 'model.json')}")
    data = {key: model.wv[key].tolist() for key in model.wv.key_to_index}

    # Save to JSON file
    with open(os.path.join(args.output, args.model_file_name), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Example: python rdf2vec_disk_test --endpoint http://localhost:9999/blazegraph/namespace/yago/sparql
    # --output walks_temp/ --queries_to_embed .\data\generated_queries\star_yago_gnce\Joined_Queries.json
    # --epochs 50 --num_walks 10 --workers 5
    main()
