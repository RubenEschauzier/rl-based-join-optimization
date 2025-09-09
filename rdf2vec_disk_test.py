import argparse
import os
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
from gensim.models import Word2Vec as GensimWord2Vec

class WalksCorpus:
    """Stream walks from a file line by line for gensim Word2Vec."""
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r") as f:
            for line in f:
                yield line.strip().split()


def main():
    parser = argparse.ArgumentParser(description="Train pyrdf2vec using disk-based walk storage.")
    parser.add_argument("--kg", required=True, help="Knowledge graph file (ttl, nt, etc.).")
    parser.add_argument("--output", required=True, help="Output folder for walks and model.")
    parser.add_argument("--num-walks", type=int, default=100, help="Number of walks per entity.")
    parser.add_argument("--depth", type=int, default=4, help="Depth of walks.")
    parser.add_argument("--dimensions", type=int, default=128, help="Embedding size.")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size.")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec epochs.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    walks_file = os.path.join(args.output, "walks.txt")

    # Load KG
    kg = KG(args.kg)

    # Walk generator
    walker = RandomWalker(max_depth=args.depth, n_walks=args.num_walks, with_reverse=True)

    print("Generating walks...")
    with open(walks_file, "w") as f:
        for entity in kg._entities:  # all entities in KG
            walks = walker.extract(kg, [entity])[0]
            for walk in walks:
                f.write(" ".join(walk) + "\n")

    print("Training Word2Vec on walks...")
    sentences = WalksCorpus(walks_file)
    model = GensimWord2Vec(
        sentences=sentences,
        vector_size=args.dimensions,
        window=args.window,
        sg=1,
        workers=4,
        epochs=args.epochs,
    )

    model.save(os.path.join(args.output, "embeddings.model"))
    print(f"Model saved to {os.path.join(args.output, 'embeddings.model')}")


if __name__ == "__main__":
    main()