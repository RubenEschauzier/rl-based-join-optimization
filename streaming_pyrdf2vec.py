import argparse
import multiprocessing
import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor

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


def validate_completeness_embeddings(model, entities):
    missing_entities = 0
    for entity in entities:
        if entity not in model.wv:
            missing_entities += 1
    return missing_entities


def validate_completeness_walks(walk_corpus, entities, min_count):
    missing_entities = 0
    insufficient_occurrences_entities = 0
    entity_counts = {}
    for walk in walk_corpus:
        for entity in walk:
            if entity not in entity_counts:
                entity_counts[entity] = 0
            entity_counts[entity] += 1
    for entity_to_embed in entities:
        if entity_to_embed not in entity_counts:
            missing_entities += 1
        else:
            if entity_counts[entity_to_embed] < min_count:
                insufficient_occurrences_entities += 1
    return missing_entities, insufficient_occurrences_entities


class MultiWalksCorpus:
    """Stream the walks of several files, one after the other."""

    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:
            yield from WalksCorpus(filepath)


_WORKER = {}


def _init_walk_worker(endpoints, depth, num_walks):
    # each worker process walks against one endpoint of its own
    _WORKER["endpoint"] = endpoints.get()
    _WORKER["depth"], _WORKER["num_walks"] = depth, num_walks


def _walk_chunk(entities):
    """Walks of a chunk of entities: [(entity, [walk line, ...] or None, error or None)]."""
    kg = KG(_WORKER["endpoint"], is_remote=True)
    walker = RandomWalker(max_depth=_WORKER["depth"], max_walks=_WORKER["num_walks"], with_reverse=True,
                          md5_bytes=None)
    try:
        walks = walker.extract(kg, entities, verbose=0)
        return [(entity, [" ".join(walk) for walk in entity_walks], None)
                for entity, entity_walks in zip(entities, walks)]
    except Exception as e:
        return [(entity, None, str(e)) for entity in entities]


def generate_walks(entities, endpoints, walks_file, walked_file, depth, num_walks, chunk_size=200):
    """Append the walks of `entities` to walks_file, in parallel over `endpoints` (one process per
    endpoint), and the walked entities to walked_file, so an interrupted run resumes where it
    stopped. Returns the entities whose walks failed."""
    manager = multiprocessing.Manager()
    endpoint_queue = manager.Queue()
    for endpoint in endpoints:
        endpoint_queue.put(endpoint)
    chunks = [entities[i:i + chunk_size] for i in range(0, len(entities), chunk_size)]
    failed, zero = [], 0
    with open(walks_file, "a") as f_walks, open(walked_file, "a") as f_walked, \
            ProcessPoolExecutor(len(endpoints), initializer=_init_walk_worker,
                                initargs=(endpoint_queue, depth, num_walks)) as pool:
        for result in tqdm(pool.map(_walk_chunk, chunks), total=len(chunks), desc="Walk chunks"):
            for entity, walks, error in result:
                if walks is None:
                    failed.append(entity)
                    continue
                for walk in walks:
                    f_walks.write(walk + "\n")
                zero += len(walks) == 0
                f_walked.write(entity + "\n")
            f_walks.flush()
            f_walked.flush()
    if zero:
        print(f"Found zero walks for {zero} entities")
    if failed:
        print(f"Failed to generate walks for {len(failed)} entities")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Train pyrdf2vec using disk-based walk storage with a SPARQL endpoint.")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint URL.")
    parser.add_argument("--endpoints", default=None,
                        help="Comma-separated SPARQL endpoints to walk in parallel (one process each); "
                             "default: --endpoint only.")
    parser.add_argument("--base_dir", default=None, help="Base directory containing query files.")
    parser.add_argument("--glob_pattern", default=None, help="Glob pattern to match files inside the base directory.")
    parser.add_argument("--entities_file", default=None,
                        help="JSON list of entities to embed, instead of reading them from query files.")
    parser.add_argument("--skip_entities_file", default=None,
                        help="JSON list of entities not to walk (e.g. covered by --extra_walks).")
    parser.add_argument("--extra_walks", nargs="*", default=[],
                        help="Existing walk files to add to the Word2Vec corpus (reused, not regenerated).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep walks.txt and only walk the entities not yet in walked_entities.txt.")
    parser.add_argument("--output", required=True, help="Output folder for walks and model.")
    parser.add_argument("--model_file_name", type=str, default="model.json",
                        help="File to where the model should be written")
    parser.add_argument("--num_walks", type=int, default=100, help="Number of walks per entity.")
    parser.add_argument("--depth", type=int, default=4, help="Depth of walks.")
    parser.add_argument("--dimensions", type=int, default=128, help="Embedding size.")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size.")
    parser.add_argument("--min_count", type=int, default=5, help="Word2Vec min count")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec epochs.")
    parser.add_argument("--workers", type=int, default=4, help="Word2Vec epochs.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    walks_file = os.path.join(args.output, "walks.txt")
    walked_file = os.path.join(args.output, "walked_entities.txt")
    endpoints = args.endpoints.split(",") if args.endpoints else [args.endpoint]

    # Read entities to embed
    if args.entities_file:
        with open(args.entities_file, 'r') as f:
            entities = json.load(f)
    else:
        # NEW: Find files using the base directory and glob pattern, filtering for .json files explicitly
        search_path = os.path.join(args.base_dir, args.glob_pattern)
        query_paths = [
            p for p in glob.glob(search_path, recursive=True)
            if p.endswith('.json') and os.path.isfile(p)
        ]

        print(f"Found {len(query_paths)} JSON files matching the pattern.")

        entities = set()
        for query_path in query_paths:
            with open(query_path, 'r') as f:
                raw_data = json.load(f)
            for i, data in tqdm(enumerate(raw_data), total=len(raw_data),
                                desc=f"Processing {os.path.basename(query_path)}"):
                _, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])
                for tp in tp_rdflib:
                    for entity in tp:
                        if isinstance(entity, URIRef):
                            entities.add(str(entity))
        entities = list(entities)

    skip = set()
    if args.skip_entities_file:
        with open(args.skip_entities_file, 'r') as f:
            skip = set(json.load(f))
    if args.resume and os.path.exists(walked_file):
        with open(walked_file, 'r') as f:
            skip |= {line.strip() for line in f if line.strip()}
    else:
        for path in (walks_file, walked_file):
            if os.path.exists(path):
                os.remove(path)
    to_walk = sorted(set(entities) - skip)
    print(f"Generating walks from {len(endpoints)} SPARQL endpoint(s) for {len(to_walk)} of {len(entities)} entities...")
    failed = generate_walks(to_walk, endpoints, walks_file, walked_file, args.depth, args.num_walks) if to_walk else []
    if failed:
        raise SystemExit(f"Walks failed for {len(failed)} entities (e.g. {failed[:3]}); rerun with --resume.")

    print("Training Word2Vec on walks...")
    sentences = MultiWalksCorpus(list(args.extra_walks) + [walks_file])
    missing_entities_walk, insufficient_entities_walk = (
        validate_completeness_walks(sentences, entities, args.min_count))
    print(f"Missing entities from walks: {missing_entities_walk} out of {len(entities)}")
    print(f"Entities with insufficient occurrences: {insufficient_entities_walk} out of {len(entities)}")

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
    missing_entities = validate_completeness_embeddings(model, entities)
    print(f"Missing {missing_entities} entities out of {len(entities)} entities")
    # Save to JSON file
    with open(os.path.join(args.output, args.model_file_name), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Updated Example command showcasing the new --base_dir and --glob_pattern interface
    # Example: python streaming_pyrdf2vec.py --endpoint http://localhost:9000 \
    # --output data/rdf2vec_embeddings/mixed_yago --base_dir ./data/generated_queries --glob_pattern "mixed_yago/*.json" \
    # --epochs 50 --num_walks 10 --workers 5
    main()