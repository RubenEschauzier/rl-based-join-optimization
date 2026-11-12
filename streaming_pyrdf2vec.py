import argparse
import logging
import multiprocessing
import os
import json
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from gensim.models import Word2Vec as GensimWord2Vec
from gensim.models.callbacks import CallbackAny2Vec
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


class Checkpoint(CallbackAny2Vec):
    """Save the model every `every` epochs (and after the last), so an interrupted training continues from there
    (--resume). gensim writes big arrays to extra .npy files next to the model, so the save goes to a temporary
    directory first; <path>.json (epochs done) is removed before the files are moved and written after, so a
    checkpoint counts only if it is complete."""

    def __init__(self, path, epochs_done, total_epochs, every, corpus):
        self.path, self.done, self.total, self.every = path, epochs_done, total_epochs, every
        self.corpus_bytes = os.path.getsize(corpus)

    def on_epoch_end(self, model):
        self.done += 1
        print(f"Word2Vec epoch {self.done}/{self.total} done", flush=True)
        if self.done % self.every and self.done != self.total:
            return
        directory, name = os.path.split(self.path)
        staging = self.path + ".saving"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        model.save(os.path.join(staging, name))
        if os.path.exists(self.path + ".json"):
            os.remove(self.path + ".json")
        for f in os.listdir(staging):
            os.replace(os.path.join(staging, f), os.path.join(directory, f))
        os.rmdir(staging)
        with open(self.path + ".json", "w") as f:
            json.dump({"epochs_done": self.done, "epochs": self.total, "corpus_bytes": self.corpus_bytes}, f)
        print(f"  checkpoint saved ({self.done} epochs)", flush=True)


def single_corpus_file(paths, output):
    """gensim's corpus_file mode reads one file (one walk per line, tokens separated by whitespace, as
    walks.txt is): several walk files are concatenated into <output>/corpus.txt once."""
    if len(paths) == 1:
        return paths[0]
    target = os.path.join(output, "corpus.txt")
    sizes = sum(os.path.getsize(p) for p in paths)
    if not (os.path.exists(target) and os.path.getsize(target) == sizes):
        print(f"Concatenating {len(paths)} walk files into {target}...", flush=True)
        with open(target + ".tmp", "wb") as out:
            for path in paths:
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, out, 64 * 2**20)
        os.replace(target + ".tmp", target)
    return target


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
    parser.add_argument("--workers", type=int, default=4,
                        help="Word2Vec threads (corpus_file mode: scales with cores, no GIL).")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                        help="Save a training checkpoint every this many epochs; --resume continues from it.")

    args = parser.parse_args()
    # gensim reports its progress (vocabulary scan, "EPOCH n - PROGRESS: at x% examples") through logging
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

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
    corpus = single_corpus_file(list(args.extra_walks) + [walks_file], args.output)
    checkpoint = os.path.join(args.output, "checkpoint.model")
    state = {}
    if args.resume and os.path.exists(checkpoint) and os.path.exists(checkpoint + ".json"):
        with open(checkpoint + ".json") as f:
            state = json.load(f)
        if state.get("corpus_bytes") != os.path.getsize(corpus) or state.get("epochs") != args.epochs:
            print(f"Ignoring {checkpoint}: trained on other walks or for another number of epochs")
            state = {}
    if state:
        model = GensimWord2Vec.load(checkpoint)
        done = state["epochs_done"]
        model.workers = args.workers
        print(f"Resuming Word2Vec from {checkpoint}: {done} of {args.epochs} epochs done")
    else:
        model = GensimWord2Vec(vector_size=args.dimensions, window=args.window, sg=1, workers=args.workers,
                               epochs=args.epochs, min_count=args.min_count)
        # corpus_file: gensim reads the walks in C, one thread per worker (the sentences iterator is GIL-bound)
        model.build_vocab(corpus_file=corpus, keep_raw_vocab=True)
        missing_entities_walk = sum(e not in model.raw_vocab for e in entities)
        insufficient_entities_walk = sum(0 < model.raw_vocab.get(e, 0) < args.min_count for e in entities)
        model.raw_vocab = {}                     # as gensim leaves it without keep_raw_vocab
        print(f"Missing entities from walks: {missing_entities_walk} out of {len(entities)}")
        print(f"Entities with insufficient occurrences: {insufficient_entities_walk} out of {len(entities)}")
        done = 0
    if done < args.epochs:
        # gensim lowers the learning rate linearly from alpha to min_alpha over all epochs: continue that line
        start_alpha = model.alpha - (model.alpha - model.min_alpha) * done / args.epochs
        model.train(corpus_file=corpus, total_words=model.corpus_total_words, epochs=args.epochs - done,
                    start_alpha=start_alpha, end_alpha=model.min_alpha, report_delay=60,
                    callbacks=[Checkpoint(checkpoint, done, args.epochs, args.checkpoint_every, corpus)])

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