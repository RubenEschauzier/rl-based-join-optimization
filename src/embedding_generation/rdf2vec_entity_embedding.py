import json
import os
import pickle
from time import sleep
import logging

from rdflib.graph import Graph
from gensim.models import Word2Vec
import itertools
import random
from tqdm import tqdm


def get_all_predicates(g):
    all_predicate_occurrence = []
    predicates = set()
    entities = set()
    subjects = set()
    objects = set()
    predicates_uri = set()

    for (subj, pred, obj) in g:
        predicates.add(str(pred))
        predicates_uri.add(pred)
        entities.add(subj)
        entities.add(obj)
        subjects.add(subj)
        objects.add(obj)
        all_predicate_occurrence.append(pred)

    predicates = list(predicates)
    predicates_uri = list(predicates_uri)
    return predicates, predicates_uri, entities, all_predicate_occurrence, subjects, objects


def get_all_subj_obj(g):
    subj_dict = {}
    obj_dict = {}
    for (subj, pred, obj) in g:
        if subj in subj_dict:
            subj_dict[subj] += 1
        else:
            subj_dict[subj] = 1
        if obj in obj_dict:
            obj_dict[obj] += 1
        else:
            obj_dict[obj] = 1
    return subj_dict, obj_dict


def load_graph(dataset_location):
    g = Graph()
    with open(dataset_location, "r") as f:
        g.parse(f, format="nt")
    return g


def generate_walks(g, predicates_uri, subjects_uri, objects_uri, num_sim_pred, num_sim_subj, num_sim_obj, depth_walk,):
    all_walks = set()
    for predicate_uri in tqdm(predicates_uri):
        triples_with_predicate = g.triples((None, predicate_uri, None))
        tp = list(triples_with_predicate)
        walks_predicate, num_walks_predicate = generate_walks_from_start_triple(g, num_sim_pred, tp, depth_walk)

        walks_predicate_tuple = [tuple(x) for x in walks_predicate]
        if len(walks_predicate_tuple) == 0:
            print("Failed to generate walks for object {}".format(predicate_uri))

        all_walks.update(walks_predicate_tuple)
    sleep(1)
    print("Number of unique walks after adding walks starting at predicates in graph: {}".format(len(all_walks)))

    for subject_uri in tqdm(subjects_uri):
        triples_with_subject = g.triples((subject_uri, None, None))
        tp = list(triples_with_subject)
        walks_subject, num_walks_subject = generate_walks_from_start_triple(g, num_sim_subj, tp, depth_walk)

        walks_subject_tuple = [tuple(x) for x in walks_subject]
        if len(walks_subject_tuple) == 0:
            print("Failed to generate walks for subject {}".format(subject_uri))
        all_walks.update(walks_subject_tuple)

    sleep(1)
    print("Number of unique walks after adding walks starting at subjects in graph: {}".format(len(all_walks)))

    for object_uri in tqdm(objects_uri):
        triples_with_object = g.triples((None, None, object_uri))
        tp = list(triples_with_object)
        walks_object, num_walks_object = generate_walks_from_start_triple(g, num_sim_obj, tp, depth_walk)

        walks_object_tuple = [tuple(x) for x in walks_object]
        if len(walks_object_tuple) == 0:
            print("Failed to generate walks for object {}".format(object_uri))

        all_walks.update(walks_object_tuple)

    sleep(1)
    print("Number of unique walks after adding walks starting at objects in graph: {}".format(len(all_walks)))
    return all_walks

def generate_walks_to_file(g, predicates_uri, subjects_uri, objects_uri, num_sim_pred, num_sim_subj, num_sim_obj,
                           depth_walk, save_walk_template):
    predicate_based_walks = set()
    for predicate_uri in tqdm(predicates_uri):
        triples_with_predicate = g.triples((None, predicate_uri, None))
        tp = list(triples_with_predicate)
        walks_predicate, num_walks_predicate = generate_walks_from_start_triple(g, num_sim_pred, tp, depth_walk)

        walks_predicate_tuple = [tuple(x) for x in walks_predicate]
        predicate_based_walks.update(walks_predicate_tuple)

    with open(save_walk_template.format(0), "wb") as f:
        pickle.dump(predicate_based_walks, f)

    print("Number of walks predicate tuple: {}".format(len(predicate_based_walks)))

    subject_based_walks = set()
    for subject_uri in tqdm(subjects_uri):
        triples_with_subject = g.triples((subject_uri, None, None))
        tp = list(triples_with_subject)
        walks_subject, num_walks_subject = generate_walks_from_start_triple(g, num_sim_subj, tp, depth_walk)

        walks_subject_tuple = [tuple(x) for x in walks_subject]
        subject_based_walks.update(walks_subject_tuple)

    with open(save_walk_template.format(1), "wb") as f:
        pickle.dump(subject_based_walks, f)
    print("Number of walks subject: {}".format(len(subject_based_walks)))

    objects_based_walks = set()
    for object_uri in tqdm(objects_uri):
        triples_with_object = g.triples((None, None, object_uri))
        tp = list(triples_with_object)
        walks_object, num_walks_object = generate_walks_from_start_triple(g, num_sim_obj, tp, depth_walk)

        walks_object_tuple = [tuple(x) for x in walks_object]
        objects_based_walks.update(walks_object_tuple)

    with open(save_walk_template.format(2), "wb") as f:
        pickle.dump(objects_based_walks, f)
    print("Number of walks subject: {}".format(len(objects_based_walks)))


def generate_walks_from_start_triple(g, num_sim, tp, max_depth_walk):
    walks_from_start_triple = []
    num_walks_generated = 0
    for i in range(num_sim):
        num_triples_chosen = 0
        walk_in_progress = []
        # Choose random triple from our list of triples
        index = random.randrange(len(tp))
        start_triple = tp[index]
        start_subject = start_triple[0]
        # This is the triple that the subject belongs to, for ordering
        start_object = start_triple[2]
        # This is the triple that the object belongs to, for ordering
        walk_in_progress.extend(list(start_triple))

        while num_triples_chosen < max_depth_walk:
            # Get all triples with starting object as subject
            start_object_as_subject = list(g.triples((start_object, None, None)))
            # All triples with starting subject as object
            start_subject_as_object = list(g.triples((None, None, start_subject)))
            # Merge all possible triples
            possible_triples = list(itertools.chain(start_object_as_subject, start_subject_as_object))
            if len(possible_triples) == 0:
                break
            # Get new triple to walk to
            new_index = random.randrange(len(possible_triples))
            start_triple = possible_triples[new_index]
            if new_index < len(start_object_as_subject):
                start_object = start_triple[2]
                walk_in_progress.extend(list(start_triple))
            else:
                extended_walk = list(start_triple)
                start_subject = start_triple[0]
                extended_walk.extend(walk_in_progress)
                walk_in_progress = extended_walk

            # walk_in_progress.extend(list(start_triple))
            num_triples_chosen += 1

        # Any walk with atleast a triple is valid for input
        if len(walk_in_progress) >= 3:
        # if len(walk_in_progress) == max_depth_walk * 3 + 3:
            # Remove the matching subject - object in walk
            last_element = walk_in_progress[-1]
            del walk_in_progress[2::3]
            walk_in_progress.append(last_element)
            walks_from_start_triple.append(walk_in_progress)
            num_walks_generated += 1
    return walks_from_start_triple, num_walks_generated


def get_all_subj_obj_in_walks(g, walks):
    subj_dict, obj_dict = get_all_subj_obj(g)
    subj_dict_walk = {}
    obj_dict_walk = {}
    for walk in walks:
        subjects = walk[0:-1:2]
        objects = walk[2::2]
        for subj in subjects:
            if subj in subj_dict_walk:
                subj_dict_walk[subj] += 1
            else:
                subj_dict_walk[subj] = 1
        for obj in objects:
            if obj in obj_dict_walk:
                obj_dict_walk[obj] += 1
            else:
                obj_dict_walk[obj] = 1
    return subj_dict_walk, obj_dict_walk, subj_dict, obj_dict


def get_average_difference_occurrences(subj_dict_walk, subj_dict, obj_dict_walk, obj_dict):
    total_difference_subj = 0
    total_missing_subj = 0
    for key, value in subj_dict.items():
        diff = value
        if key in subj_dict_walk:
            diff = value - subj_dict_walk[key]
        else:
            total_missing_subj += 1
        total_difference_subj += diff
    avg_diff_subj = total_difference_subj / len(subj_dict.items())

    total_difference_obj = 0
    total_missing_obj = 0
    for key, value in obj_dict.items():
        diff = value
        if key in obj_dict_walk:
            diff = value - obj_dict_walk[key]


def train_model(walks):
    corpus = [[str(word) for word in walk] for walk in tqdm(walks)]
    vector_dim = 128
    print("Training model...")
    # Enable logging to see progress
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec(corpus, min_count=1, window=5, vector_size=vector_dim, epochs=50, workers=6)
    return model


def save_rdf_predicate_model(model, predicates, location):
    vector_dim = 128
    with open(location, 'w') as f:
        # First write predicates to file
        for predicate in predicates:
            if predicate in model.wv:
                to_write = str(predicate) + '[sep]' + ' '.join([str(x) for x in model.wv[predicate]])
                f.write(to_write)
                f.write('\n')
                pass
            else:
                to_write = str(predicate) + '[sep]' + ' '.join(['0' for _ in range(vector_dim)])
                f.write(to_write)
                f.write('\n')
        # Write non-predicates to file
        for key in model.wv.index_to_key:
            if key not in predicates:
                to_write = str(key) + '[sep]' + ' '.join([str(x) for x in model.wv[key]])
                f.write(to_write)
                f.write('\n')

def convert_text_to_json(text_file_location, output_location):
    vector_dict = {}
    with open(text_file_location, 'r') as file:
        data = file.read().strip().split('\n')

    for entity in data:
        split = entity.split('[sep]')
        vector = [float(x) for x in split[1].strip().split(' ')]
        vector_dict[split[0]] = vector

    with open(output_location, 'w', encoding='utf-8') as f:
        json.dump(vector_dict, f, ensure_ascii=False, indent=4)


def rdf2vec_embedding(num_sim_pred, num_sim_subj, num_sim_obj, depth_walk, dataset_location, output_location,
                      save_model=True):
    g = load_graph(dataset_location)
    predicates, predicates_uri, entities, all_pred_occurrences, subjects_uri, objects_uri = get_all_predicates(g)
    walks = generate_walks(g, predicates_uri, subjects_uri, objects_uri,
                           num_sim_pred, num_sim_subj, num_sim_obj,
                           depth_walk)

    model = train_model(walks)
    if save_model:
        save_rdf_predicate_model(model, predicates, output_location)
        convert_text_to_json(output_location, output_location.replace('.txt', '.json'))

# Ugly solution. Easy to fix: Use the pyrdf2vec library which uses an endpoint.
def rdf2vec_embedding_low_memory(num_sim_pred, num_sim_subj, num_sim_obj, depth_walk, dataset_location, output_location,
                                 walk_file_template, save_model=True):
    g = load_graph(dataset_location)
    predicates, predicates_uri, entities, all_pred_occurrences, subjects_uri, objects_uri = get_all_predicates(g)
    # Separating the walk generation in parts and the model training should hopefully encourage the GC to discard
    # previously generated walks, thus allowing memory usage to be reduced. This is only a problem for yago (possibly
    # also wikidata). Again, for future reference, just use the library with an endpoint..
    generate_walks_to_file(g, predicates_uri, subjects_uri, objects_uri,
                           num_sim_pred, num_sim_subj, num_sim_obj,
                           depth_walk, walk_file_template)
    walks = []
    for i in range(3):
        with open(walk_file_template.format(i), 'rb') as f:
            walks.extend(pickle.load(f))

    model = train_model(walks)
    if save_model:
        save_rdf_predicate_model(model, predicates, output_location)
        convert_text_to_json(output_location, output_location.replace('.txt', '.json'))


def main_rdf2vec(num_sim_pred, num_sim_subj, num_sim_obj, depth_walk, instantiation_benchmark_location,
                 vector_save_location):
    rdf2vec_embedding(num_sim_pred, num_sim_subj, num_sim_obj, depth_walk,
                      instantiation_benchmark_location,
                      vector_save_location,
                      True)

