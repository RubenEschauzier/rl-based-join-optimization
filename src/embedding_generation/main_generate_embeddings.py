import json
import os
import pickle

from onehot_entity_embedding import main_onehot_embedding
from rdf2vec_entity_embedding import main_rdf2vec, rdf2vec_embedding_low_memory, train_model

if __name__ == "__main__":
    project_root = os.getcwd()
    dataset_name = "lubm"
    instantiation_benchmark_location = fr"{project_root}/large_data/lubm.nt"
    temp_save_loc_template = "walks_{}.txt"

    project_root = os.getcwd()
    use_saved_walks = False
    onehot_embed = False
    rdf2vec_embed = True
    memory_save = False

    # Rdf2Vec params
    n_sim_pred = 1000
    n_sim_subj = 100
    n_sim_obj = 100
    depth_walk = 3
    rdf2vec_vector_save_location = os.path.join(project_root,
                                                "data", "rdf2vec_embeddings", dataset_name,
                                                "rdf2vec_vectors_{}_depth_{}_test.txt".format(dataset_name, depth_walk))

    if use_saved_walks:
        walks = []
        for i in range(3):
            with open(temp_save_loc_template.format(i), 'rb') as f:
                walks.extend(pickle.load(f))
        model = train_model(walks)
        model.wv.save_word2vec_format("temp_vectors.txt", binary=False)
        vector_dict = {}
        for word, vector in zip(model.wv.index_to_key, model.wv.vectors):
            vector_dict[word] = vector
        with open(rdf2vec_vector_save_location.replace('.txt', '.json'), 'w', encoding='utf-8') as f:
            json.dump(vector_dict, f, ensure_ascii=False, indent=2)

    elif onehot_embed:
        onehot_embedding_save_location = os.path.join(project_root, "output", "entity_embeddings",
                                                      "embeddings_onehot_encoded.txt")
        main_onehot_embedding(instantiation_benchmark_location, onehot_embedding_save_location)

    elif rdf2vec_embed and memory_save:

        rdf2vec_embedding_low_memory(n_sim_pred, n_sim_subj, n_sim_obj, depth_walk,
                     instantiation_benchmark_location, rdf2vec_vector_save_location, temp_save_loc_template)

    elif rdf2vec_embed:
        main_rdf2vec(n_sim_pred, n_sim_subj, n_sim_obj, depth_walk,
                     instantiation_benchmark_location, rdf2vec_vector_save_location)

    else:
        raise NotImplementedError("The boolean combination is not implemented.")