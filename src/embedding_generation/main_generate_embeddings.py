import os

from onehot_entity_embedding import main_onehot_embedding
from rdf2vec_entity_embedding import main_rdf2vec

if __name__ == "__main__":
    dataset_name = "wikidata"
    instantiation_benchmark_location = r"C:\Users\ruben\projects\rl-based-join-optimization\large_data\wikidata.ttl"

    project_root = os.getcwd()
    onehot_embed = False
    rdf2vec_embed = True

    if onehot_embed:
        onehot_embedding_save_location = os.path.join(project_root, "output", "entity_embeddings",
                                                      "embeddings_onehot_encoded.txt")
        main_onehot_embedding(instantiation_benchmark_location, onehot_embedding_save_location)

    if rdf2vec_embed:
        n_sim_pred = 1000
        n_sim_subj = 100
        n_sim_obj = 100
        depth_walk = 3
        rdf2vec_vector_save_location = os.path.join(project_root, "data", "rdf2vec_embeddings", dataset_name,
                                                    "rdf2vec_vectors_{}_depth_{}.txt".format(dataset_name, depth_walk))

        main_rdf2vec(n_sim_pred, n_sim_subj, n_sim_obj, depth_walk,
                     instantiation_benchmark_location, rdf2vec_vector_save_location)