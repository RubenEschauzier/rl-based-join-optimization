import os

from onehot_entity_embedding import main_onehot_embedding
from rdf2vec_entity_embedding import main_rdf2vec

if __name__ == "__main__":
    # This is a system-wide instantiation of watdiv used for every project. This is to prevent mixing watdiv
    # instantiations and queries causing invalid benchmark results. For use outside of my (Ruben's) computer change
    # this. This is a minimal working example, depending on how well this works I might change it to use pyrdf2vec
    # And thus a sparql endpoint (But if you're reading this I was lazy).
    instantiation_benchmark_location = r"C:\Users\ruben\benchmarks\watdiv\dataset.nt"

    project_root = os.getcwd()
    onehot_embed = False
    rdf2vec_embed = True

    if onehot_embed:
        onehot_embedding_save_location = os.path.join(project_root, "output", "entity_embeddings",
                                                      "embeddings_onehot_encoded.txt")
        main_onehot_embedding(instantiation_benchmark_location, onehot_embedding_save_location)

    if rdf2vec_embed:
        n_sim_pred = 1000
        n_sim_subj = 200
        n_sim_obj = 200
        depth_walk = 2
        rdf2vec_vector_save_location = os.path.join(project_root, "data", "output", "entity_embeddings",
                                                    "rdf2vec_vectors_depth_{}.txt".format(depth_walk))

        main_rdf2vec(n_sim_pred, n_sim_subj, n_sim_obj, depth_walk,
                     instantiation_benchmark_location, rdf2vec_vector_save_location)