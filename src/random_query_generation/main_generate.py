from src.random_query_generation.generate_path import main_sample_paths

if __name__ == "__main__":
    main_sample_paths(endpoint_url="http://localhost:8890/sparql",
                      default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
                      n_paths=2,
                      max_size=5,
                      proportion_unique_predicates=.97,
                      path_start_type="?s",
                      output_dir=r"C:\Users\ruben\projects\rl-based-join-optimization\data\pretrain_data"
                                 r"\generated_queries",
                      file_name_literal="path_with_literal.json",
                      file_name_non_literal="path_without_literal.json")
