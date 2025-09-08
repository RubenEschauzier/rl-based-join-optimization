from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import subprocess

from tqdm import tqdm


def get_cardinality_method_fast(executable, method, query_location, data_location, output_location,
                                filename_to_qkey, queries):
    output_file = os.path.join(output_location, f"{method}_estimation_result.txt")
    cmd = [
        executable,
        "-q",
        "-m", method,
        "-i", query_location,
        "-d", data_location,
        "-o", output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Method {method} failed: {result.stderr}")

    return map_output_to_queries(output_file, filename_to_qkey, queries)


def map_output_to_queries(output_file, filename_to_qkey, queries):
    cardinalities = {query: {} for query in queries}
    with open(output_file, "r") as f:
        for line in f:
            if line.startswith("/"):  # lines with file paths
                parts = line.strip().split()
                filepath = parts[0]
                filename = filepath.split("/")[-1] 
                cardinality = float(parts[1])      
                elapsed = float(parts[2])

                if filename in filename_to_qkey:
                    query, key = filename_to_qkey[filename]
                    cardinalities[query][key] = (cardinality, elapsed)
    return cardinalities

def run_cardinality(args):
    exe, method, query_path, data_location_tmpl, output_dir, file_to_qkey, queries = args
    return method, get_cardinality_method_fast(
        exe, method,
        query_path, data_location_tmpl.format(method),
        output_location=output_dir,
        filename_to_qkey=file_to_qkey,
        queries=queries
    )

def main_fast(exe, methods, query_path, output_dir, data_location_tmpl):
    os.makedirs(output_dir, exist_ok=True)

    query_to_keys_to_sub_query_file = read_json_dict(
        os.path.join(query_path, 'query_to_sub_query_file.json'))
    file_to_qkey = build_filename_to_qkey(query_to_keys_to_sub_query_file)
    queries = {query: {} for query in query_to_keys_to_sub_query_file}

    query_to_method_to_sub_query_key = {query: create_method_dict(methods) 
                                        for query in query_to_keys_to_sub_query_file}
    
    # inside your main_fast
    tasks = [
        (exe, method, query_path, data_location_tmpl, output_dir, file_to_qkey, queries)
        for method in methods
    ]

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_cardinality, t) for t in tasks]
        for future in as_completed(futures):
            method, cardinality_result = future.result()
            for query, cardinality_keys in cardinality_result.items():
                query_to_method_to_sub_query_key[query][method] = cardinality_keys

    # for method in methods:
    #     cardinality_result = get_cardinality_method_fast(exe, method, 
    #                                 query_path, data_location_tmpl.format(method),
    #                                 output_location=output_dir,
    #                                 filename_to_qkey=file_to_qkey,
    #                                 queries=queries)
    #     for query, cardinality_keys in cardinality_result.items():
    #         query_to_method_to_sub_query_key[query][method] = cardinality_keys

    with open(os.path.join(output_dir, 'cardinalities.json'), 'w', encoding='utf-8') as f:
        json.dump(query_to_method_to_sub_query_key, f, indent=2)

    return query_to_method_to_sub_query_key

def create_method_dict(methods):
    return {method: {} for method in methods}


def build_filename_to_qkey(query_to_keys_to_sub_query_file):
    filename_to_qkey = {}
    for query, submap in query_to_keys_to_sub_query_file.items():
        for key, fname in submap.items():
            filename_to_qkey[fname] = (query, key)
    return filename_to_qkey


def build_summary_all_methods(executable, methods_g_care, input_dataset, output_location_graph_and_summary_template):
    for method in methods_g_care:
        output_location_graph_and_summary = output_location_graph_and_summary_template.format(method)
        build_binary_graph_and_summary(executable, 
                                       method, 
                                       input_dataset, 
                                       output_location_graph_and_summary)


def build_binary_graph_and_summary(executable, method, input_location, output_location_graph_and_summary):
    os.makedirs(output_location_graph_and_summary, exist_ok=True)
    cmd = [
        executable,
        "-b",
        "-m", method,
        "-i", input_location,
        "-d", output_location_graph_and_summary,
        "-o", os.path.join(output_location_graph_and_summary, f"{method}_build_result.txt" ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

def read_json_dict(loc):
    with open(loc, 'r') as f:
        return json.load(f)
    
if __name__ == "__main__":
    dataset = "lubm"
    query_name = "path_lubm"
    methods_g_care_graph = ["cset", "impr", "sumrdf", "wj", "jsub"]
    # This assumes the script is run from the root
    project_root = os.getcwd()
    g_care_graph_executable = "/root/projects/gcare/build/gcare_graph"
    input_location_dataset = f"{project_root}/data/benchmark_g_care_format/{dataset}/{dataset}.txt"
    # Template: Fill in with actual method being used
    output_location_graph_and_summary_templ = f"{project_root}/data/benchmark_g_care_format/{dataset}/{{}}/"
    output_location_query_estimation = f"{project_root}/data/query_enumeration_g_care/{query_name}_output/"
    build_first = False
    if build_first:
        build_summary_all_methods(g_care_graph_executable, 
                                  methods_g_care_graph, 
                                  input_location_dataset, 
                                  output_location_graph_and_summary_templ)
        
    queries_location = f"{project_root}/data/query_enumeration_g_care/{query_name}"

    # TEMP TESTING:
    # query_to_keys_to_sub_query_file = read_json_dict(
    #     os.path.join(queries_location, 'query_to_sub_query_file.json'))
    
    # file_to_qkey = build_filename_to_qkey(query_to_keys_to_sub_query_file)
    # queries = {query: {} for query in query_to_keys_to_sub_query_file}

    # result_card = map_output_to_queries(
    #     os.path.join(output_location_query_estimation, "cset_estimation_result.txt"), 
    #     file_to_qkey, queries
    #     )
    # print(result_card["SELECT * WHERE { <http://example.org/125122> <http://example.org/8> ?o1 . ?o1 <http://example.org/12> ?o2 }"])
    
    main_fast(g_care_graph_executable, methods_g_care_graph, 
              queries_location, output_location_query_estimation, output_location_graph_and_summary_templ)
    
