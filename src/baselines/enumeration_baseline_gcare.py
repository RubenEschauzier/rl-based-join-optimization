import json
import os
import subprocess

from tqdm import tqdm


def get_cardinality_method(executable, method, sub_query_location, data_location, output_location,
                           query_key_mapping):
    cmd = [
        executable,
        "-q",
        "-m", method,
        "-i", sub_query_location,
        "-d", data_location,
        "-o", os.path.join(output_location, f"{method}_estimation_result.txt")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return map_output_to_sub_query_key(query_key_mapping, os.path.join(output_location, f"{method}_estimation_result.txt"))


def map_output_to_sub_query_key(query_key_mapping, output_file_location):
    pred_results = {}
    normalized_mapping = {
        k: v.replace("\\", "/").split("/")[-1]  # keep only filename like sub_query_0.txt
        for k, v in query_key_mapping.items()
    }
    with open(output_file_location, "r") as f:
        for line in f:
            # check if this line is a path-containing line
            if line.startswith("/"):
                parts = line.strip().split()
                filepath = parts[0]
                # get just the file name from output
                filename = filepath.split("/")[-1] 
                # first number after file (average cardinality estimate)
                cardinality = float(parts[1])      
                elapsed = float(parts[2])
                # find which key in mapping this belongs to
                for k, fname in normalized_mapping.items():
                    if fname == filename:
                        pred_results[k] = [cardinality, elapsed]
                        break
    os.remove(output_file_location)
    os.remove(output_file_location+".err")
    return pred_results



def build_summary_all_methods(executable, methods_g_care, input_dataset, output_location_graph_and_summary_template):

    for method in methods_g_care:
        output_location_graph_and_summary = output_location_graph_and_summary_template.format(method)
        build_binary_graph_and_summary(executable, 
                                       method, 
                                       input_dataset, 
                                       output_location_graph_and_summary)

def get_cardinality_estimates_g_care(executable, methods, query_path, data_location_tmpl):

    method_to_sub_query_cardinalities = {}
    os.makedirs(os.path.join(query_path, "pred"), exist_ok=True)
    sub_query_to_dir = read_json_dict(os.path.join(query_path, 'sub_query_to_file.json'))

    for method in methods:

        sub_query_to_cardinality = get_cardinality_method(executable, method, 
                               os.path.join(query_path, "sub_queries"),
                               data_location_tmpl.format(method),
                               os.path.join(query_path, "pred"),
                               sub_query_to_dir)
        method_to_sub_query_cardinalities[method] = sub_query_to_cardinality
    return method_to_sub_query_cardinalities

def main(executable, methods, queries_location, data_location_tmpl):
    query_to_directory = read_json_dict(os.path.join(queries_location, 'query_to_file.json'))
    for _, query_dir in tqdm(query_to_directory.items()):
        m_to_sub_query_card = get_cardinality_estimates_g_care(executable, 
                                         methods, 
                                         os.path.join(queries_location, query_dir),
                                         data_location_tmpl
                                         )
        with open(os.path.join(queries_location, query_dir, "cardinalities.json"), 'w') as f:
            json.dump(m_to_sub_query_card, f)

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
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)


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
    build_first = False
    if build_first:
        build_summary_all_methods(g_care_graph_executable, 
                                  methods_g_care_graph, 
                                  input_location_dataset, 
                                  output_location_graph_and_summary_templ)
        
    queries_location = f"{project_root}/data/query_enumeration_g_care/{query_name}"

    main(g_care_graph_executable, methods_g_care_graph, queries_location, output_location_graph_and_summary_templ)
    
