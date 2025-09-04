import json
import os
import subprocess

def main_cardinality_estimation_g_care(executable, method, query_location):

    query_to_directory = read_json_dict(os.path.join(queries_location, 'query_to_file.json'))

    for query, query_dir in query_to_directory.items():
        sub_query_to_file = read_json_dict(os.path.join(queries_location, query_dir, 'sub_query_to_file.json'))
        for sub_query_file in sub_query_to_file.values():
            sub_query_loc = os.path.join(query_location, query_dir, sub_query_file)
            cmd = [
                executable,
                "-q",
                "-m", method
            ]
            print(sub_query_loc)
        print(sub_query_to_file)
        break


def build_summary_all_methods(executable, methods_g_care, input_dataset, output_location_graph_and_summary_template):

    for method in methods_g_care:
        output_location_graph_and_summary = output_location_graph_and_summary_template.format(method)
        build_binary_graph_and_summary(executable, 
                                       method, 
                                       input_dataset, 
                                       output_location_graph_and_summary)
        break

def get_cardinality_estimates_g_care(query, methods):
    pass

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
    input_location_dataset = f"{project_root}/data/benchmark_g_care_format/{dataset}/{dataset}.txt".format(dataset, dataset)
    # Template: Fill in with actual method being used
    output_location_graph_and_summary_templ = f"{project_root}/data/benchmark_g_care_format/{dataset}/{"{}"}/"
    build_first = False
    if build_first:
        build_summary_all_methods(g_care_graph_executable, 
                                  methods_g_care_graph, 
                                  input_location_dataset, 
                                  output_location_graph_and_summary_templ)
        
    queries_location = f"{project_root}/data/query_enumeration_g_care/{query_name}/{dataset}"
    main_cardinality_estimation_g_care(queries_location)
    
