def print_plan_tree(node, prefix="", is_last=True):
    """
    Recursively prints the query execution tree.
    """
    # Extract execution metrics
    desc = node.get('description', 'Unknown Node')
    rows = node.get('result_rows', 0)
    cost = node.get('estimated_operation_cost', 0)
    time = node.get('total_time', 0)

    # Format the current node's output
    connector = "└── " if is_last else "├── "
    label = f"{desc} [Rows: {rows} | Cost: {cost} | Time: {time}ms]"
    print(prefix + connector + label)

    # Update the prefix for child nodes
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Traverse children
    children = node.get('children', [])
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_plan_tree(child, child_prefix, is_last_child)


# The raw QLever execution plan
qlever_output = {'meta': {'time_query_planning': 1}, 'query_execution_tree': {'cache_status': 'computed', 'children': [{'cache_status': 'computed', 'children': [{'cache_status': 'computed', 'children': [{'cache_status': 'computed', 'children': [{'cache_status': 'computed', 'children': [], 'column_names': ['?s'], 'description': 'IndexScan POS ?s <http://example.com/13000080> <http://example.com/10724425>', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 6250, 'estimated_size': 6250, 'estimated_total_cost': 6250, 'operation_time': 1, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 1, 'status': 'fully materialized completed', 'total_time': 1}, {'cache_status': 'computed', 'children': [], 'column_names': ['?s'], 'description': 'IndexScan POS ?s <http://example.com/13000080> <http://example.com/8719681>', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 6250, 'estimated_size': 6250, 'estimated_total_cost': 6250, 'operation_time': 0, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 1, 'status': 'fully materialized completed', 'total_time': 0}], 'column_names': ['?s'], 'description': 'Join on ?s', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 16875, 'estimated_size': 4375, 'estimated_total_cost': 29375, 'operation_time': 0, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 1, 'status': 'fully materialized completed', 'total_time': 1}, {'cache_status': 'computed', 'children': [], 'column_names': ['?s'], 'description': 'IndexScan POS ?s <http://example.com/13000087> <http://example.com/2447036>', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 6250, 'estimated_size': 6250, 'estimated_total_cost': 6250, 'operation_time': 0, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 142, 'status': 'fully materialized completed', 'total_time': 0}], 'column_names': ['?s'], 'description': 'Join on ?s', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 13687, 'estimated_size': 3062, 'estimated_total_cost': 49312, 'operation_time': 1, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 1, 'status': 'fully materialized completed', 'total_time': 2}, {'cache_status': 'computed', 'children': [], 'column_names': ['?s'], 'description': 'IndexScan POS ?s <http://example.com/13000087> <http://example.com/2941846>', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 43750, 'estimated_size': 43750, 'estimated_total_cost': 43750, 'operation_time': 1, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 67656, 'status': 'fully materialized completed', 'total_time': 1}], 'column_names': ['?s'], 'description': 'Join on ?s', 'details': None, 'estimated_column_multiplicities': [1.0], 'estimated_operation_cost': 48955, 'estimated_size': 2143, 'estimated_total_cost': 142017, 'operation_time': 1, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 1, 'result_rows': 1, 'status': 'fully materialized completed', 'total_time': 4}, {'cache_status': 'computed', 'children': [], 'column_names': ['?s', '?o4'], 'description': 'IndexScan PSO ?s <http://example.com/13000087> ?o4', 'details': {'num-blocks-all': 72, 'num-blocks-read': 1, 'num-elements-read': 20724}, 'estimated_column_multiplicities': [9.023223876953125, 128.54287719726562], 'estimated_operation_cost': 2239474, 'estimated_size': 2239474, 'estimated_total_cost': 2239474, 'operation_time': 0, 'original_operation_time': 0, 'original_total_time': 0, 'result_cols': 2, 'result_rows': 20724, 'status': 'lazily materialized completed', 'total_time': 0}], 'column_names': ['?s', '?o4'], 'description': 'Join on ?s', 'details': {'time-for-filtering-blocks': 1}, 'estimated_column_multiplicities': [6.316256523132324, 1.10984468460083], 'estimated_operation_cost': 2255152, 'estimated_size': 13535, 'estimated_total_cost': 4636643, 'operation_time': 2, 'original_operation_time': 2, 'original_total_time': 6, 'result_cols': 2, 'result_rows': 12, 'status': 'lazily materialized completed', 'total_time': 6}}

if __name__ == "__main__":
    root_node = qlever_output.get('query_execution_tree', {})
    print("QLever Execution Plan:")
    print_plan_tree(root_node)