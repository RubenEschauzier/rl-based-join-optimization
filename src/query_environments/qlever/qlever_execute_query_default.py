import asyncio
import aiohttp
import json


class QLeverOptimizerClient:
    def __init__(self, http_endpoint: str):
        """
        Initializes the client.
        :param http_endpoint: The HTTP URL (e.g., "http://localhost:7001")
        """
        self.http_endpoint = http_endpoint

    def _apply_join_order(self, query_obj, join_order: list[int]) -> str:
        """
        Recursively wraps triples in {} to force the join order, with proper indentation.
        """
        if not join_order:
            return query_obj["query"]

        triples = query_obj["triple_patterns"]

        def build_nested_bgp(indices, indent_level=1):
            indent = "  " * indent_level

            # Base case: Just one triple
            if len(indices) == 1:
                return f"{indent}{{ {triples[indices[0]]} }}"

            # Recursive case: Split into [rest] and [last]
            last_idx = indices[-1]
            remaining_indices = indices[:-1]

            # Nest the previous results and join with the new triple
            # We add newlines and indentation for readability
            inner_nested = build_nested_bgp(remaining_indices, indent_level + 1)
            last_triple = f"{indent}  {{ {triples[last_idx]} }}"

            return f"{indent}{{\n{inner_nested}\n{last_triple}\n{indent}}}"

        forced_bgp = build_nested_bgp(join_order)
        return f"SELECT * WHERE {{\n{forced_bgp}\n}}"

    async def execute_plan(self, query_obj, join_order: list[int] = None, timeout: str = "10s") -> dict:
        """
        Executes the query via HTTP and retrieves the full execution plan + costs.
        """
        formatted_query = self._apply_join_order(query_obj, join_order)
        print(formatted_query)
        # We use 'application/qlever-results+json' to get the runtimeInformation field
        headers = {
            "Accept": "application/qlever-results+json",
            "Content-Type": "application/sparql-query"
        }

        params = {
            # "query": formatted_query,
            "timeout": timeout
        }

        async with aiohttp.ClientSession() as session:
            try:
                # Note: Sending query as data with the sparql-query content type
                # is the most robust way to handle large queries.
                async with session.post(self.http_endpoint, params=params, headers=headers,
                                        data=formatted_query) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "results": result.get("res", []),
                            "runtime_info": result.get("runtimeInformation", {}),
                            "query_plan": result.get("queryExecutionPlan", ""),
                            "time_total": result.get("time", {}).get("total", "0ms")
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": error_text}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Query timed out"}
            except Exception as e:
                return {"success": False, "error": str(e)}


    def extract_join_sequence(self, qlever_result: dict) -> list:
        """
        Walks the QLever runtime_info tree and extracts all Join operations
        in chronological execution order (bottom-up).
        """
        if not qlever_result.get("success") or "runtime_info" not in qlever_result:
            return []

        root_node = qlever_result["runtime_info"].get("query_execution_tree", {})
        join_sequence = []

        def walk_tree(node):
            if not node:
                return

            # 1. Post-order traversal: recurse into children first.
            # This ensures we hit the deepest (earliest executed) joins first.
            for child in node.get("children", []):
                walk_tree(child)

            # 2. Check if the current node is a Join
            description = node.get("description", "")
            if description.startswith("Join"):
                join_sequence.append({
                    "operation": description,
                    "actual_op_time_ms": node.get("operation_time", 0),
                    "actual_total_time_ms": node.get("total_time", 0),
                    "actual_rows": node.get("result_rows", 0),
                    "estimated_op_cost": node.get("estimated_operation_cost", 0),
                    "estimated_total_cost": node.get("estimated_total_cost", 0),
                    "estimated_rows": node.get("estimated_size", 0),
                    "cache_status": node.get("cache_status", "")
                })

        walk_tree(root_node)
        return join_sequence

    def extract_rl_metrics(self, qlever_result: dict) -> dict | None:
        """Extracts the exact metrics requested for the RL environment."""
        joins = self.extract_join_sequence(qlever_result)
        if not joins:
            return None
        total_rows = 0
        per_join_rows = []

        for join in joins:
            total_rows += join["actual_rows"]
            per_join_rows.append(join["actual_rows"])

        time_string = qlever_result['time_total']
        if "ms" in time_string:
            latency = float(time_string.split("ms")[0]) / 1000
        elif "s" in time_string:
            print(time_string)
            latency = float(time_string.split("ms")[0])
        elif "m" in time_string:
            print(time_string)
            latency = float(time_string.split("m")[0]) * 60
        else:
            raise ValueError(f"Encountered unknown time_total value: {time_string}")

        return {
            "per_join_rows": per_join_rows,
            "total_cost": total_rows,
            "latency": latency,
        }

    def print_join_stats(self, joins: list):
        """Prints the join sequence in a readable table format."""
        print(
            f"\n{'Step':<6} | {'Time (Op/Tot)':<15} | {'Rows (Actual / Est.)':<22} | {'Cost Est. (Op/Tot)':<22} | {'Cache'}")
        print("-" * 95)
        for i, j in enumerate(joins):
            time_str = f"{j['actual_op_time_ms']}ms / {j['actual_total_time_ms']}ms"
            row_str = f"{j['actual_rows']} / {j['estimated_rows']}"
            cost_str = f"{j['estimated_op_cost']} / {j['estimated_total_cost']}"

            print(f"Join {i + 1:<1} | {time_str:<15} | {row_str:<22} | {cost_str:<22} | {j['cache_status']}")

if __name__ == "__main__":
    async def main():
        # Only HTTP endpoint is needed now
        client = QLeverOptimizerClient("http://localhost:8888")

        query_object = {
            "triple_patterns": [
                '?s <http://example.com/13000080> <http://example.com/12610595> .',
                '?s <http://example.com/13000080> <http://example.com/12772511> .',
                '?s <http://example.com/13000179> <http://example.com/12920312> .',
                '?s <http://example.com/13000179> <http://example.com/12939215> .',
                '?s <http://example.com/13000083> ?o4 .'
            ],
            "query": 'SELECT * WHERE { ... }'
        }

        # Force a specific join order
        result = await client.execute_plan(
            query_obj=query_object,
            join_order=[4, 3, 2, 1, 0],
            timeout="10s"
        )

        if result["success"]:
            joins_with_runtime_info = client.extract_join_sequence(result)
            client.print_join_stats(joins_with_runtime_info)
            # runtime_info contains the cost estimates for every node in your join tree
            print("Successfully retrieved cost estimates!")
            print(f"Total Execution Time: {result['time_total']}")
            print(f"Result object: {result}")
            # Accessing estimates: result['runtime_info']['cost_estimate']
        else:
            print(f"Query Failed: {result['error']}")


    asyncio.run(main())