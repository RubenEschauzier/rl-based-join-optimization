import asyncio
import logging
import math
import os

import aiohttp
import json
import ray


@ray.remote
class QLeverOptimizerClientRay:
    def __init__(self, http_endpoint: str):
        """
        Initializes the client as a stateful Ray Actor.
        :param http_endpoint: The HTTP URL (e.g., "http://localhost:7001")
        """
        self.http_endpoint = http_endpoint
        self.query_timeouts = {}
        self.default_timeout = "60s"
        self.default_timeout_s = 60
        self._session: aiohttp.ClientSession | None = None

        self.logger = logging.getLogger(f"QLeverActor.{http_endpoint}")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join("logs", f"actor_{http_endpoint.replace('/', '_')}.log"))
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self.logger.addHandler(handler)

    async def create_session(self):
        # Limit connections to prevent overwhelming a single-core target
        connector = aiohttp.TCPConnector(limit=10)
        self._session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            await self.create_session()
        return self._session

    async def execute_plan(self, query_obj, join_order: list[int] = None, timeout: str = "10s", parse_local=True,) -> dict:
        return await self._execute(query_obj, join_order, timeout, parse_local)

    async def _execute(self, query_obj, join_order: list[int] = None, timeout: str = "10s", parse_local=True,) -> dict:
        """
        Executes the query via HTTP and retrieves the full execution plan + costs.
        """
        # timeout = self.default_timeout
        # if query_obj["query"] in self.query_timeouts:
        #     timeout = self.format_latency(self.query_timeouts[query_obj["query"]])

        formatted_query = self._apply_join_order(query_obj, join_order)
        # We use 'application/qlever-results+json' to get the runtimeInformation field
        headers = {
            "Accept": "application/qlever-results+json",
            "Content-Type": "application/sparql-query"
        }

        # Add an internal fallback timeout slightly larger than the QLever timeout
        client_timeout = aiohttp.ClientTimeout(total=self.default_timeout_s + 30)
        params = {
            "timeout": timeout,
            # Ensure empty response to save memory
            "send": 0
        }
        session = await self._get_session()
        result = None
        try:
            # Note: Sending query as data with the sparql-query content type
            # is the most robust way to handle large queries.
            async with session.post(self.http_endpoint, params=params, headers=headers,
                                    data=formatted_query, timeout=client_timeout) as response:
                if response.status == 200:
                    result = await response.json()

                    response_size_mb = len(json.dumps(result).encode('utf-8')) / 1024 ** 2
                    self.logger.debug(f"[MEM] response_payload={response_size_mb:.1f}MB")
                    # time_total = result.get("time", {}).get("total", "0ms")

                    # # Tighten bounds on successful execution
                    # if time_total != "0ms":
                    #     time_in_seconds = self.decode_to_seconds(time_total)
                    #     self.query_timeouts[query_obj["query"]] = max(min((time_in_seconds * 2), self.default_timeout_s),
                    #                                                1)
                    result = {
                        "success": True,
                        "runtime_info": result.get("runtimeInformation", {}),
                        "query_plan": result.get("queryExecutionPlan", ""),
                        "time_total": result.get("time", {}).get("total", "0ms")
                    }
                else:
                    error_text = await response.text()
                    result = {"success": False, "error": error_text}

        except asyncio.TimeoutError:
            result = {"success": False, "error": "Query timed out"}
        except Exception as e:
            result = {"success": False, "error": str(e)}

        if parse_local:
            return self.extract_signal(result)
        else:
            return result

    def _walk_tree(self, node: dict, join_sequence: list):
        """Recursively extracts Join operations via post-order traversal."""
        if not node:
            return

        for child in node.get("children", []):
            self._walk_tree(child, join_sequence)

        description = node.get("description", "")
        if description.startswith("Join"):
            status = node.get("status", "")

            # Check if the join actually finished computing its rows
            is_completed = "completed" in status.lower()

            join_sequence.append({
                "operation": description,
                "status": status,

                # Latency of operation, even if failed, this time represents actual CPU effort spent.
                # It serves as a great lower bound for your value function.
                "actual_op_time_ms": node.get("operation_time", 0),

                # Only trust the row count if the operation fully completed.
                "actual_rows": node.get("result_rows", 0),
                "is_cardinality_valid": is_completed,

                "estimated_op_cost": node.get("estimated_operation_cost", 0),
                "estimated_rows": node.get("estimated_size", 0),
            })

    def _extract_join_sequence_success(self, qlever_result: dict) -> list:
        """Parses joins from a standard successful execution payload."""
        if not qlever_result.get("success") or "runtime_info" not in qlever_result:
            return []

        root_node = qlever_result["runtime_info"].get("query_execution_tree", {})
        join_sequence = []
        self._walk_tree(root_node, join_sequence)
        return join_sequence

    def _extract_join_sequence_fallback(self, parsed_qlever_result: dict) -> list:
        """Parses joins from an error or timeout payload."""
        root_node = parsed_qlever_result["runtimeInformation"]
        join_sequence = []
        self._walk_tree(root_node, join_sequence)
        return join_sequence

    def extract_join_sequence(self, qlever_result: dict) -> list:
        """
        Attempts to extract joins from a successful execution,
        falling back to error parsing if the standard extraction fails.
        """
        joins = self._extract_join_sequence_success(qlever_result)
        if not joins:
            if 'error' not in qlever_result:
                raise ValueError("Unknown qlever result structure")
            joins = self._extract_join_sequence_fallback(qlever_result['error'])
        return joins

    def extract_query_success(self, qlever_result: dict) -> dict:
        joins = self._extract_join_sequence_success(qlever_result)

        if not joins:
            raise ValueError("Join sequence is empty in query result")

        total_rows = 0
        per_join_rows = []
        is_valid_join_row = []

        for join in joins:
            total_rows += join["actual_rows"]
            per_join_rows.append(join["actual_rows"])
            is_valid_join_row.append(True)

        time_string = qlever_result['time_total']
        latency = self.decode_to_seconds(time_string)

        return {
            "per_join_rows": per_join_rows,
            "total_cost": total_rows,
            "latency": latency,
            "is_error": False,
            "is_valid_join_row": is_valid_join_row,
            "time_total": time_string
        }

    def extract_signal(self, qlever_result: dict) -> dict:
        if qlever_result["success"]:
            return self.extract_query_success(qlever_result)
        else:
            parsed_error_result = self._parse_error_result_qlever(qlever_result)
            joins = self._extract_join_sequence_fallback(parsed_error_result)

            if not joins:
                raise ValueError("Join sequence is empty in query result")

            total_rows = 0
            per_join_rows = []
            is_valid_join_row = []

            for join in joins:
                total_rows += join["actual_rows"]
                per_join_rows.append(join["actual_rows"])
                is_valid_join_row.append(join["is_cardinality_valid"])

            latency = parsed_error_result["time"]["total"] / 1000

            return {
                "per_join_rows": per_join_rows,
                "total_cost": total_rows,
                "latency": latency,
                "is_error": True,
                "is_valid_join_row": is_valid_join_row,
                "time_total": qlever_result['time_total']
            }

    @staticmethod
    def _parse_error_result_qlever(qlever_result: dict):
        if "error" not in qlever_result:
            raise ValueError("Unknown qlever result structure")

        try:
            parsed_result = json.loads(qlever_result["error"])
        except json.decoder.JSONDecodeError as e:
            print(qlever_result)
            print()
            raise e

        if "runtimeInformation" not in parsed_result:
            print(parsed_result)
            raise ValueError("Unknown qlever result structure")

        return parsed_result

    @staticmethod
    def _apply_join_order(query_obj, join_order: list[int]) -> str:
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

    @staticmethod
    def print_join_stats(joins: list):
        """Prints the join sequence in a readable table format."""
        print(
            f"\n{'Step':<6} | {'Time (Op/Tot)':<15} | {'Rows (Actual / Est.)':<22} | {'Cost Est. (Op/Tot)':<22} | {'Cache'}")
        print("-" * 95)
        for i, j in enumerate(joins):
            time_str = f"{j['actual_op_time_ms']}ms / {j.get('actual_total_time_ms', 0)}ms"
            row_str = f"{j['actual_rows']} / {j['estimated_rows']}"
            cost_str = f"{j['estimated_op_cost']} / {j.get('estimated_total_cost', 0)}"

            print(f"Join {i + 1:<1} | {time_str:<15} | {row_str:<22} | {cost_str:<22} | {j.get('cache_status', 'N/A')}")

    @staticmethod
    def format_latency(time_s: float) -> str:
        """Converts seconds to a formatted string (e.g., '500ms' or '2s')."""
        if time_s < 1:
            return f"{math.ceil(time_s * 1000)}ms"
        return f"{math.ceil(time_s)}s"

    @staticmethod
    def decode_to_seconds(time_str_raw: str) -> float:
        """Parses a time string into seconds."""
        time_str = str(time_str_raw).strip().lower()

        if "ms" in time_str:
            return float(time_str.split("ms")[0]) / 1000
        elif "s" in time_str:
            return float(time_str.split("s")[0])
        elif "m" in time_str:
            return float(time_str.split("m")[0]) * 60
        else:
            raise ValueError(f"Encountered unknown time_total value: {time_str}")