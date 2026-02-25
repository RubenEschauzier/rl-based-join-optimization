import asyncio
import aiohttp
import json
import uuid

from src.datastructures.query import Query


class QLeverOptimizerClient:
    def __init__(self, http_endpoint: str, ws_endpoint: str):
        """
        Initializes the client.
        :param http_endpoint: The HTTP URL (e.g., "http://localhost:7001")
        :param ws_endpoint: The WebSocket URL (e.g., "ws://localhost:7001")
        """
        self.http_endpoint = http_endpoint
        self.ws_endpoint = ws_endpoint

    def _apply_join_order(self, query, join_order: list[int]) -> str:
        """
        Injects the join order hint into the SPARQL query.
        """
        if not join_order:
            return query.query
        query_start = "SELECT * WHERE {\n"
        triple_patterns = query["triple_patterns"]
        query = "{ " + f"{triple_patterns[join_order[0]]}  \n {triple_patterns[join_order[1]]} \n" +" }"
        for i in range(2, len(join_order)):
            query = "{ \n" + query + f"\n {triple_patterns[join_order[1]]}" + " }"


        final_query = query_start + query + "\n}"
        # Note: Update this syntax based on how your specific QLever build
        # or optimizer fork parses hints (e.g., using a specific PREFIX or comment).
        return final_query

    async def _track_progress(self, query_id: str, timeout: int) -> list:
        """
        Connects to QLever's WebSocket to receive live execution statistics.
        """
        stats = []
        try:
            async with aiohttp.ClientSession() as session:
                # QLever typically tracks progress by matching the query ID
                ws_url = f"{self.ws_endpoint}/?query_id={query_id}"

                async with session.ws_connect(ws_url, timeout=timeout) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            stats.append(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except asyncio.TimeoutError:
            # Expected behavior if the query exceeds the timeout limit
            pass
        except Exception as e:
            print(f"WebSocket tracking error: {e}")

        return stats

    async def execute_plan(self, query, join_order: list[int] = None, timeout: int = 60) -> dict:
        """
        Executes the query and tracks progress. Returns any available cost estimates
        even if the query times out.
        """
        formatted_query = self._apply_join_order(query, join_order)
        query_id = f"query_{uuid.uuid4().hex}"

        params = {
            "query": formatted_query,
            "timeout": timeout,
            "query_id": query_id
        }

        # 1. Start listening to the WebSocket in the background
        listener_task = asyncio.create_task(self._track_progress(query_id, timeout))

        query_result = None
        error_msg = None

        # 2. Fire the main HTTP query
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.http_endpoint, data=params, timeout=timeout) as response:
                    if response.status == 200:
                        query_result = await response.json()
                    else:
                        error_msg = await response.text()

        except asyncio.TimeoutError:
            error_msg = f"Query timed out after {timeout} seconds."
        except Exception as e:
            error_msg = str(e)

        # 3. Ensure the WebSocket listener closes when the query is done or times out
        if not listener_task.done():
            listener_task.cancel()

        # Gather whatever stats were safely collected before the timeout
        live_stats = await asyncio.gather(listener_task, return_exceptions=True)
        stats_result = live_stats[0] if not isinstance(live_stats[0], Exception) else []

        return {
            "query_completed": query_result is not None,
            "final_result": query_result,
            "error": error_msg,
            "live_stats": stats_result  # Your cost estimates live here!
        }

if __name__ == "__main__":
    async def main():
        client = QLeverOptimizerClient("http://localhost:7001", "ws://localhost:7001")
        query_object = {"triple_patterns": ['?s <http://example.com/13000080> <http://example.com/12610595> .',
                                            '?s <http://example.com/13000080> <http://example.com/12772511> .',
                                            '?s <http://example.com/13000179> <http://example.com/12920312> .',
                                            '?s <http://example.com/13000179> <http://example.com/12939215> .',
                                            '?s <http://example.com/13000083> ?o4 .'],
                        "query": 'SELECT * WHERE {  ?s <http://example.com/13000080> <http://example.com/12610595> .  ?s <http://example.com/13000080> <http://example.com/12772511> .  ?s <http://example.com/13000179> <http://example.com/12939215> .  ?s <http://example.com/13000179> <http://example.com/12920312> .  ?s <http://example.com/13000083> ?o4 . }'}

        result = await client.execute_plan(
            query=query_object,
            join_order=[4,3,2,1,0],
            timeout=10
        )
        print(f"Collected {len(result['live_stats'])} statistic updates.")

    asyncio.run(main())