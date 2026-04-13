import aiohttp
import ray

from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient


@ray.remote
class MultiEndpointWorker:
    """One Ray actor managing multiple QLever endpoints via async HTTP."""

    def __init__(self, endpoints: list[str]):
        self.clients = {url: QLeverOptimizerClient(url) for url in endpoints}
        self._inflight = {url: 0 for url in endpoints}
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10 * len(self.clients))
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    def _pick_endpoint(self) -> str:
        return min(self._inflight, key=self._inflight.get)

    async def execute_plan(self, query_obj, join_order=None, timeout="60s", parse_local=True) -> dict:
        url = self._pick_endpoint()
        self._inflight[url] += 1

        try:
            return await self.clients[url].execute_plan(
                query_obj=query_obj,
                join_order=join_order,
                timeout=timeout,
                parse_local=parse_local
            )
        finally:
            self._inflight[url] -= 1

    async def teardown(self):
        """Gracefully closes all aiohttp sessions when the worker is destroyed."""
        for client in self.clients.values():
            await client.close()