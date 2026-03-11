from typing import List


class PlanBestPerformanceCache:
    def __init__(self):
        self.plan_performance_cache = {}

    def add_execution(self, plan: List[int], query: str, latency: float, total_rows: int):
        plan_identifier = self.create_plan_identifier(plan, query)
        self.add_execution_raw(plan_identifier, latency, total_rows)

    def add_execution_raw(self, plan_identifier, latency: float, total_rows: int):
        """
        Function that registers an execution, if it is not the best execution in some way it does nothing
        :param plan_identifier:
        :param latency:
        :param total_rows:
        :return:
        """
        if plan_identifier in self.plan_performance_cache:
            performance = self.plan_performance_cache[plan_identifier]
            # Our target is latency, so best plan is determined by this
            if performance["latency"] > latency:
                performance["latency"] = latency
                performance["total_rows"] = total_rows
        else:
            self.plan_performance_cache[plan_identifier] = { "latency": latency, "total_rows": total_rows }

    def get_target(self, plan: List[int], query: str) -> dict:
        plan_identifier = self.create_plan_identifier(plan, query)
        return self.get_target_raw(plan_identifier)

    def get_target_raw(self, plan_identifier) -> dict:
        # The cache content must always be set as you can only try to construct latencies from plans
        # that have been executed
        if plan_identifier not in self.plan_performance_cache:
            raise KeyError(f"No plan: {plan_identifier}")
        return self.plan_performance_cache[plan_identifier]

    @staticmethod
    def create_plan_identifier(plan: List[int], query: str) -> tuple[tuple[int, ...], str]:
        return tuple(plan), query
