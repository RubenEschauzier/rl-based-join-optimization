from typing import List, Optional


class PlanBestPerformanceCache:
    def __init__(self):
        self.plan_performance_cache = {}

    def add_execution(self, plan: List[int], query: str, latency: float, total_rows: int, is_censored: bool = False):
        plan_identifier = self.create_plan_identifier(plan, query)
        self.add_execution_raw(plan_identifier, latency, total_rows, is_censored)

    def add_execution_raw(self, plan_identifier, latency: float, total_rows: int, is_censored: bool = False):
        """
        Registers an execution. Retains the execution with the lowest latency.
        """
        if plan_identifier in self.plan_performance_cache:
            performance = self.plan_performance_cache[plan_identifier]

            # Update if the new execution is strictly faster
            if performance["latency"] > latency:
                performance["latency"] = latency
                performance["total_cost"] = total_rows
                performance["is_censored"] = is_censored

        else:
            self.plan_performance_cache[plan_identifier] = {
                "latency": latency,
                "total_cost": total_rows,
                "is_censored": is_censored
            }

    def get_target(self, plan: List[int], query: str) -> dict:
        plan_identifier = self.create_plan_identifier(plan, query)
        return self.get_target_raw(plan_identifier)

    def get_target_raw(self, plan_identifier) -> dict:
        if plan_identifier not in self.plan_performance_cache:
            raise KeyError(f"No plan: {plan_identifier}")
        return self.plan_performance_cache[plan_identifier]

    @staticmethod
    def create_plan_identifier(plan: List[int], query: str) -> tuple[tuple[int, ...], str]:
        return tuple(plan), query