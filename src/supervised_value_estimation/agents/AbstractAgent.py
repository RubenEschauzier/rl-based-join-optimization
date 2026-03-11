class AbstractCostAgent:
    def setup_episode(self, query):
        raise NotImplementedError

    def estimate_costs(self, possible_next_plans, query_state):
        raise NotImplementedError
