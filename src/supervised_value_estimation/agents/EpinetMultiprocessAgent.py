import torch
from typing import List
from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.typings.typings import EnvState


class EpinetMultiprocessAgent(AbstractCostAgent):
    def __init__(self, model,
                 inference_queue, result_queue, worker_id,
                 disk_cache,
                 precomputed_indexes, precomputed_masks,
                 alpha_mlp, alpha_ensemble,
                 head_name = "plan_cost"):
        self.model = model
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.disk_cache = disk_cache
        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks
        self.alpha_mlp = alpha_mlp
        self.alpha_ensemble = alpha_ensemble
        self.head_name = head_name

    def setup_episode(self, query):
        """
        Starts a search episode. Samples a single z and embeds the query based on this z
        :param query:
        :return:
        """
        self.model.eval()
        z = torch.randn((1, self.model.epi_index_dim), device=torch.device('cpu'))

        # TODO: This + any other cache on disk should use a one write many read architecture
        # if query.query[0] not in self.disk_cache:
        embedded, embedded_prior = self.model.embed_query_batched(query)[0], self.model.embed_query_batched_prior(query)
            # self.disk_cache.store_embeddings(query.query[0], (embedded, embedded_prior))
        # else:
        #     embedded, embedded_prior = self.disk_cache.get_embeddings(query.query[0], torch.device('cpu'))

        return {"embedded": embedded, "embedded_prior": embedded_prior, "z": z, "query_idx": 0}

    def estimate_costs(self, possible_next, query_state):
        # Format the plans
        formatted_plans = [(p,) for p in possible_next]

        # Prepare query plan structures on CPU
        prep_trees, prep_idx, prep_masks = self.model.prepare_cost_estimation_inputs(
            formatted_plans, query_state["embedded"], self.precomputed_indexes,
            self.precomputed_masks, target_device=torch.device('cpu')
        )

        # Delegate forward pass to GPU server
        payload = {
            "prep_trees": prep_trees, "prep_idx": prep_idx,
            "prep_masks": prep_masks, "z": query_state["z"]
        }
        self.inference_queue.put((self.worker_id, payload))

        # Compute (tiny) ensemble models on CPU while we wait for GPU
        prior_trees, prior_idx, prior_masks = self.model.prepare_ensemble_prior_inputs(
            formatted_plans, query_state["embedded_prior"], self.precomputed_indexes,
            self.precomputed_masks, query_idx=query_state["query_idx"]
        )

        unweighted_priors = self.model.compute_ensemble_prior_from_prepared(
            prior_trees, prior_idx, prior_masks
        )

        ensemble_prior = torch.matmul(query_state["z"], unweighted_priors[self.head_name]).view(-1, 1)

        # Get GPU cost estimates
        gpu_results = self.result_queue.get()
        est_cost = gpu_results["est_cost"][self.head_name]
        mlp_prior = gpu_results["mlp_prior"][self.head_name]
        learnable_mlp = gpu_results["learnable_mlp"][self.head_name]

        # Combine for total output
        epinet_cost = est_cost + learnable_mlp + (self.alpha_mlp * mlp_prior) + (self.alpha_ensemble * ensemble_prior)

        # Combine relevant (frozen) environment states and return. This will be stored in the buffer
        environment_state: List[EnvState] = []
        for i, plan in enumerate(formatted_plans):
            environment_state.append({
                "unweighted_ensemble_prior": { key: output.cpu().numpy()[:,i] for key, output in unweighted_priors.items() },
                "prepared_trees": prep_trees.cpu().numpy()[i],
                "prepared_idx": prep_idx.cpu().numpy()[i],
                "prepared_masks": prep_masks.cpu().numpy()[i],
                "z": query_state["z"].cpu().numpy()
            })

        return epinet_cost.view(-1).tolist(), environment_state
