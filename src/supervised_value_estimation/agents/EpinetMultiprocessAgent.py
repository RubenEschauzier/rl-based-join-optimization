import queue

import numpy as np
import torch
from typing import List
from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.typings.typings import EnvState


class EpinetMultiprocessAgent(AbstractCostAgent):
    def __init__(self, model,
                 inference_queue, result_queue, uncertainty_queue, worker_id,
                 precomputed_indexes, precomputed_masks,
                 alpha_mlp, alpha_ensemble,
                 head_name = "plan_cost",
                 annealing_method = "epistemic_uncertainty", ema_alpha=.05):
        self.model = model
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_id = worker_id

        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks

        self.alpha_mlp = alpha_mlp
        self.alpha_ensemble = alpha_ensemble

        self.head_name = head_name

        self.annealing_method = annealing_method
        self.uncertainty_queue = uncertainty_queue
        self.head_uncertainties = {}
        self.ema_alpha = ema_alpha
        self.blending_weight = 0

    def setup_episode(self, query):
        """
        Starts a search episode. Samples a single z and embeds the query based on this z
        :param query:
        :return:
        """
        self.model.eval()
        z = torch.randn((1, self.model.epi_index_dim), device=torch.device('cpu'))

        embedded, embedded_prior = self.model.embed_query_batched(query)[0], self.model.embed_query_batched_prior(query)

        # TODO: Validate this
        # At start of new episode, check if the agent has new uncertainty estimates to update annealing parameters
        try:
            while True:
                estimate_variances = self.uncertainty_queue.get_nowait()
                self.update_uncertainty(estimate_variances)
                self.get_annealing_coefficients()
        except queue.Empty:
            pass

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

        # Get GPU cost estimates
        gpu_results = self.result_queue.get()
        gpu_results = self.inference_result_to_torch(gpu_results)

        # Anneal the latency and cost estimates according to their respective uncertainties.
        # As cost estimation is through a pretrained epinet, this will slowly anneal from cost to latency once
        # the latency epinet gets trained
        epinet_latency = self.estimate_cost_head(gpu_results, query_state, unweighted_priors, "latency")
        epinet_cost = self.estimate_cost_head(gpu_results, query_state, unweighted_priors, "plan_cost")

        combined_value_estimate = (self.blending_weight * epinet_latency.view(-1)
                                   + (1 - self.blending_weight) * epinet_cost.view(-1))

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

        if self.blending_weight != 0:
            #TODO: Validate this blending weight to be a 'normal' value
            pass
        return combined_value_estimate.tolist(), environment_state

    def estimate_cost_head(self, results, query_state, unweighted_priors, head_name):
        est_cost = results["est_cost"][head_name]
        mlp_prior = results["mlp_prior"][head_name]
        learnable_mlp = results["learnable_mlp"][head_name]
        ensemble_prior = torch.matmul(query_state["z"], unweighted_priors[head_name]).view(-1, 1)

        # Combine for total output
        return est_cost + learnable_mlp + (self.alpha_mlp * mlp_prior) + (
                    self.alpha_ensemble * ensemble_prior)

    def update_uncertainty(self, estimate_variances):
        for key, variance in estimate_variances.items():
            if key not in self.head_uncertainties:
                self.head_uncertainties[key] = { "average": variance, "min": variance, "max": variance}
            else:
                current_uncertainty = self.head_uncertainties[key]["average"]
                new_uncertainty = (1-self.ema_alpha) * current_uncertainty + self.ema_alpha * variance
                self.head_uncertainties[key]["average"] = new_uncertainty

                if self.head_uncertainties[key]["min"] > new_uncertainty:
                    self.head_uncertainties[key]["min"] = new_uncertainty

                if self.head_uncertainties[key]["max"] < new_uncertainty:
                    self.head_uncertainties[key]["max"] = new_uncertainty


    def get_annealing_coefficients(self):
        if self.annealing_method == "epistemic_uncertainty":
            #TODO: Validate the shape of this function and its values in practice
            self.blending_weight = np.exp(-1 * self.head_uncertainties["latency"]["average"])
            pass
        else:
            raise NotImplementedError("Annealing Method Not Implemented")
        pass

    @staticmethod
    def inference_result_to_torch(result):
        for key, heads in result.items():
            for head, output in heads.items():
                result[key][head] = torch.tensor(output)
        return result
