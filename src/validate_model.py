import torch
import numpy as np

from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN
from src.rl_fine_tuning_qr_dqn_learning import make_env_execution_cost


def validate_qr_dqn_model(emb_model, query_env, occurrences, dataset,
                          model_ckp, predict_fn):
    model = MaskableQRDQN.load(model_ckp)
    validation_env = make_env_execution_cost(emb_model, dataset,
                                             False, query_env,
                                             occurrences=occurrences)
    n_queries = len(dataset)
    for _ in range(n_queries):
        query_reward = 0
        query_done = False

        obs, _ = validation_env.reset()
        while not query_done:
            mask = validation_env.action_masks()
            action, _ = predict_fn(model, obs, mask)
            obs, reward, terminated, truncated, _ = validation_env.step(action)
            query_reward += reward
            if terminated:
                obs, _ = validation_env.reset()
                query_done = True

    pass

def validate_ppo_model(dataset, model_ckp, predict_fn):
    pass

def validate_cardinality_estimation_methods():
    pass

def load_model(ckp_location):
    pass


def qr_dqn_variance_penalty(model, obs, mask, lambda_penalty):
    with torch.no_grad():
        # Get quantiles (batch_size, n_actions, n_quantiles)
        quantiles = model.policy.quantile_net(torch.as_tensor(obs, device=model.device).float(),
                                              deterministic=True,
                                              action_masks=mask)
        # Expected Q-values = mean over quantiles
        q_values = quantiles.mean(dim=2).cpu().numpy()[0]
        # Variance over quantiles
        variances = quantiles.var(dim=2).cpu().numpy()[0]

    penalized_q_values = q_values - lambda_penalty * variances
    best_action = int(np.argmax(penalized_q_values))
    return best_action, penalized_q_values