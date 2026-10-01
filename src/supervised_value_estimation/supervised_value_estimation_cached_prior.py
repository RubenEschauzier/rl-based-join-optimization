import os
import sys
from time import perf_counter, sleep

import diskcache
import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from src.datastructures.query_cardinality_dataset import QueryCardinalityDataset
from src.models.epistemic_neural_network import MultiHeadEpistemicNetwork, prepare_epinet_model
from src.pretrain_procedure import DualMetricScheduler
from src.utils.epinet_utils.epinet_evaluation import SplitAccumulator, metric_names, query_statistics
from src.utils.epinet_utils.epinet_report import RunReporter, write_json
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.early_stopping import ConvergenceEarlyStopping
from src.utils.training_utils.training_tracking import TrainSummary, ExperimentWriter

# Get the path of the parent directory (the root of the project)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.models.model_instantiator import ModelFactory
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset, prepare_data
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask
import torch


def prepare_cardinality_estimator(model_config, model_directory=None):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    if model_directory:
        load_weights_from_pretraining(gine_conv_model, model_directory,
                                      "embedding_model.pt",
                                      ["head_cardinality.pt"],
                                      float_weights=True)
    return gine_conv_model


def print_param_count(epinet_cost_estimation, train_epi_network):
    params_cost_estimate = list(epinet_cost_estimation.cost_estimation_model.parameters())

    total_params_cost_estimation = 0
    for param in params_cost_estimate:
        total_params_cost_estimation += param.numel()
    print(f"Cost estimation model has {total_params_cost_estimation} parameters")

    if train_epi_network:
        total_params = 0
        for param in epinet_cost_estimation.parameters():
            total_params += param.numel()
        print(f"Epinet model has {total_params - total_params_cost_estimation} parameters")


def fetch_or_compute_priors(query_batch, valid_indices, query_plans, cache, epinet, precomputed_indexes,
                            precomputed_masks, device, zero_ensemble_prior=False):
    """Retrieves priors for all heads from cache or computes and stores them if missing.

    With `zero_ensemble_prior` (alpha_ensemble == 0) the GNN-ensemble prior is multiplied
    by zero everywhere it is used, so it is returned as zeros instead of being computed:
    a pass over epi_index_dim GNNs per query, and the dominant cost of a run.
    """
    if zero_ensemble_prior:
        return [
            {head: torch.zeros((epinet.epi_index_dim, len(query_plans[query_batch.query[i]])), device=device)
             for head in epinet.head_names}
            for i in valid_indices
        ]

    priors = {}
    missing_indices = []

    for i in valid_indices:
        q_id = query_batch.query[i]
        cached_priors = cache.get(q_id)
        if cached_priors is not None:
            # Retrieve all cached heads and move to target device
            priors[i] = {head: tensor.to(device) for head, tensor in cached_priors.items()}
        else:
            missing_indices.append(i)

    if missing_indices:
        with torch.no_grad():
            embedded_prior = epinet.embed_query_batched_prior(query_batch.to(device))
            for i in missing_indices:
                q_id = query_batch.query[i]
                plans_current_query = query_plans[q_id]

                unweighted_ensemble_prior = epinet.compute_ensemble_prior(
                    plans_current_query, embedded_prior, precomputed_indexes, precomputed_masks, i
                )

                # Separate CPU tensors for caching and GPU tensors for immediate use
                cpu_priors = {head: tensor.cpu() for head, tensor in unweighted_ensemble_prior.items()}
                device_priors = {head: tensor.to(device) for head, tensor in unweighted_ensemble_prior.items()}

                cache[q_id] = cpu_priors
                priors[i] = device_priors

    return [priors[i] for i in valid_indices]

def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_cached(val_loader, query_plans_val, epinet_cost_estimation, val_cache,
                    mean_vals, std_vals, train_loss, device, n_val_epi_indexes,
                    sigma, alpha_mlp, alpha_ensemble, precomputed_indexes, precomputed_masks,
                    evaluation_noise_std, joint_taus, evaluation_seed,
                    split="val", zero_ensemble_prior=False, head_name="plan_cost"):
    """One pass over a split, returning a SplitAccumulator of per-query statistics.

    Metrics that depend on the fitted base-noise baseline are produced by
    `accumulator.summarize(base_noise_std, epinet_noise_std)` afterwards, so validation can fit the noise
    and test can reuse it without a second pass.
    """
    accumulator = SplitAccumulator(split, joint_taus, evaluation_noise_std, std_vals[head_name])
    generator = torch.Generator(device=device)
    pbar = tqdm(total=len(val_loader.dataset), desc=f"Evaluating [{split}]", leave=False, position=1,
                dynamic_ncols=True, mininterval=300)
    start = perf_counter()

    for query_batch in val_loader:
        valid_indices = [i for i, q_id in enumerate(query_batch.query)
                         if q_id in query_plans_val and len(query_plans_val[q_id]) > 0]
        if not valid_indices:
            pbar.update(len(query_batch.query))
            continue

        unweighted_ensemble_priors = fetch_or_compute_priors(
            query_batch, valid_indices, query_plans_val, val_cache, epinet_cost_estimation, precomputed_indexes,
            precomputed_masks, device, zero_ensemble_prior=zero_ensemble_prior,
        )

        with torch.no_grad():
            _synchronize(device)
            base_start = perf_counter()
            embedded = epinet_cost_estimation.embed_query_batched(query_batch.to(device))
            _synchronize(device)
            accumulator.seconds_base += perf_counter() - base_start

            for valid_idx, unweighted_ensemble_prior in zip(valid_indices, unweighted_ensemble_priors):
                plans_query = query_plans_val[query_batch.query[valid_idx]]
                query_seed = evaluation_seed + plans_query[0][2]

                base_start = perf_counter()
                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
                )
                _synchronize(device)
                epinet_start = perf_counter()
                accumulator.seconds_base += epinet_start - base_start

                estimated_head_val = estimated_cost[head_name]
                val_loss_epinet, repeated_target, epinet_cost_estimates = loss_epinet(
                    unweighted_ensemble_prior[head_name],
                    epinet_cost_estimation,
                    train_loss,
                    estimated_head_val,
                    last_feature,
                    plans_query,
                    n_val_epi_indexes,
                    sigma, alpha_mlp, alpha_ensemble,
                    generator,
                    device,
                    head_name=head_name,
                    epistemic_seed=query_seed,
                )
                _synchronize(device)
                accumulator.seconds_epinet += perf_counter() - epinet_start

                n_plans = estimated_head_val.shape[0]
                accumulator.add(
                    query_statistics(
                        epinet_cost_estimates.reshape(n_val_epi_indexes, n_plans),
                        estimated_head_val,
                        repeated_target[:n_plans],
                        evaluation_noise_std,
                        joint_taus,
                        query_seed,
                        std_vals[head_name],
                    ),
                    loss_epinet=val_loss_epinet.item(),
                )
        pbar.update(len(query_batch.query))

    pbar.close()
    accumulator.eval_seconds = perf_counter() - start
    return accumulator


def train_on_batch_cached(query_batch, valid_indices, query_plans, cache,
                          precomputed_indexes, precomputed_masks,
                          epinet_cost_estimation,
                          optimizer, loss,
                          n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble,
                          generator, device, l2_lambda=0.0, anchor_seed=0, zero_ensemble_prior=False):
    acc_loss = torch.tensor(0.0, device=device)

    unweighted_ensemble_priors = fetch_or_compute_priors(
        query_batch, valid_indices, query_plans, cache, epinet_cost_estimation, precomputed_indexes, precomputed_masks,
        device, zero_ensemble_prior=zero_ensemble_prior,
    )

    embedded = epinet_cost_estimation.embed_query_batched(query_batch.to(device))

    for valid_idx, unweighted_ensemble_prior in zip(valid_indices, unweighted_ensemble_priors):
        q_id = query_batch.query[valid_idx]
        plans_query = query_plans[q_id]

        estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
            plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
        )
        estimated_cost = estimated_cost["plan_cost"]

        loss_epinet_val, _, _ = loss_epinet(
            unweighted_ensemble_prior["plan_cost"],
            epinet_cost_estimation,
            loss,
            estimated_cost,
            last_feature,
            plans_query,
            n_epi_indexes_train,
            sigma, alpha_mlp, alpha_ensemble,
            generator, device,
            anchor_seed=anchor_seed,
        )
        acc_loss += loss_epinet_val

    acc_loss /= len(valid_indices)

    if l2_lambda > 0.0:
        # Equation 9 regularizes lambda * (||zeta||^2 + ||eta||^2) over both the base and
        # epinet parameters. Kept in the loss (as deepmind/enn's add_l2_weight_decay does)
        # rather than as AdamW's decoupled decay, so the coefficient means what Theorem 4
        # says it means.
        acc_loss = acc_loss + l2_lambda * squared_parameter_norm(epinet_cost_estimation)

    acc_loss.backward()
    optimizer.step()
    return acc_loss.detach().cpu().item()


def squared_parameter_norm(model):
    """||theta||^2 over the trainable parameters, i.e. Equation 9's penalty term."""
    return torch.stack([
        parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad
    ]).sum()


@torch.no_grad()
def measure_prior_scale(loader, query_plans, cache, epinet_cost_estimation,
                        precomputed_indexes, precomputed_masks,
                        alpha_mlp, alpha_ensemble, device,
                        n_batches=4, n_epi_indexes=128, head_name="plan_cost", zero_ensemble_prior=False):
    """Measure sigma_0: the std over the epistemic index of the total prior term.

    Also reports the two prior terms separately. They are scaled independently
    (alpha_mlp, alpha_ensemble) but nothing keeps them balanced, and a term that is an
    order of magnitude smaller than the other contributes almost nothing to the epistemic
    signal no matter how expensive it was to compute.
    """
    totals = {"mlp": [], "ensemble": [], "combined": []}

    for batch_index, query_batch in enumerate(loader):
        if batch_index >= n_batches:
            break
        valid_indices = [i for i, q_id in enumerate(query_batch.query)
                         if q_id in query_plans and len(query_plans[q_id]) > 0]
        if not valid_indices:
            continue

        priors = fetch_or_compute_priors(
            query_batch, valid_indices, query_plans, cache, epinet_cost_estimation,
            precomputed_indexes, precomputed_masks, device, zero_ensemble_prior=zero_ensemble_prior,
        )
        embedded = epinet_cost_estimation.embed_query_batched(query_batch.to(device))
        indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(n_epi_indexes)

        for valid_idx, prior in zip(valid_indices, priors):
            plans_query = query_plans[query_batch.query[valid_idx]]
            _, last_feature = epinet_cost_estimation.estimate_cost_full(
                plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
            )
            n_plans = last_feature.shape[0]

            mlp_term = alpha_mlp * epinet_cost_estimation.compute_mlp_prior_batched(
                last_feature, indexes
            )[head_name].reshape(n_epi_indexes, n_plans)
            ensemble_term = alpha_ensemble * torch.matmul(indexes, prior[head_name])

            totals["mlp"].append(mlp_term.std(dim=0).mean().item())
            totals["ensemble"].append(ensemble_term.std(dim=0).mean().item())
            totals["combined"].append((mlp_term + ensemble_term).std(dim=0).mean().item())

    if not totals["combined"]:
        raise ValueError("Could not measure the prior scale: no valid queries in the loader.")

    means = {key: float(np.mean(values)) for key, values in totals.items()}
    print(f"Prior scale (std over epistemic index) -> "
          f"alpha_mlp * mlp_prior: {means['mlp']:.4f} | "
          f"alpha_ensemble * ensemble_prior: {means['ensemble']:.4f} | "
          f"combined sigma_0: {means['combined']:.4f}")
    return means["combined"]


def theorem_4_lambda(sigma, n_train_points, prior_scale):
    """Equation 9's regularization coefficient, lambda = sigma^2 / (N * sigma_0^2).

    Theorem 4 fixes lambda rather than leaving it free: a *weak* prior (small sigma_0)
    demands a *large* penalty. `prior_scale` is sigma_0, the standard deviation over the
    epistemic index of the total prior term, measured on the training data.

    Note this is the coefficient for a loss that averages over the N data points; the
    paper sums over them and so writes the same constant against a summed loss.
    """
    if sigma <= 0:
        return 0.0
    if n_train_points <= 0 or prior_scale <= 0:
        raise ValueError(
            f"Need positive n_train_points and prior_scale, got {n_train_points} and {prior_scale}"
        )
    return (sigma ** 2) / (n_train_points * prior_scale ** 2)

def loss_epinet(unweighted_ensemble_priors,
                epinet_cost_estimation,
                loss,
                estimated_cost, last_feature,
                plans, n_epi_indexes,
                sigma, alpha_mlp, alpha_ensemble,
                generator,
                device,
                head_name = "plan_cost",
                epistemic_seed=None,
                anchor_seed=0):
    last_feature_detached = last_feature.detach()
    n_plans = estimated_cost.shape[0]

    # The anchor vectors c_i are the per-plan target perturbations; they are fixed per plan
    # so they stay consistent across epochs. `anchor_seed` gives each training seed its own
    # draw, so seed-to-seed variance includes this source of randomness. 0 keeps the
    # original per-plan seeds.
    generator.manual_seed(plans[0][2] + anchor_seed * 2 ** 32)
    # (n_plans, epi_index_dim)
    c_vectors = torch.randn((n_plans, epinet_cost_estimation.epi_index_dim), generator=generator, device=device)
    c_vectors = torch.nn.functional.normalize(c_vectors, dim=-1)

    # (n_epi_indexes, epi_index_dim)
    epistemic_generator = None
    if epistemic_seed is not None:
        epistemic_generator = torch.Generator(device=device)
        epistemic_generator.manual_seed(epistemic_seed)
    epinet_indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(
        n_epi_indexes,
        generator=epistemic_generator,
    )
    # (n_epi_indexes, n_plans)
    ensemble_prior = torch.matmul(epinet_indexes, unweighted_ensemble_priors)
    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Row-major contiguous flattening.
    # Elements are grouped by epistemic index (ei), iterating through all plans for a given index before moving to the next:
    # (plan_1_ei_1, plan_2_ei_1, ..., plan_n_ei_1,  plan_1_ei_2, plan_2_ei_2, ..., plan_n_ei_2,  ..., plan_n_ei_k)
    # where ei = epistemic index sampled during training.
    ensemble_prior_flat = ensemble_prior.view(-1, 1)

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Row-major contiguous block. Grouped by epistemic index: evaluates all plans
    # using the first sampled index, then all plans using the second index, etc.
    mlp_prior = epinet_cost_estimation.compute_mlp_prior_batched(last_feature_detached, epinet_indexes)
    mlp_prior = mlp_prior[head_name]

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Matches the row-major, index-grouped layout of the priors.
    learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp_batched(last_feature_detached, epinet_indexes)
    learnable_mlp_prior = learnable_mlp_prior[head_name]

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: repeat(n_epi_indexes, 1) copies the full block of base plan costs K times.
    # This aligns the base costs identically with the row-major, index-grouped layout of the Epinet.
    estimated_cost_exp = estimated_cost.repeat(n_epi_indexes, 1)
    

    # Shape: (n_epi_indexes * n_plans)
    # Layout: Row-major contiguous flattening.
    # Grouped by epistemic index: calculates the perturbation (z * c) for all plans using
    # the first sampled index, then all plans using the second index, etc.
    
    # TODO: Maybe this c_vectors should be correlated within a query. Not completely but a little
    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    anchor_term_flat = anchor_matrix.view(-1)

    raw_targets = torch.tensor([plan[1] for plan in plans], device=device)

    # Base model loss (purely deterministic, no variance from z)
    loss_base = loss(estimated_cost.squeeze(-1), raw_targets.float())

    # Epinet loss (we detach the base model so it doesn't receive z-variance gradients)
    epinet_estimated_cost_detached = estimated_cost_exp.detach() + (
            learnable_mlp_prior + alpha_mlp * mlp_prior + alpha_ensemble * ensemble_prior_flat
    )

    # (n_epi_indexes * n_plans)
    raw_targets_exp = raw_targets.repeat(n_epi_indexes)

    perturbed_targets = raw_targets_exp + sigma * anchor_term_flat

    unperturbed_target = raw_targets_exp

    
    loss_epinet_only = loss(epinet_estimated_cost_detached.squeeze(-1), perturbed_targets)
    loss_total = loss_base + loss_epinet_only

    return loss_total, unperturbed_target, epinet_estimated_cost_detached


def train_simulated_epinet_cached(queries_train: QueryCardinalityDataset, query_plans_train,
                                  mean_train, std_train,
                                  queries_val, query_plans_val,
                                  model_builder_fn, model_kwargs, model_state_dict,
                                  epinet_cost_estimation: MultiHeadEpistemicNetwork,
                                  device,
                                  query_batch_size, n_epi_indexes_train,
                                  sigma, alpha_mlp, alpha_ensemble, lr, weight_decay, n_epochs,
                                  n_epi_indexes_val,
                                  writer,
                                  cache_directory,
                                  queries_test=None,
                                  query_plans_test=None,
                                  l2_mode="manual",
                                  prior_scale_target=None,
                                  clear_prior_cache=True,
                                  validate_every=1,
                                  plot_calibration=True,
                                  use_tensorboard=False,
                                  evaluation_noise_std=0.2,
                                  joint_taus=(1, 2, 4, 8, 16),
                                  evaluation_seed=0,
                                  anchor_seed=0,
                                  objective_metric="val_jnll_tau8_epinet_excess_per_target",
                                  early_stopping_patience=None,
                                  early_stopping_min_delta=0.0,
                                  early_stopping_min_epochs=0,
                                  debug_single_batch = False,
                                  trial: optuna.Trial = None):
    """Train the epinet and evaluate it on validation (and optionally test) every epoch.

    Model selection -- early stopping, the LR schedule, Optuna pruning and the best epoch
    in final_summary.json -- only ever looks at validation. Test is evaluated alongside
    so its curves can be inspected, but a paper number should be read at the best
    *validation* epoch, which is what final_summary.json reports.

    `plot_calibration` now controls all per-epoch figures (report_<split>.png), and
    `use_tensorboard` additionally logs scalars and figures to <run>/tensorboard.
    """

    import shutil
    # The frozen priors depend only on the prior config and the seed that built them, so a
    # caller that guarantees those are unchanged (a sweep reusing one ensemble across
    # trials) can keep the cache. Rebuilding it is a pass over epi_index_dim GNNs for
    # every query and dominates everything else in a short run.
    if clear_prior_cache:
        shutil.rmtree(cache_directory, ignore_errors=True)
    os.makedirs(cache_directory, exist_ok=True)
    train_cache = diskcache.Cache(os.path.join(cache_directory, "train_cache"), size_limit=50 * 1024 ** 3)
    # Test queries are a subset of the original validation file, so they share its cache.
    val_cache = diskcache.Cache(os.path.join(cache_directory, "val_cache"), size_limit=50 * 1024 ** 3)

    if clear_prior_cache:
        # Prevents stale priors between runs when the ensemble is randomly re-initialized.
        train_cache.clear()
        val_cache.clear()

    joint_taus = tuple(int(tau) for tau in joint_taus)
    # A zero-weighted ensemble prior never reaches any output, so skip computing it.
    zero_ensemble_prior = alpha_ensemble == 0.0

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    evaluate_test = queries_test is not None and query_plans_test is not None and not debug_single_batch
    splits = ("val", "test") if evaluate_test else ("val",)
    train_summary = TrainSummary(metric_names(splits, joint_taus))
    if objective_metric not in train_summary.best_values:
        raise ValueError(f"Unknown objective metric: {objective_metric}")
    if not objective_metric.startswith("val_"):
        raise ValueError(f"The objective must be a validation metric, got {objective_metric}")

    early_stopper = None
    if early_stopping_patience is not None:
        objective_mode = train_summary.best_values[objective_metric]["type"]
        early_stopper = ConvergenceEarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            min_epochs=early_stopping_min_epochs,
            mode=objective_mode,
        )

    # Predefine a generator so the perturbation vectors are consistent among epochs
    generator = torch.Generator(device=device)
    epinet_cost_estimation.to(device)

    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)
    loader_val = DataLoader(queries_val, batch_size=1, shuffle=False)
    loader_test = DataLoader(queries_test, batch_size=1, shuffle=False) if evaluate_test else None

    # Flag to overfit on a single batch to validate that the model works on very small data
    if debug_single_batch:
        print("DEBUG MODE: Fitting on a single training batch subset.")

        # Create single valid batch from training data
        valid_indices = []
        for idx in range(len(queries_train)):
            q_id = queries_train[idx].query
            if q_id in query_plans_train:
                valid_indices.append(idx)
            if len(valid_indices) == query_batch_size:
                break

        if not valid_indices:
            raise ValueError("No valid training queries found to debug.")

        # Take subset with valid data batch
        subset_dataset = queries_train[valid_indices]

        # Overwrite data loaders to only use the one training batch
        loader = DataLoader(subset_dataset, batch_size=query_batch_size, shuffle=False)
        loader_val = DataLoader(subset_dataset, batch_size=1, shuffle=False)
        query_plans_val = query_plans_train

    # TODO: Temp removal of freezing of cost estimation model parameters for debugging
    # Freeze cost estimation model parameters for epinet training
    # for param in epinet_cost_estimation.cost_estimation_model.parameters():
    #     param.requires_grad = False
    # epinet_cost_estimation.cost_estimation_model.eval()

    params_all = list(epinet_cost_estimation.parameters())
    params_trainable = [p for p in epinet_cost_estimation.parameters() if p.requires_grad]

    num_trainable = sum(p.numel() for p in params_trainable)
    num_fixed = sum(p.numel() for p in params_all if not p.requires_grad)

    print( f"Parameter counts -> Trainable: {num_trainable:,} | Fixed: {num_fixed:,} "
           f"| Total: {num_trainable + num_fixed:,}")

    prior_scale = measure_prior_scale(
        loader, query_plans_train, train_cache, epinet_cost_estimation,
        precomputed_indexes, precomputed_masks, alpha_mlp, alpha_ensemble, device,
        zero_ensemble_prior=zero_ensemble_prior,
    )

    if prior_scale_target:
        # Targets are standardized, so a randomized prior of unit scale says "before
        # seeing data this plan's cost could plausibly sit anywhere within ~1 std of the
        # mean". Left uncalibrated the prior scale is an artifact of the feature width and
        # Glorot fan-in, not a modelling choice -- at feature width 264 it comes out
        # several times the target range and swamps the signal.
        rescale = prior_scale_target / prior_scale
        alpha_mlp *= rescale
        alpha_ensemble *= rescale
        prior_scale = prior_scale_target
        print(f"Prior auto-calibration -> target sigma_0={prior_scale_target:g}, "
              f"scaling alphas by {rescale:.4f} => alpha_mlp={alpha_mlp:.4f}, "
              f"alpha_ensemble={alpha_ensemble:.4f} (their ratio is preserved; adjust the "
              f"configured ratio if one prior term dominates the other above)")

    l2_lambda = 0.0
    optimizer_weight_decay = weight_decay
    if l2_mode == "theorem4":
        n_train_points = sum(len(plans) for plans in query_plans_train.values())
        l2_lambda = theorem_4_lambda(sigma, n_train_points, prior_scale)
        # Equation 9's penalty now lives in the loss, so decoupled decay would double count.
        optimizer_weight_decay = 0.0
        print(f"Theorem 4 L2 -> sigma={sigma:g} N={n_train_points:,} "
              f"sigma_0={prior_scale:.4f} => lambda={l2_lambda:.3e} "
              f"(config weight_decay={weight_decay:g} ignored)")

    optimizer = torch.optim.AdamW(params_trainable, lr=lr, weight_decay=optimizer_weight_decay)
    scheduler = DualMetricScheduler(optimizer,
                                    patience=5,
                                    threshold=1e-3,
                                    threshold_mode='rel',
                                    factor=0.3,
                                    min_lr=lr * 1e-3,
                                    )
    previous_lr = scheduler.get_last_lr()

    print_param_count(epinet_cost_estimation, True)
    sleep(1)

    loss = torch.nn.MSELoss(reduction='mean')
    reporter = RunReporter(writer.experiment_directory, use_tensorboard=use_tensorboard,
                           write_figures=plot_calibration)
    metrics_per_epoch = {}

    # noinspection PyTypeChecker
    total_train_queries = len(loader.dataset)
    pbar = tqdm(total=total_train_queries, position=0, leave=True, dynamic_ncols=True, mininterval=300)

    def evaluate(split_loader, split_plans, split_name):
        return validate_cached(
            split_loader, split_plans, epinet_cost_estimation, val_cache, mean_train, std_train,
            loss, device, n_epi_indexes_val, sigma, alpha_mlp, alpha_ensemble,
            precomputed_indexes, precomputed_masks,
            evaluation_noise_std, joint_taus, evaluation_seed,
            split=split_name, zero_ensemble_prior=zero_ensemble_prior,
        )

    for epoch in range(1, n_epochs + 1):
        batch_losses = []
        n_empty_plans = 0
        n_non_empty_plans = 0
        processed = 0
        epoch_start = perf_counter()

        pbar.reset()
        pbar.set_description(f"Epoch {epoch}/{n_epochs} [Train]")

        for query_batch in loader:
            optimizer.zero_grad()

            valid_indices = [i for i, q_id in enumerate(query_batch.query)
                             if q_id in query_plans_train and len(query_plans_train[q_id]) > 0]

            n_empty_plans += (len(query_batch.query) - len(valid_indices))
            n_non_empty_plans += len(valid_indices)
            processed += len(query_batch.query)

            pbar.set_postfix(
                empty_plans=n_empty_plans,
                non_empty_plans=n_non_empty_plans,
                queries=processed
            )

            if not valid_indices:
                pbar.update(len(query_batch.query))
                continue

            batch_loss = train_on_batch_cached(
                query_batch, valid_indices, query_plans_train, train_cache,
                precomputed_indexes, precomputed_masks,
                epinet_cost_estimation, optimizer, loss,
                n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble, generator, device,
                l2_lambda=l2_lambda, anchor_seed=anchor_seed, zero_ensemble_prior=zero_ensemble_prior,
            )

            batch_losses.append(batch_loss)
            pbar.update(len(query_batch.query))

        epoch_train_loss = np.mean(batch_losses)
        epoch_train_seconds = perf_counter() - epoch_start

        # Evaluation is expensive: batch_size=1 over every query, n_epi_indexes_val index
        # samples each. Worth it every epoch for a final run, wasteful inside a sweep.
        is_last_epoch = epoch == n_epochs
        if validate_every > 1 and epoch % validate_every != 0 and not is_last_epoch:
            continue

        val_accumulator = evaluate(loader_val, query_plans_val, "val")
        # Both fitted noise stds (base network and epinet) are fitted on validation only
        # and reused for test, so no fitted model ever sees the test residuals.
        base_noise_std = val_accumulator.fitted_base_noise_std()
        epinet_noise_std = val_accumulator.fitted_epinet_noise_std()
        epoch_metrics, val_curves = val_accumulator.summarize(base_noise_std, epinet_noise_std)
        curves_per_split = {"val": val_curves}
        del val_accumulator
        if evaluate_test:
            test_metrics, curves_per_split["test"] = evaluate(
                loader_test, query_plans_test, "test"
            ).summarize(base_noise_std, epinet_noise_std)
            epoch_metrics.update(test_metrics)

        epoch_metrics.update({
            "train_loss": float(epoch_train_loss),
            "train_epoch_seconds": epoch_train_seconds,
        })
        metrics_per_epoch[epoch] = epoch_metrics
        train_summary.update(epoch_metrics, epoch)

        scheduler.step(train_loss=epoch_metrics['train_loss'],
                       val_metric=epoch_metrics['val_loss_plan_cost_unscaled'])

        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file([], best, per_epoch, epinet_cost_estimation, epoch)
        reporter.log_epoch(epoch, writer.get_epoch_dir(epoch), epoch_metrics, curves_per_split)

        objective_value = epoch_metrics[objective_metric]
        print(f"Epoch {epoch}: {objective_metric}={objective_value:.5f} | "
              f"tau8 dependence gain val={epoch_metrics.get('val_jnll_tau8_dependence_gain', float('nan')):.4f}"
              + (f" test={epoch_metrics.get('test_jnll_tau8_dependence_gain', float('nan')):.4f}"
                 if evaluate_test else ""))

        if scheduler.get_last_lr() != previous_lr:
            print(f"INFO: Lr Updated from {previous_lr} to {scheduler.get_last_lr()}")
            previous_lr = scheduler.get_last_lr()

        if trial:
            trial.report(objective_value, epoch)
            if trial.should_prune():
                reporter.close()
                train_cache.close()
                val_cache.close()
                raise optuna.TrialPruned()

        if early_stopper and early_stopper.step(objective_value, epoch):
            print(
                f"EARLY STOP: {objective_metric} did not improve by "
                f"{early_stopping_min_delta:g} for {early_stopping_patience} epochs. "
                f"Best value {early_stopper.best_value:.6f} at epoch {early_stopper.best_epoch}."
            )
            break

    pbar.close()
    reporter.close()
    train_cache.close()
    val_cache.close()

    best_epoch = train_summary.best_values[objective_metric]["epoch"]
    if metrics_per_epoch:
        # Written last, so its presence marks a finished run (used to skip done seeds).
        write_json(os.path.join(writer.experiment_directory, "final_summary.json"), {
            "objective_metric": objective_metric,
            "best_epoch": best_epoch,
            "last_epoch": max(metrics_per_epoch),
            "metrics_at_best_epoch": metrics_per_epoch.get(best_epoch, {}),
            "metrics_at_last_epoch": metrics_per_epoch[max(metrics_per_epoch)],
        })
    return train_summary.best_values[objective_metric]


def main_simulated_epinet_training(cfg: DictConfig,
                                   train_dataset,
                                   val_dataset,
                                   oracle_model,
                                   epinet_cost_estimation,
                                   model_kwargs,
                                   model_builder_fn,
                                   device,
                                   writer):

    debug_single_batch = OmegaConf.select(cfg, "debug.debug_single_batch", default=False)
    evaluation_noise_std = OmegaConf.select(cfg, "evaluation.noise_std", default=0.2)
    joint_taus = OmegaConf.select(cfg, "evaluation.joint_taus", default=[1, 2, 4, 8, 16])
    evaluation_seed = OmegaConf.select(cfg, "evaluation.seed", default=0)
    objective_metric = OmegaConf.select(
        cfg,
        "training.objective_metric",
        default="val_jnll_tau8_epinet_excess_per_target",
    )
    early_stopping_patience = OmegaConf.select(cfg, "training.early_stopping_patience", default=None)
    early_stopping_min_delta = OmegaConf.select(cfg, "training.early_stopping_min_delta", default=0.0)
    early_stopping_min_epochs = OmegaConf.select(cfg, "training.early_stopping_min_epochs", default=0)
    # "theorem4" derives lambda = sigma^2 / (N * sigma_0^2) from the measured prior scale
    # and puts it in the loss; "manual" keeps hyperparameters.weight_decay in AdamW.
    l2_mode = OmegaConf.select(cfg, "hyperparameters.l2_mode", default="manual")
    # Rescales alpha_mlp/alpha_ensemble so the measured prior std hits this value.
    prior_scale_target = OmegaConf.select(cfg, "hyperparameters.prior_scale_target", default=None)
    writer.create_experiment_directory()

    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated,
                                     debug_single_batch=debug_single_batch,
                                     query_batch_size=cfg.hyperparameters.query_batch_size)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val,
                                         debug_single_batch=debug_single_batch,
                                         query_batch_size=cfg.hyperparameters.query_batch_size)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train_cost, std_train_cost = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(
        query_plans_dict_val,
        mean_train_cost,
        std_train_cost,
    )

    mean_train = {"plan_cost": mean_train_cost}
    std_train = {"plan_cost": std_train_cost}

    model_state_dict = epinet_cost_estimation.state_dict()

    # Configure explicit cache directory from configuration or default to a fixed path
    cache_directory = getattr(cfg.dataset, 'prior_cache_dir', os.path.join(os.getcwd(), ".prior_cache"))

    return train_simulated_epinet_cached(
        queries_train=train_dataset,
        query_plans_train=train_plans,
        mean_train=mean_train,
        std_train=std_train,
        queries_val=val_dataset,
        query_plans_val=val_plans,
        model_builder_fn=model_builder_fn,
        model_kwargs=model_kwargs,
        model_state_dict=model_state_dict,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer,
        cache_directory=cache_directory,
        query_batch_size=cfg.hyperparameters.query_batch_size,
        n_epi_indexes_train=cfg.hyperparameters.n_epi_indexes_train,
        n_epi_indexes_val=cfg.hyperparameters.n_epi_indexes_val,
        sigma=cfg.hyperparameters.sigma,
        alpha_mlp=cfg.hyperparameters.alpha_mlp,
        alpha_ensemble=cfg.hyperparameters.alpha_ensemble,
        lr=cfg.hyperparameters.lr,
        weight_decay=cfg.hyperparameters.weight_decay,
        n_epochs=cfg.hyperparameters.n_epochs,
        evaluation_noise_std=evaluation_noise_std,
        joint_taus=joint_taus,
        evaluation_seed=evaluation_seed,
        l2_mode=l2_mode,
        prior_scale_target=prior_scale_target,
        objective_metric=objective_metric,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_epochs=early_stopping_min_epochs,
        debug_single_batch=debug_single_batch,
    )


def main_supervised_value_estimation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_dataset, val_dataset = prepare_data(
        cfg.dataset.endpoint_location,
        cfg.dataset.queries_train,
        cfg.dataset.queries_val,
        cfg.dataset.rdf2vec_vector_location,
        cfg.dataset.occurrences_location,
        cfg.dataset.tp_cardinality_location
    )

    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config, model_directory=cfg.models.oracle.dir
    )

    model_kwargs = {
        "full_gnn_config": cfg.models.embedder.config,
        "config_ensemble_prior": cfg.models.epinet.prior_config,
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "mlp_dimension": cfg.hyperparameters.mlp_dimension,
        "model_weights": cfg.models.epinet.model_file,
        "cost_only": True,
        "epinet_feature_mode": OmegaConf.select(
            cfg, "hyperparameters.epinet_feature_mode", default="mlp"
        ),
        "epinet_hidden_dim": OmegaConf.select(
            cfg, "hyperparameters.epinet_hidden_dim", default=None
        ),
    }

    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(cfg.hyperparameters.mlp_dimension, 1),
        }
    }
    heads_config_prior = {
        'plan_cost': {
            'layer': torch.nn.Linear(5, 1),
        }
    }

    epinet_cost_estimation = prepare_epinet_model(**model_kwargs, device=device,
                                                  heads_config=heads_config, heads_config_prior=heads_config_prior)
    experiment_base_dir = cfg.models.output.experiment_base_dir

    writer = ExperimentWriter(experiment_base_dir, "simulated_cost_epinet_training",
                              OmegaConf.to_container(cfg, resolve=True),
                              {k: v for k, v in model_kwargs.items() if
                               k not in ("model_weights", "heads_config", "heads_config_prior")})

    return main_simulated_epinet_training(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        oracle_model=oracle_model,
        epinet_cost_estimation=epinet_cost_estimation,
        model_kwargs=model_kwargs,
        model_builder_fn=prepare_epinet_model,
        device=device,
        writer=writer
    )


@hydra.main(version_base=None,
            config_path="../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="simulated_supervised_cost_estimation_mixed_yago_train_epinet.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    best_epinet_dir = find_best_epoch_directory(cfg.models.epinet.experiment_dir, "val_loss_cost_unscaled")
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_p99_q_error")

    cfg.models.embedder.dir = str(best_embedder_dir)
    cfg.models.oracle.dir = str(best_oracle_dir)
    cfg.models.epinet.dir = str(best_epinet_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    OmegaConf.set_struct(cfg, True)
    return main_supervised_value_estimation(cfg)


if __name__ == "__main__":
    main()