# System card: epinet plan-cost model (trial 85) and amortised-DP join planner

**Status:** research prototype. Offline and simulated results only; no real query executions.
**Date:** 30 September 2026. **Dataset:** mixed YAGO (star, path and complex SPARQL queries, 2–10 triple patterns).
**Code state:** GPULab training ran commit `e08797f`. The noise-fitted evaluation and the amortised-DP prototype are **uncommitted** (see §6).

This card covers two systems and the evaluation infrastructure built around them.

| System | What it does | Headline result (held-out test) |
|---|---|---|
| **A. Epinet plan-cost model**, sweep trial 85 | Predicts the log cost of left-deep join plans, with a joint epistemic distribution across the plans of one query | With observation noise fitted on validation, beats a fitted-noise base network on joint NLL at τ ≥ 4 on 6/6 seeds (+0.175 per target at τ = 8). Per-plan NLL is about a tie. |
| **B. Amortised DP planner** (prototype, no epinet) | Picks join orders greedily with a learned cost-to-go, instead of running dynamic programming | Geometric-mean cost ratio to the exact optimum is **1.071 greedy / 1.062 beam 8**, against 1.25–1.30 for all baselines, at 6–7 ms planning |

---

## 1. Shared setting

- **Task.** Left-deep join ordering for SPARQL basic graph patterns. A plan is an order of the triple patterns.
- **Cost used everywhere: C_out** = card(first pattern) + Σ card(each intermediate result), the same function as `JoinPlan` in `src/baselines/enumeration.py`. The first join is symmetric, so the cheaper orientation is used.
- **"Ground truth"** is the cardinality predicted by a large **oracle** GNN (`pretrain_experiment_huge_oracle_graph_norm_hll`, best epoch by p99 q-error). All targets and all evaluations here are relative to that oracle, not to true cardinalities or measured latency.
- **Splits.**
  - Train: 105,764 queries.
  - The original validation file (11,755 queries) is split 50/50 into **val** (5,877) and **test** (5,878).
  - The 588 queries the Optuna sweep validated on (and so selected trial 85 with) are forced into val, so the test set influenced no choice.
  - The split is saved in `experiments/experiment_outputs/mixed_yago/epinet_rerun_trial85/split.json`.
- **Weighting.** Every metric averages over *queries*: per-plan quantities get weight 1/n_plans. Query sizes range from 2 to thousands of plans.

---

## 2. System A: epinet plan-cost model (trial 85)

### 2.1 Model
- **Base network:** frozen query-embedding GNN, then a left-deep tree convolution, then an MLP, then a `plan_cost` head. Trained jointly with the epinet on simulated costs.
- **Epinet:** index dimension 16, learnable epinet hidden width 64, prior epinet hidden width 128, epinet features = MLP features only.
- **Prior:** MLP prior only (`alpha_mlp = 1.0`, `alpha_ensemble = 0.0`, so the GNN-ensemble prior is skipped), with no prior-scale calibration.
- **Training hyperparameters:** σ (target perturbation) = 0.0339; index samples per step K = **256** (the sweep used 32); lr 1e-3, weight decay 1e-3, batch of 8 queries.
- **Config:** `experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed/simulated_supervised_cost_estimation_mixed_yago_rerun_trial85.yaml`. Every setting is pinned there, because the base config changed after the sweep ran.

### 2.2 Training
- **Seeds:** 10 planned on GPULab, 2 per GPU; 6 have outputs (seeds 0–5). Each seed changes model initialisation, data order, index sampling and the anchor vectors.
- **Stopping:** up to 30 epochs, early stopping on validation joint NLL at τ = 8 (patience 5, at least 8 epochs). Best epochs for seeds 0–5: 20, 6, 27, 30, 18, 22.

### 2.3 Evaluation protocol
- **Joint NLL at τ ∈ {1, 2, 4, 8, 16}:** scored only on queries with at least τ distinct plans. The old metric padded small queries with repeated plans, so old sweep numbers are **not comparable**. Four baselines:
  - the epinet;
  - the epinet with samples shuffled independently per plan (same marginals, no dependence across plans);
  - the base network with fixed noise;
  - the base network with its noise **fitted on val**.
- **Noise fitting** (checkpoint re-scoring, `evaluate_epinet_checkpoints.py`):
  - The epinet's observation-noise std is fitted on val by per-plan (τ = 1) NLL over a 49-point grid, then applied unchanged to test.
  - Fitting on τ = 1 keeps the choice neutral with respect to the joint claims.
  - Sanity check: re-scoring reproduces each seed's logged training objective exactly (e.g. −0.08303 for seed 0), which confirms the checkpoint and split loaded as trained.
- **Also reported:**
  - point accuracy: MSE and q-error;
  - calibration: 50/80/95% interval coverage and calibration error;
  - selective prediction: area under the risk–coverage curve, and a skill score (1 = oracle, 0 = random);
  - plan-selection regret (mean vs Thompson sampling);
  - timing.

### 2.4 Results (test split, best-val epoch; seeds 0–5)

Mean ± 95% t-interval over 6 seeds; gains are paired within seed. Best epochs: 20, 6, 27, 30, 18, 22.

| metric | value | seeds > 0 |
|---|---|---|
| fitted noise std: epinet / base | 0.133 ± 0.006 / 0.194 ± 0.008 | – |
| gain over fitted base, τ = 1 | −0.012 ± 0.037 | 2/6 |
| gain over fitted base, τ = 2 | +0.025 ± 0.032 | 4/6 |
| gain over fitted base, τ = 4 | +0.103 ± 0.026 | 6/6 |
| gain over fitted base, τ = 8 | **+0.175 ± 0.021** | 6/6 |
| gain over fitted base, τ = 16 | **+0.200 ± 0.014** | 6/6 |
| dependence gain τ = 8 (vs shuffled), fixed → fitted noise | 0.156 → **0.284 ± 0.013** | 6/6 |
| coverage 50/80/95%, fitted noise | 68.1 / 89.6 / 96.9% (base: 66.2% at 50%) | – |
| calibration error: fixed / fitted / base | 0.094 / 0.071 / 0.051 | – |
| selective-prediction skill | 0.26 ± 0.20 (seed 1: −0.10) | 5/6 |
| MSE (standardised): epinet mean / base | 0.0393 / 0.0377 | – |
| regret (log-cost): base / epinet mean / Thompson | 0.059 / 0.059 / 0.086 | – |

**How to read this:**
- **The central claim holds.** The advantage is in **joint** prediction and grows with group size: coherent samples across the plans of a query. Per-plan it's a tie with a well-fitted Gaussian. This is the property approximate Thompson sampling needs (Osband et al., 2023).
- **Calibration** improves once the double-counted noise is removed, but stays worse than the base network. The centre is still too wide, and the residuals are heavy-tailed.
- **Thompson sampling** costs about 3 points of extra regret per decision offline: the price of exploration.
- **Seed 1** early-stopped at epoch 6 and has negative selective-prediction skill. Early stopping on the fixed-noise objective may stop too early.

### 2.5 Intended use and limitations
- **Intended use:** research on uncertainty-aware join ordering; initialisation for online training (Thompson or information-directed exploration).
- **Not validated on real latencies.** The simulated targets are deterministic, so the "noise" is an evaluation assumption; online latencies will need their own aleatoric model.
- **Uncertainty shouldn't be over-trusted:** it is informative (positive skill on 5 of 6 seeds) but weakly and unevenly so.
- **Scope of the results:** the τ = 16 results cover only queries with at least 16 plans (about 46% of test).

---

## 3. System B: amortised-DP join planner (prototype)

### 3.1 Idea
Learn the dynamic program instead of running it. The state is the **set** S of joined patterns, since the optimal future cost doesn't depend on join order. The model predicts:
- log card(S), and
- the optimal log **cost-to-go** G(S), where G(S) = min over a of [card(S ∪ a) + G(S ∪ a)] and G(full) = 0.

The agent ranks a candidate prefix by log(Σ predicted prefix cardinalities + predicted G). Greedy decoding on this is Q-greedy. Exact DP is used **only offline, as the teacher**.

### 3.2 Model (`src/supervised_value_estimation/amortized_dp/model.py`, 327,682 parameters)
- **Inputs:** 200-d triple-pattern embeddings from the same frozen GNN as the learned-cardinality baseline, plus the join graph (patterns sharing a **variable**).
- **Contracted join graph:** S is collapsed into one super-node, encoded with DeepSets plus an |S| embedding, so it's order-invariant by construction. Three rounds of message passing run over the super-node and the remaining patterns (hidden width 128).
- **Card head:** reads the set encoding only, so it's a pure set function (tested).
- **Cost-to-go head:** reads the contextual super-node, the pooled remaining patterns and the fraction of the query left.

### 3.3 Training (`train_amortized_dp.py`, config `amortized_dp_mixed_yago.yaml`)
- **Data:**
  - 30,000 random training queries, of which 4,009 are not variable-connected (see §5), leaving **25,991**.
  - Every connected subset is labelled with oracle log-cardinality, with exact G from bitmask DP.
  - 814,250 training states with 2,830,444 (state, child) pairs.
- **Losses:**
  - MSE on standardised log card and on log G;
  - **listwise DP distillation**: cross-entropy between the model's and the exact softmax(−log Q / T) over each state's children, with T = 0.1 (model) and 0.05 (target).
- **Schedule:** AdamW, lr 1e-3, cosine schedule, 30 epochs (about 25 s each on an RTX 3080).
- **Selection:** greedy decoding on 1,303 val queries. Best epoch 27, with val geometric-mean ratio 1.062.

### 3.4 Evaluation protocol (`compare_agents.py`)
- **Harness:** every agent runs through the existing `multiprocess_validate_agent` and `beam_search`, unchanged.
- **Scoring:** a drop-in `SimulatedCostExecutionStrategy` replaces the database. A plan scores C_out(plan) / C_out(exact left-deep DP optimum), both under the oracle, so 1.0 is optimal. That is an exact regret.
- **Test set:** 5,071 plannable test queries; 807 were excluded as not variable-connected, identically for all agents.
- **Planning time:** measured per query in single-threaded CPU workers.

### 3.5 Results (5,071 held-out test queries)

| agent | geo-mean ratio | p95 | p99 | optimal | within 2× | plan ms (mean) |
|---|---|---|---|---|---|---|
| exact DP over learned cardinality GNN | 1.300 | 6.45 | 59.7 | 68.4% | 89.7% | 53.2 |
| existing cardinality agent, beam 1 | 1.264 | 4.80 | 33.3 | 68.9% | 90.1% | 10.3 |
| existing cardinality agent, beam 8 | 1.943 | 50.6 | 615 | 57.0% | 77.6% | 23.6 |
| cardinality agent ranking by C_out, beam 1 | 1.254 | 4.53 | 32.0 | 69.1% | 90.2% | 12.8 |
| cardinality agent ranking by C_out, beam 8 | 1.295 | 6.15 | 56.1 | 68.4% | 89.7% | 15.3 |
| plan-cost model (trial 85 base head), beam 1 | 1.285 | 4.23 | 138 | 67.4% | 91.1% | 5.4 |
| plan-cost model (trial 85 base head), beam 8 | 1.249 | 3.87 | 103 | 67.5% | 92.5% | 9.3 |
| **amortised DP, greedy** | **1.071** | **1.43** | **4.06** | **75.1%** | **97.1%** | **6.1** |
| **amortised DP, beam 8** | **1.062** | **1.36** | **3.37** | 74.9% | **97.6%** | 6.9 |

Geometric-mean ratio by query size (triple patterns):

| agent | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| plan-cost model, beam 8 | 1.378 | 1.113 | 1.340 | 1.144 | 1.287 | 1.287 |
| amortised DP, greedy | 1.040 | 1.079 | 1.080 | 1.090 | 1.183 | 1.417 |
| amortised DP, beam 8 | 1.042 | 1.094 | 1.067 | 1.113 | 1.145 | 1.282 |

All agents are optimal on 2-pattern queries (ratio 1.000).

### 3.6 Caveats (read before citing)
1. **The fair head-to-head is against the plan-cost model.** Both learn from the oracle. The learned-cardinality baselines use a different, weaker GNN (mean q-error ≈ 16), and its disagreement with the oracle counts as error; exact DP over its estimates already scores 1.30. Compare those rows as a family, not as an architecture test.
2. **Denser teacher.** The amortised DP gets exact cost-to-go for every connected subset; the plan-cost model gets best-of-sampled-completion targets. The gain mixes architecture and target; an ablation training the same network on sampled-plan targets would separate them.
3. **Less data.** The amortised DP trained on 26k queries, the plan-cost model on 105k.
4. **Largest queries are the weakest point.** At 10 patterns greedy scores 1.42, and only beam 8 edges past the plan-cost model.
5. **No size extrapolation tested:** train and test sizes follow the same distribution. **Not tested on real latencies.**
6. **One training seed.** No confidence intervals yet.

### 3.7 Intended use
A prototype showing that a DP-aligned, set-based value function can replace dynamic programming at greedy planning cost. The next step is attaching an epinet to the cost-to-go head, for Thompson sampling or information-directed exploration online.

---

## 4. Evaluation and infrastructure changes

| Component | Change |
|---|---|
| `src/utils/epinet_utils/epinet_evaluation.py` (new) | All metrics in §2.3. Noise-grid NLL and calibration values per query; the epinet noise is fitted on val and reused for test. |
| `epinet_report.py`, `summarize_epinet_rerun.py` (new) | `metrics.jsonl`, per-epoch figures, TensorBoard; cross-seed tables with 95% CIs and figures. Re-scored checkpoints override training-time metrics. |
| Dyadic joint sampling | **Removed** from metrics, loss, tests and configs. It mostly measured repeated draws of the same plan. |
| `rerun_epinet_seeds.py`, `launch_seeds_parallel.py`, `dockerfiles/gpulab/submit_epinet_rerun.sh` (new) | Held-out split; one or more seeds per job, run sequentially or in parallel on one GPU; GPULab halt handling (exit 123, restartable); atomic `split.json`. |
| `evaluate_epinet_checkpoints.py` (new) | Re-scores saved checkpoints with fitted noise; exact sanity check against the training logs. |
| `amortized_dp/` (new) | Labels and DP teacher, model, agents (including a cumulative-C_out cardinality agent), simulated execution strategy, training, comparison. |
| Tests | 52 passing, covering the evaluation module and the amortised-DP package (DP = brute force = `JoinOrderEnumerator`; model invariances; greedy with exact values = DP optimum). |

## 5. Issues found in existing code

- **Queries that need a cartesian product.** `build_adj_list` treats a shared **constant** as a join edge, but `beam_search` joins only on shared **variables**. As a result:
  - 10–16% of queries have no plan without a cartesian product;
  - `beam_search` returns nothing for them;
  - `multiprocess_validate_agent` **silently skips** them, so other validation scripts may score agents on fewer queries than intended.
- **`CardinalityEstimatorValidationAgent` ranks by the current set's cardinality, not by plan cost.** At the last step every candidate ties, so with beam 8 the final choice is arbitrary (1.94 against 1.26 at beam 1).
- **Double-counted noise in the epinet evaluation.** The fixed evaluation noise (0.2) equalled the base residual std (0.193), while the epinet's own spread already covered its error. This caused the ~1.4× overdispersion seen before noise fitting.
- **Online pipeline** (`online_supervised_value_estimation.py`), reviewed but not changed:
  - it starts from an old `star_yago_gnce` checkpoint and dataset;
  - the online loss has no epinet stop-gradient and no separate base loss;
  - timeout labels disagree with the adaptive timeouts actually used;
  - a join-size indexing bug;
  - an unclamped head-blending weight;
  - a `numpy.dtypes.StringDType` import (needs numpy 2; the environments pin 1.26.4);
  - no baseline optimizer in validation;
  - possible QLever result-cache effects on measured latency.

## 6. Reproduction

```bash
# System A: training (GPULab; 5 jobs x 2 seeds in parallel), then re-scoring with fitted noise
PROJECT=phdexperimentsruben CLUSTER_ID=5 SEEDS_PER_JOB=2 dockerfiles/gpulab/submit_epinet_rerun.sh
python -m src.supervised_value_estimation.evaluate_epinet_checkpoints
python -m src.supervised_value_estimation.summarize_epinet_rerun experiments/experiment_outputs/mixed_yago/epinet_rerun_trial85

# System B
python -m src.supervised_value_estimation.amortized_dp.train_amortized_dp
python -m src.supervised_value_estimation.amortized_dp.compare_agents
```

**Outputs:**
- A: `experiments/experiment_outputs/mixed_yago/epinet_rerun_trial85/`, with per seed `seed-*/noise_fitted_eval/` and cross-seed `summary/`.
- B: `experiments/experiment_outputs/mixed_yago/amortized_dp/`, with the model in `run-30-09-2026-10-15-40/` and the results in `comparison-30-09-2026-10-42-25/summary.md`.

**Before relying on these results, commit the uncommitted code:**
- `src/supervised_value_estimation/amortized_dp/`
- `evaluate_epinet_checkpoints.py`
- the noise-fitting changes to `epinet_evaluation.py`, `epinet_report.py` and `summarize_epinet_rerun.py`
- `amortized_dp_mixed_yago.yaml`

## 7. Open items

1. Find out why seeds 6–9 have no outputs, and run them.
2. Amortised DP ablations: sampled-plan targets vs exact DP targets; without the rank loss; without contraction. Add a size-extrapolation split (train ≤ 6 patterns, test 8–10) and more seeds.
3. Put an epinet on the cost-to-go head, then run joint-NLL and Thompson-regret evaluation over candidate actions.
4. Fix the online-pipeline blockers in §5 before any online run.
