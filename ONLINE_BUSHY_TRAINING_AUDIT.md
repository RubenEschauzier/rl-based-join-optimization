# Online bushy planning: training audit and recommendations

Date: 2026-10-15

## Executive summary

The current system is not simply a larger model trained on exact latency. It is a jointly updated cardinality/value/latency model trained on selectively observed merge timings and stitched minimum-time completions, without directly learning from total query latency or timeout penalties.

The earlier explanation—self-generated data, limited coverage, and search exploiting prediction errors—is plausible but incomplete. More concrete issues are:

1. Total query latency and timeout censoring do not directly supervise the latency heads.
2. Timing targets combine per-merge minima with minimum-cost stitching, favoring optimistic predictions rather than expected latency or tail safety.
3. Forest masks omit component topology and physical properties that can affect latency.
4. Training reconstructs a fixed merge schedule instead of retaining every forest trajectory actually visited by search.
5. Latency and step losses can change the pretrained cardinality/value heads and shared forest encoder.
6. Lazy QLever execution makes the assumption of exact, plan-independent observed cardinalities worth checking.

**Scope and verification:** This was a read-only source audit. No production source was changed. Small source-derived reproductions exercised the parser, observation store, stitching, and tree utilities. Actual checkpoint/autograd verification was unavailable: the local Python environments lacked PyTorch, and the experiment outputs were not present. Runtime effects below are distinguished from verified code behavior.

## 1. What the supplied logs establish

### Beam search is not consistently worse

At validation round 20:

| Planner | Mean latency | Timeouts |
| --- | ---: | ---: |
| Latency greedy | 557.7 ms | 1.4% |
| Latency beam-4 | 452.5 ms | 1.0% |
| Latency beam-8 | 405.2 ms | 0.8% |

Beam search sometimes helps substantially and sometimes hurts. This is consistent with prediction error and search exploiting underestimation, but does not prove that out-of-distribution states are the cause.

### A few catastrophic plans dominate the mean

Assuming a timeout contributes approximately 30 seconds, round-20 latency-greedy receives approximately `0.014 × 30,000 = 420 ms` of its 557.7 ms mean from timeouts alone. Its successful executions average roughly 140 ms, versus GNCE's 116 ms.

The problem is both ordinary ranking quality and rare catastrophic mistakes—not uniformly five-times-slower execution. Training should explicitly address catastrophic outcomes.

### C_out degradation starts before the latency-policy switch

C_out/QLever moves from 0.312 at round 0 to 0.392 at round 5, while collection still uses C_out. Switching collection to latency therefore cannot explain all degradation.

### Checked-in defaults are not necessarily the executed configuration

The [checked-in online configuration](experiments/experiment_configs/amortized_dp/bushy_online_mixed_yago_v2.yaml) specifies 300 validation queries and 256 training queries per round. The supplied output has 500 validation queries and approximately 3,900 plans per round. Inspect the run's resolved configuration before assuming its hyperparameters match the defaults.

## 2. Target-construction findings

### 2.1 Actual query latency and censoring do not directly supervise the model

[`ForestObservations.add()`](src/supervised_value_estimation/amortized_dp/online/observations.py#L111-L126) records total latency, censoring, and timeout.

However, [`query_rows()`](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L99-L135) constructs targets from observed row counts, stitched C_out, observed merge times, and stitched remaining merge times. [`online_loss()`](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L138-L174) never directly compares a prediction with the executed plan's total latency.

Consequences:

- A genuine timeout does not produce an explicit constraint that the plan costs at least its timeout threshold.
- Unfinished expensive joins generally lack timing targets.
- Completed cheap portions can still contribute training examples.
- The model can underestimate the expensive, unfinished portion without receiving the strongest available negative signal.

**Verified reproduction:** Holding parsed merge observations fixed while changing total latency and censoring left timing targets unchanged. Total latency can still affect later collection through adaptive timeouts; the missing link is direct loss supervision.

**Priority:** Highest-priority training-signal issue.

### 2.2 The timing target is a fastest-observed stitched completion

[`ForestObservations.add()`](src/supervised_value_estimation/amortized_dp/online/observations.py#L116-L120) retains the minimum observed time for each operand-mask pair. [`latency_to_go()`](src/supervised_value_estimation/amortized_dp/online/observations.py#L128-L160) then minimizes over completions assembled from those timings.

This introduces two layers of optimism:

1. Fastest observed timing for each merge.
2. Cheapest combination of those timings.

**Verified reproduction:** Repeated executions with 30 ms and 300 ms merge-time totals retained a 30 ms target.

This can approximate best-case execution, but is not a target for mean latency or timeout risk. The asymmetric latency loss also penalizes underprediction only one-tenth as strongly as overprediction under the checked-in configuration.

Restricting available plans gives an upper bound under a deterministic, correct cost model. Taking noisy minima does **not** guarantee an upper bound on expected real execution latency.

### 2.3 Forest masks may not be a sufficient latency state

The [forest representation](src/supervised_value_estimation/amortized_dp/bushy_model.py#L1-L29) represents components as sets of patterns, not internal trees. Components with identical pattern masks therefore have identical representations even if they were produced by different join trees, sorted orders, physical operators, or materialization strategies.

Those differences can affect the next join's cost. The timing store also identifies each merge only by its two operand masks, allowing a cheap way to build a component to be combined with a cheap continuation measured for another physical realization of that component.

**Verified constructed reproduction:** Two executed plans costing 101 ms and 1001 ms yielded a stitched target of 2 ms. This is a feasible completion cost only if the cheap continuation transfers between the component realizations.

The reproduction demonstrates the assumption; it does not quantify its violation in the actual workload.

**Implication:** More model capacity cannot recover topology or physical properties absent from the input.

The same assumption affects [beam deduplication](src/supervised_value_estimation/amortized_dp/bushy_train.py#L709-L739): histories reaching the same forest are collapsed. This is appropriate for set-based C_out but potentially unsafe for physical latency.

### 2.4 Reconstructed training trajectories can differ from searched trajectories

[`query_rows()`](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L103-L111) reconstructs states using `tree_to_merges()`, which visits the left subtree first in postorder.

For a tree `((0, 1), (2, 3))`, training reconstruction builds `{0,1}` first. Planning could have built `{2,3}` first. These yield different intermediate forests despite producing the same final tree.

**Verified reproduction:** The actual tree utilities produced different first-merge forests for those schedules.

Executing a searched tree therefore does not guarantee that its searched forest trajectory enters training. Retaining actual planning trajectories would directly address this coverage mismatch.

Candidate rows are additionally discarded unless their union has an observed cardinality and their child has a known C_out completion. Thus latency supervision is gated by C_out-label availability.

### 2.5 Exact cardinality assumptions need checking against lazy QLever execution

[`_finished()`](src/supervised_value_estimation/amortized_dp/online/runtime_tree.py#L34-L36) accepts statuses containing `completed` or `materialized`.

The repository's [saved QLever example](src/supervised_value_estimation/utils/visualize_qlever_runtime_output.py#L26-L27) contains a scan with:

- Status `lazily materialized completed`.
- One of 72 blocks read.
- `result_rows = 20724`.

**Verified reproduction:** The actual parser accepts these row counts. On this saved example, its four merge timings also sum to the root's 6 ms total.

This does not establish that every join cardinality is wrong. It establishes that lazy/block-filtered results enter a path that treats observed rows as exact subset cardinalities. Whether those counts are plan-independent requires verification against the real workload and QLever semantics.

Observations overwrite previous cardinalities. If counts depend on execution context, this can create contradictory C_out targets. Log repeated disagreements for the same subset.

## 3. Frozen models and gradient-flow audit

| Component | Source behavior | Assessment |
| --- | --- | --- |
| External pretrained embedding GNN | Embeddings produced under `no_grad`, cached, outside optimizer | Effectively frozen |
| Forest set encoder and message passing | Included in `AdamW(model.parameters())` | Trainable |
| Pretrained cardinality and G heads | Included in optimizer | Trainable |
| New latency and step heads | Used in loss and included in optimizer | Not accidentally frozen |
| Latency inputs from cardinality/G outputs | Not detached | Latency loss can update cardinality/G |
| Step input from cardinality output | Not detached | Step loss can update cardinality head |
| Epinet modules in bushy path | Not called by its loss or search | No useful epinet training or uncertainty-aware selection in this path |

Sources: [embedding extraction](src/supervised_value_estimation/amortized_dp/bushy_train.py#L80-L88), [optimizer construction](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L275-L279), and [head dependencies](src/supervised_value_estimation/amortized_dp/bushy_model.py#L185-L216).

No obvious missing-gradient or accidentally frozen latency-head bug was found by source inspection. The concerning interaction is the opposite: fresh latency heads can modify the pretrained encoder and cardinality/value heads while learning sparse, optimistic targets.

This is a plausible mechanism for C_out degradation, including during warm-up. Warm-up changes the collection objective, not which losses or parameters are trained.

**Recommended ablation:** Train only new latency/step heads initially, preserving pretrained features and outputs. Then test staged unfreezing with lower encoder learning rates. Joint training is not inherently incorrect; its benefit needs measurement.

**Verification still needed:** Backpropagate each loss separately on a real batch and measure gradient norms and parameter updates by module. Source inspection establishes routes, not their numerical magnitude.

### Loss interpretation cautions

- The logged `rank` loss ranks C_out, not latency.
- Listwise cross-entropy has a nonzero target-entropy floor. Rising raw loss across changing replay data is not by itself proof of worse ranking.
- Per-state normalization and combining unstandardized log-costs through `logaddexp` are not demonstrated bugs.
- Encoding state identity as `query * 100000 + local_state` does not collide merely because there are many queries. Collision requires overlapping local-state ranges across the stride; that was not established here.
- Dropping timing above cached subtrees is conservative and is not automatically a parser bug.
- Misaligned observations are not rejected before storage, but the supplied logs report 100% alignment. Moreover, subset-derived observations from a reordered tree are not automatically all incorrect. This is a robustness concern, not a demonstrated explanation for this run.

## 4. Where to add diagnostic observations

### After execution and parsing

At the [execution ingestion point](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L391-L395), record:

- Completion status and cache-status breakdown.
- Fraction of joins/scans with usable timing and cardinality.
- Timing residuals before clamping negative values to zero.
- Sum of extracted merge times versus top-join time versus total query latency.
- Timeout, cancellation, transport failure, and parser failure separately.
- Endpoint identity and execution order to detect server/cache effects.

### In the observation store

At [`ForestObservations.add()`](src/supervised_value_estimation/amortized_dp/online/observations.py#L111-L120), retain:

- Timing sample counts and distribution summaries, rather than only minima.
- Cardinality disagreement for identical masks across executions.
- Timing variation by subtree topology, operator, sorting, and materialization.
- Which execution supplied each retained timing, to audit cross-execution stitching.

### During row construction

At [`query_rows()`](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L99-L135), measure:

- Actual searched forests versus reconstructed training forests.
- Candidate rows dropped by reason.
- Valid latency and step fractions by query size and forest depth.
- Fraction of latency targets that are trivial terminal zeros.
- Number of observed alternatives per parent forest.

### During optimization

At [`online_loss()`](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L138-L174), measure:

- Each loss's gradient norm into shared encoder, card, G, latency, and step modules.
- Gradient alignment/conflict between C_out and latency tasks.
- Parameter update norm relative to parameter norm.
- Predicted step-plus-remaining Q ranking accuracy, not only individual-head MSE.
- Frequency of negative unstandardized log1p predictions and values hitting inference floors.
- Target entropy and excess cross-entropy/KL for the listwise loss.

### During validation

At [validation](src/supervised_value_estimation/amortized_dp/online/train_online_bushy.py#L293-L359), report:

- Per-query predictions versus observed total latency.
- Plan disagreement with GNCE and previous checkpoints.
- Tail regressions and timeout identities.
- Same-tree timing variability across repeated measurements.
- Selection regret within the executed candidate set.
- Candidate-oracle latency: the fastest available candidate per query.
- Successful-query metrics separately from timeout-inclusive metrics.

## 5. Where to collect additional execution data

Allocate training-query execution slots across:

1. GNCE's plan as a stable strong anchor.
2. Current learned greedy plan.
3. A learned beam candidate.
4. An uncertainty- or diversity-selected candidate.

This gives within-query comparisons of whether the learned planner actually beats GNCE, rather than only observing its own local neighborhood.

Retain actual search trajectories. For additive set-based targets, alternate valid merge schedules can augment forest coverage without new database executions. Do not assume this augmentation supplies exact physical-latency targets if costs depend on topology or execution context.

## 6. Recommended latency-training alternative

### First experiment: topology-aware complete-plan reranking

1. Generate candidates from GNCE greedy/beam and learned/diverse alternatives.
2. Encode the whole candidate tree, preserving topology.
3. Predict actual total query latency.
4. Train within-query comparisons and use timeout lower bounds.
5. Keep GNCE's plan as a fallback candidate.

For successful executions, regress an appropriate latency transform. For genuine timeout threshold `tau`, use a censoring-aware likelihood or at least a lower-bound penalty:

$$
L_{\mathrm{timeout}} = \left[\max\left(0, \log \tau - \widehat{\log T}\right)\right]^2.
$$

Transport or infrastructure failures must not be treated as trustworthy runtime lower bounds.

**Important:** Adding total latency as a target to the existing final-forest representation is insufficient. Every complete plan for a query collapses to the same full-set forest. A total-plan predictor needs topology or physical-plan information to distinguish those candidates.

Before training, measure candidate-oracle performance. If selecting the fastest executed candidate barely improves on GNCE, candidate generation—not the reranker—is the limiting factor.

### If retaining the forest planner

Prioritize:

- Direct executed-plan latency supervision in addition to merge losses.
- Explicit timeout supervision.
- Distributional or robust timing estimates instead of per-merge minima.
- Topology or physical properties in component states.
- A latency-ranking loss aligned with the actual search score.
- Head-only initialization, then conservative unfreezing.
- Offline C_out replay or distillation to preserve useful pretrained behavior.
- Training coverage matching the forests encountered by beam search.

More capacity becomes useful after target and representation issues are addressed.

## 7. Epinet assessment

The left-deep path implements:

- [Epinet training and prior calibration](src/supervised_value_estimation/amortized_dp/train_amortized_dp.py#L215-L259).
- [Thompson-sampled plan collection](src/supervised_value_estimation/amortized_dp/online/exploration.py#L34-L63).
- [CVaR-based robust selection](src/supervised_value_estimation/amortized_dp/agents.py#L139-L165).

The [epinet loss](src/utils/epinet_utils/epinet_loss.py#L63-L86) detaches base predictions and features for its auxiliary training term. Those mechanisms are not wired into the bushy trainer. Constructing epinet modules alone does not activate them; the left-deep agents/calibration also use different state inputs and cannot be assumed to be drop-in bushy implementations.

Useful roles after improving the base signal:

- **Collection:** Thompson sampling with one epistemic index held fixed across an entire plan.
- **Deployment:** Calibrated risk-aware candidate ranking.
- **Diagnostics:** Test whether uncertainty predicts ranking error, large residuals, and timeouts.

Epinet uncertainty cannot distinguish physically different plans with identical model inputs, and cannot learn a timeout penalty absent from the loss. Calibration must be verified rather than assumed.

## 8. Recommended experimental order

1. **Verify target fidelity:** lazy row counts, repeated subset consistency, and extracted-time versus total-latency agreement.
2. **Run a head-only ablation:** isolate damage to the pretrained C_out model.
3. **Execute GNCE anchors and learned candidates on the same training queries.**
4. **Measure candidate-oracle improvement:** establish the available optimization headroom.
5. **Train a topology-aware total-latency reranker with censoring.**
6. **Add epinet-driven collection and risk-aware selection** once uncertainty is calibrated against a trustworthy signal.

This directly tests whether actual latency supervision improves on GNCE without first requiring a stitched forest value function to be a correct physical execution model.