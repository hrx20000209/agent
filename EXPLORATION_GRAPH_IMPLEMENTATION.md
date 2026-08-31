# MobileExplorer Progressive Graph Implementation

## Architecture

The implementation keeps the existing `GoalExplorer` and ADB controller. It adds three replaceable components:

1. `ProgressiveBeliefGraph` with immutable `GraphSnapshot` reads and live writes.
2. `StateGraphInformationMatrix` plus `PredictiveElementScorer` for exploration.
3. `GraphDistiller` plus `GraphReasoningGate` for normal/enhanced/skipped reasoning.

Temporal invariant:

```text
capture UI i -> freeze snapshot generation g -> make gate/ranking decisions from g
             -> exploration/execution writes generation g+1... -> available at i+1
```

No graph operation invokes an LLM/VLM.

## Candidate matrix schema

Each `CandidateInformationRow` contains:

- UI: stable element identity, text, content description, role, probe type, normalized position, bounds, clickable/scrollable/enabled/selected/checked, nearby context.
- Node maturity: visits, decision entropy, outgoing/valid degree, explored/candidate counts, coverage, snapshot generation.
- Exact edge history: masked `has_exact_history`, probes, status, confidence, alignment/execution/skip/IG/rollback/cost statistics, inverse level, destination.
- Destination: visits, entropy, degree, two-hop subtree size, known labels, recent-path membership.
- Context history: role + probe type + coarse UI context + InformationNeed type aggregate rates.
- InformationNeed: target, affordance, unresolved-information, action-type matches and risk conflict.
- Safety/recovery: blocked element/context, rollback/deep recovery, cross-package, risk and recoverability.
- Cost: expected probe, rollback and total cost.
- Predictions: path probability, expected IG, recoverability, cost, predictive value and final score.

Unavailable exact statistics are `null`, never misleading zeroes.

Element identity is SHA-1 over:

```text
action type + role/class + resource-id (description/text fallback) + normalized position bucket
```

## First-version exploration formula

All values are clipped to `[0, 1]`.

```text
Need = .42 target
     + .25 affordance
     + .25 unresolved information
     + .08 action-type match
     - .50 risk conflict

PathProbability = .42 Need
                + .22 exact/context alignment rate
                + .18 exact/context execution hit rate
                + .12 contextual alignment rate
                + .06 historical realized IG
                + .07 novelty (only when exact history is absent)

ExpectedIG = .38 Need
           + .22 node decision entropy
           + .18 UI novelty
           + .14 historical realized IG
           + .08 destination value
           + .10 coverage * novelty

PredictiveValue = PathProbability * ExpectedIG

FinalScore = PredictiveValue * EstimatedRecoverability / ExpectedCost
```

Safety is a hard feasibility gate. Blocked, destructive, low-recoverability, disabled, or high-risk candidates receive score zero and are removed before selection.

`new_element_count` is not used as information gain. UI novelty and realized entropy reduction are recorded separately.

## How history affects exploration

- Never explored: relies on current UI, InformationNeed, uncertainty, novelty and contextual history.
- Later model actions repeatedly align: raises path probability.
- Execution repeatedly reaches the predicted destination: raises confidence/path probability.
- Useful destination structure and positive realized IG: raises expected IG.
- Rollback failures: lowers recoverability and eventually blocks the edge.
- Irrelevant/repeated/no-effect branches: reduce history utility and novelty.
- High node coverage: emphasizes safe unexplored candidates that can reduce remaining entropy.

## Graph edge updates

All statistics are updated through:

- `record_probe`
- `record_realized_information_gain`
- `record_inference_alignment`
- `record_execution_verification`
- `record_skip_result`
- `record_rollback_result`

Edge statuses progress through:

```text
SPECULATIVE -> OBSERVED -> INFERENCE_ALIGNED -> VERIFIED -> REUSABLE
                                                   \-> INVALID/BLOCKED on repeated failures
```

## GraphFact and deterministic distillation

`GraphFact` contains:

- fact type: `VERIFIED`, `OBSERVED`, `DONE`, or `NO_RELEVANT_EVIDENCE`
- factual action label and source edge ID
- evidence labels and internal edge status/certainty
- need match, historical utility, freshness, taken/risk flags and utility score

Algorithm:

1. Retrieve valid outgoing edges from the current node.
2. Convert local edges to facts.
3. Compute task-conditioned utility:

   ```text
   NeedMatch * Certainty * HistoricalUtility * Freshness
   ```

4. Remove weak, risky, stale or already-consumed facts.
5. Greedily select up to `max_graph_facts` under the token budget.
6. Render a fixed factual template. No recommendations and no raw confidence numbers.

Real ADB smoke-trace example:

```text
[Memory]
Need: save contact.
Observed: Save led to {Add info to save as a contact., OK}.
```

## Three-way gate

`NORMAL_INFERENCE`:

- graph reasoning is off; or
- current UI has no snapshot node; or
- no task-relevant fact passes the utility threshold.

`GRAPH_ENHANCED_INFERENCE`:

- no safe reusable edge passes skip conditions; and
- at least one local fact fits the graph token budget.

`SKIP_INFERENCE` requires all of:

- exact current UI node match in the previous-generation snapshot;
- status `VERIFIED` or `REUSABLE`;
- confidence at least `0.82`;
- node entropy at most `0.35`;
- age at most 40 graph generations;
- risk at most `0.25`;
- rollback rate, when known, at least `0.80`;
- destination not in the recent path.

The stored action is executed through the existing ADB action layer. The successor accessibility state is captured immediately. A mismatch records an execution miss and failed skip, stops reuse, and returns the next iteration to the model.

## Logs

- `explore_results/candidate_matrix.jsonl`: every candidate feature/prediction/rank plus selection update.
- `explore_results/explore_log.jsonl`: actual probe, destination, realized IG and rollback trace.
- `explore_results/graph_distillation.jsonl`: raw/candidate/selected fact counts, fact types/edge IDs, chars/tokens, injected flag and graph mode.
- `explore_results/graph_metrics.jsonl`: future real-action rank, top-1/top-K hit, aligned edge and graph-history usage.
- graph JSON: nodes, edges, generations and all cumulative statistics.

## Ablations

Exploration:

```bash
python main.py --exploration_policy information_need
python main.py --exploration_policy graph_matrix
python main.py --exploration_policy graph_matrix --disable_exact_history
python main.py --exploration_policy graph_matrix --disable_contextual_history
python main.py --exploration_policy graph_matrix --disable_information_need
python main.py --exploration_policy graph_matrix --disable_cost
python main.py --exploration_policy graph_matrix --disable_recovery_history
```

Reasoning:

```bash
python main.py --graph_reasoning off
python main.py --graph_reasoning briefing
python main.py --graph_reasoning distill
python main.py --graph_reasoning skip_only
python main.py --graph_reasoning distill_and_skip
```

Graph cache budget:

```bash
python main.py --graph_memory_budget_mb 1
```

## Physical-device memory/throughput evaluation

The evaluator rejects emulators by default.

```bash
python evaluate_graph_memory_adb.py \
  --adb_serial YOUR_PHYSICAL_SERIAL \
  --duration_sec 8 \
  --budgets_mb 0.001,0.005,0 \
  --adb_retries 3 \
  --output_dir evaluation_results/physical_phone_8s
```

Outputs:

- `samples.jsonl`: node/edge counts, Python deep size, serialized size, process RSS and pruning at each probe.
- `summary.json` / `summary.csv`: fixed-window probe throughput per budget, rollback outcomes, device `MemAvailable`, foreground App PSS and memory slopes.

`--allow_emulator` exists only for explicitly non-final smoke testing.

## Remaining heuristics / future learned predictor data

Still heuristic: state signature, entropy estimator, contextual key, scorer weights, fact threshold, cost fallback, freshness and skip thresholds.

The logs now provide supervised rows for a future predictor:

- full UI/node/exact/context/need/recovery/cost feature matrix;
- selected/rank labels;
- actual destination and rollback outcome;
- future model-action rank/alignment;
- execution hit/miss and skip success;
- realized entropy reduction and measured exploration cost.
