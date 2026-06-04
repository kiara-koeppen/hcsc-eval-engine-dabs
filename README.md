# HCSC Automated Evaluation Engine — DABs Demo

A Databricks Asset Bundle (DAB) demo built for HCSC's Program Evaluation / Health
Economics team. It shows how to orchestrate a config-driven causal **program-evaluation
engine** (data processing → matching → modeling → insights) in a way that **scales flat**
as you add matching methods, evaluation models, and studies — eliminating the
exponential If/else-task branching the team is hitting today.

> **The problem we're solving.** Today the team's job emits a "next phase" value from a
> notebook, and an **If/else condition task per candidate notebook** routes true/false to
> the matching path, then to the model path. Every new matching method or model adds a new
> task + a new condition + DAG rewiring, and model routing gets copy-pasted under each
> matching branch — an `M × N` blowup. With a target of ~10 matching methods × ~20 models,
> that's hundreds of hand-wired tasks **per study**.
>
> **The fix.** Drive selection with a **config table**, dispatch on a **parameter** inside
> a single notebook per stage, and fan out across studies with a **`for_each`** task. The
> DAG stays at a fixed 3 stages no matter how many methods, models, or studies you add. The
> only condition task that remains is the legitimate **pass/fail data-quality gate**.

---

## What's in the box

Two jobs you can run side by side to make the contrast obvious in a demo:

| Job | Pattern | Point |
|-----|---------|-------|
| **`eval_engine_before`** | An If/else condition task per candidate notebook (⚠️ current state) | Watch the DAG: 2 matching × 2 models already = 4 model tasks + 4 routing conditions, most **skipped** every run. Doesn't scale; no study-level fan-out. |
| **`eval_engine_after`** | `load_config` → `for_each(studies)` → `run_single_study` (✅ scalable) | Flat DAG. Add a study = one config row. Add a method/model = one branch in a notebook. `for_each` fans out (and parallelizes) over all studies. |
| `run_single_study` | The reusable per-study pipeline invoked by `for_each` | feature_engineering → **quality_gate** → matching → modeling → insights |
| `setup_eval_engine` | One-time data setup | Builds `study_config` + synthetic `cohorts` |

### The "after" DAG (flat, regardless of scale)

```
load_config ──▶ for_each(studies, concurrency=5) ──▶ run_single_study (per study)
                                                         │
   feature_engineering ─▶ quality_gate ─true─▶ matching ─▶ modeling ─▶ insights
                              └─false─▶ (study stops; nothing downstream runs)
```

`matching` and `modeling` are **single parameterized notebooks**. `matching_method`
(`exact`|`knn`|`ipw`) and `model_method` (`att`|`ate`|`did`|`gee`|`mixed`) select the
algorithm *inside* the notebook — not via a different task.

### The "before" DAG (grows with every notebook)

```
prep_and_route ─▶ gate ─true─┬─▶ route_exact ─true─▶ exact_matching ─┬─▶ route_exact_att ─true─▶ model (exact+att)
                             │                                       └─▶ route_exact_did ─true─▶ model (exact+did)
                             └─▶ route_knn   ─true─▶ knn_matching   ─┬─▶ route_knn_att   ─true─▶ model (knn+att)
                                                                     └─▶ route_knn_did   ─true─▶ model (knn+did)
```

The model layer is **duplicated under every matching branch** (a converging task can't read
a task value from a sibling branch that got skipped). That's the `M × N` explosion.

---

## Why DABs (the orchestration value)

- **Define once, deploy anywhere.** `databricks.yml` parameterizes `catalog`/`schema` per
  target (`dev` / `test` / `prod`). Promote the same code across environments by changing
  variables, not code.
- **Parameterized notebooks.** Notebooks read `dbutils.widgets` (catalog, schema, study_id,
  method choices). The job YAML feeds those widgets from job parameters / config — exactly
  the "the YAML feeds the pipeline" idea from our call.
- **Resource globbing.** `include: resources/*.yml` — drop a new `*.job.yml` in `resources/`
  and it's part of the bundle. No central registry to maintain.
- **Source-controlled + reproducible.** The whole pipeline (jobs + notebooks + config) is
  one versioned bundle you `deploy` and `run` with a single command.

---

## Modular / multi-team track (post-meeting update, June 2026)

The working session surfaced three requirements the original "consolidate into one
parameterized notebook" approach did not fit:

1. **Keep notebooks separate.** Multiple model-dev teams build in parallel; merging
   everything into one notebook causes breakage. Notebooks must stay in their own files,
   each owned by a team.
2. **Let teams plug in new methods/models** without touching anyone else's code.
3. **Handle structurally different models** like **LME** (one baseline + several
   performance periods), whose feature engineering and data shape differ from the standard
   single pre/post studies.

The **modular track** (`resources/modular.job.yml`) handles all three with a **config-driven
branch**: the config decides, then the job splits into the shared loop plus **one dedicated
branch per structurally-different model**.

```
study_config_modular ─(load_config_branched)─▶ split by model_family + set branch flags
   ├─ has_standard?   ─true─▶ for_each(standard studies) ─▶ run_study_modular
   │                             dispatch_feature ─▶ gate ─▶ dispatch_matching ─▶ dispatch_model ─▶ insights
   │                             (shared pipeline; dispatchers run the config-named leaf per stage)
   ├─ has_lme_mixed?  ─true─▶ run_lme_mixed   (OWN branch, OWN notebooks, NO loop)
   │                             feature_lme_mixed ─▶ gate ─▶ matching_lme_mixed ─▶ model_lme_mixed ─▶ insights
   └─ has_lme_growth? ─true─▶ run_lme_growth  (OWN branch, OWN notebooks, NO loop)
                                 feature_lme_growth ─▶ gate ─▶ matching_lme_growth ─▶ model_lme_growth ─▶ insights
```

See `docs/modular.png` for the diagram.

- **Standard family** (ATT, ATE, DID, pre/post — single baseline + performance period) shares
  one per-study pipeline (`run_study_modular`) and fans out with `for_each`. Each stage is a
  thin **dispatcher** that runs the config-named leaf via `dbutils.notebook.run(<leaf>, ...)`,
  so every model stays in its **own** notebook owned by its team. (Job `notebook_path` is
  static and cannot be set from a parameter — verified against the docs — so the notebook is
  selected in code, where the path is a runtime string. Each leaf runs as its own child run,
  preserving per-notebook visibility.)
- **Each structurally-different model gets its own branch.** LME models do **not** ride the
  standard pipeline and are **not** looped. Each is a separate branch off the job
  (`run_lme_mixed`, `run_lme_growth`) with its **own** feature/matching/model notebooks, so it
  can diverge in shape completely. This matches what the team described: "the next path is also
  LME but completely different... we're not going to use the for each." `lme_mixed` (constant
  per-period effect) and `lme_growth` (effect that changes over time, a slope) are two example
  branches — rename or add more for your real models.

**What this buys each requirement:**

| Requirement | How the modular track meets it |
|-------------|-------------------------------|
| Separate, team-owned notebooks | Every method/model is its **own file** under `src/notebooks/registry/`. No shared mega-notebook to merge. |
| Plug in a new standard method/model | Add a leaf notebook + one row in `study_config_modular`. No DAG edit. (`matching_stratified` demonstrates a "third team" contributing a method.) |
| Structurally different model | Gets its **own branch + own notebook trio** off the job (e.g. `run_lme_mixed`, `run_lme_growth`), not forced through the standard pipeline and not looped. Add another = its rows + a `has_<family>` flag in `load_config_branched` + a branch in the job. |
| Observability | Standard leaves run as child runs via `dbutils.notebook.run`; each LME branch's stages are first-class tasks. Both fully visible in the run tree. |

**Standard leaves / dispatchers** (`src/notebooks/registry/`): `feature_standard`,
`matching_standard` (exact/knn/ipw), `matching_stratified`, `model_att`, `model_did`, run by
`dispatch_feature` / `dispatch_matching` / `dispatch_model`. **LME branch notebooks:**
`feature_lme_mixed` / `matching_lme_mixed` / `model_lme_mixed` and `feature_lme_growth` /
`matching_lme_growth` / `model_lme_growth`. Routing: `load_config_branched`.

> The branched engine was deployed and run green in `kk_test` (June 2026): `load_config_branched`
> routed all three branches, the 3 standard studies ran through the shared `for_each` loop, and
> each LME model ran through its own branch (`run_lme_mixed`, `run_lme_growth`) with its own
> notebooks. A non-serverless cluster variant (`*_cluster` jobs) mirrors the same branches.

---

## Architecture / data flow

```
study_config ──(load_config)──▶ studies[]  ──for_each──▶  per study:
cohorts ──▶ feature_engineering ─▶ feat_<study>  ─▶ matching ─▶ matched_<study> ─▶ modeling ─▶ evaluation_results ─▶ insights ─▶ evaluation_insights
                     │
                     └─ quality gate: writes quality_passed task value; gate short-circuits bad-data studies
```

**Tables** (in `${catalog}.${schema}`):
- `study_config` — one row per evaluation; the 3 method columns drive dispatch.
- `cohorts` — synthetic patient-level treatment/control data for all studies.
- `feat_<study_id>`, `matched_<study_id>` — per-study intermediate outputs.
- `evaluation_results` — one row per completed study (`source` = `before`/`after`).
- `evaluation_insights` — plain-language interpretation per study.

The synthetic data bakes in a known treatment effect per study and makes treated patients
sicker (confounding), so matching/weighting actually changes the estimate vs. a naive
comparison. **`STUDY_006`** has tiny n + heavy missingness and is intentionally **failed by
the quality gate** to demonstrate the short-circuit.

---

## Prerequisites

- Databricks CLI **v0.230+** (built with v0.292.0). `databricks --version`
- A profile in `~/.databrickscfg` (this demo uses **`kk_test`**).
- Serverless compute enabled in the workspace (jobs run serverless — no cluster config).
- The target catalog (`kk_test`) must already exist; the setup notebook creates the schema.

### ⚠️ Known CLI hiccup: Terraform download

Some CLI versions fail to download Terraform with
`unable to verify checksums signature: openpgp: key expired`. Point the CLI at a local
Terraform binary (`brew install terraform`):

```bash
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform
export DATABRICKS_TF_VERSION=1.12.1
```

---

## Deploy & run

```bash
# from repo root
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform DATABRICKS_TF_VERSION=1.12.1  # if needed

databricks bundle validate -t dev
databricks bundle deploy   -t dev

# 1) one-time: build config + synthetic data
databricks bundle run setup_eval_engine -t dev

# 2) the scalable pattern — fans out over all studies
databricks bundle run eval_engine_after -t dev

# 3) (optional) the current-state pattern, for contrast — runs one study
databricks bundle run eval_engine_before -t dev
```

### Modular / multi-team track

```bash
# one-time: registry config + single-period + multi-period (LME) data
databricks bundle run setup_modular -t dev

# branch: standard family loops; each LME model runs its own branch
databricks bundle run eval_engine_modular -t dev
```

Add a **standard** model: drop `src/notebooks/registry/model_<x>.py`, add a row to
`study_config_modular` with `model_nb = model_<x>`, re-run `eval_engine_modular`. No DAG edit.
Same for a new matching method (`matching_nb`) or feature flow (`feature_nb`).

Add another **structurally-different** model (its own branch): add its notebook trio under
`src/notebooks/registry/`, a config row with `model_family = <family>`, a `has_<family>` flag
in `load_config_branched.py`, a `run_<family>` job, and one branch in `eval_engine_modular`.

### Run without serverless (existing cluster)

For workspaces where serverless is not enabled, use the `*_cluster` jobs and point them at
an existing all-purpose cluster via the `compute_cluster_id` variable:

```bash
CID=0712-123456-abcde12   # your cluster: Compute -> cluster -> Configuration, or: databricks clusters list

databricks bundle deploy                  -t dev --var="compute_cluster_id=$CID"
databricks bundle run setup_modular_cluster   -t dev --var="compute_cluster_id=$CID"
databricks bundle run eval_engine_modular_cluster -t dev --var="compute_cluster_id=$CID"
```

Only notebook tasks bind to the cluster; `condition`, `for_each`, and `run_job` tasks need
no compute. The dispatchers' `dbutils.notebook.run` children execute on the **same** cluster,
so no serverless is required anywhere in the flow.

Inspect results:

```sql
SELECT * FROM kk_test.eval_engine_demo.evaluation_results ORDER BY source, study_id;
SELECT * FROM kk_test.eval_engine_demo.evaluation_insights ORDER BY study_id;
```

Deploy to another environment by switching the target (`-t test` / `-t prod`); only the
`catalog`/`schema` variables change.

### Scale knobs to show in the demo
- **Add a study:** insert a row into `study_config`, re-run `eval_engine_after`. The DAG is
  unchanged; `for_each` just iterates one more time.
- **Add a matching method (e.g. `psm`):** add a branch in `src/notebooks/matching.py`. No
  job edits.
- **Add a model (e.g. `synthetic_control`):** add a branch in `src/notebooks/modeling.py`.
  No job edits.
- **Parallelism:** `for_each_task.concurrency` in `resources/eval_engine_after.job.yml`.

---

## File / folder structure

```
hcsc-eval-engine-dabs/
├── databricks.yml                       # bundle config + dev/test/prod targets + variables
├── resources/
│   ├── setup.job.yml                    # config + synthetic data job
│   ├── run_single_study.job.yml         # per-study pipeline (the for_each unit)
│   ├── eval_engine_after.job.yml        # ✅ config + for_each (scalable, in-notebook dispatch)
│   ├── eval_engine_before.job.yml       # ⚠️ if/else explosion (current state)
│   ├── modular.job.yml                  # ✅ config-driven branch: standard loop + 1 branch per LME model
│   └── modular_cluster.job.yml          # same branches, bound to an existing cluster (no serverless)
└── src/
    ├── setup/
    │   ├── 00_create_config_and_data.py # study_config + synthetic cohorts
    │   └── 01_create_modular_config.py  # study_config_modular + single/multi-period data
    └── notebooks/
        ├── load_config.py               # emits studies[] task value for for_each
        ├── feature_engineering.py       # dispatcher + quality gate (emits quality_passed)
        ├── matching.py                  # dispatcher: exact | knn | ipw
        ├── modeling.py                  # dispatcher: att | ate | did | gee | mixed (+ bootstrap CI)
        ├── insights.py                  # plain-language interpretation
        ├── registry/                    # MODULAR track: separate, team-owned notebooks
        │   ├── load_config_branched.py  # splits studies by model_family; drives the branch
        │   ├── dispatch_feature.py      # (standard) runs the config-named feature leaf, sets quality_passed
        │   ├── dispatch_matching.py     # (standard) runs the config-named matching leaf
        │   ├── dispatch_model.py        # (standard) runs the config-named model leaf
        │   ├── insights_modular.py      # plain-language interpretation (reads study_config_modular)
        │   ├── feature_standard.py      # standard leaf: single baseline + performance period
        │   ├── matching_standard.py     # standard leaf: exact | knn | ipw
        │   ├── matching_stratified.py   # standard leaf: a "third team" plug-in matching method
        │   ├── model_att.py             # standard leaf (ATT team)
        │   ├── model_did.py             # standard leaf (DID team)
        │   ├── feature_lme_mixed.py     # lme_mixed branch: own feature notebook
        │   ├── matching_lme_mixed.py    # lme_mixed branch: own matching notebook
        │   ├── model_lme_mixed.py       # lme_mixed branch: mixed-effects estimator
        │   ├── feature_lme_growth.py    # lme_growth branch: own feature notebook
        │   ├── matching_lme_growth.py   # lme_growth branch: own matching notebook
        │   └── model_lme_growth.py      # lme_growth branch: growth-curve (slope) estimator
        └── _legacy/                     # notebooks used by the "before" job
            ├── prep_and_route.py        # emits chosen_matching / chosen_model / quality_passed
            ├── exact_matching.py
            ├── knn_matching.py
            └── run_model.py             # shared by all 4 model tasks (att/did)
```

---

## Configuration / variables

| Variable | Where | Default (dev) | Notes |
|----------|-------|---------------|-------|
| `catalog` | `databricks.yml` `variables` / per target | `kk_test` | UC catalog for all assets |
| `schema`  | `databricks.yml` `variables` / per target | `eval_engine_demo` (dev), `_test`, `_prod` | per-environment schema |
| job params | each `*.job.yml` | passed to notebook widgets | `study_id`, `feature_method`, `matching_method`, `model_method` |
| `for_each_task.concurrency` | `eval_engine_after.job.yml` | `5` | parallel studies |

Notebooks never hardcode catalog/schema — all read via `dbutils.widgets`.

---

## Notes / caveats

- The estimators here (ATT/ATE/DiD/GEE/mixed) are intentionally lightweight (numpy/pandas,
  dependency-free) so the demo runs anywhere on serverless. They illustrate the
  orchestration pattern, not production-grade causal inference — in production these stage
  notebooks would call the team's real models. DiD studies (STUDY_002, STUDY_005) best show
  the estimate diverging from the naive number as confounding is removed.
- Built and validated against the `kk_test` workspace, June 2026.
