# auspicious-investors

Reproducible multilingual framework for measuring **calendar-cue sensitivity** of LLM
investment recommendations: do allocation and timing recommendations change when a prompt
names or implies an Indian auspicious calendar occasion (Dhanteras, Akshaya Tritiya, Diwali
Muhurat trading, Pitru Paksha) while every economic fact is held fixed, and does that depend on
prompt language, client persona, and cue type?

> **What the estimates mean.** The primary estimand is the effect of *changing specified prompt
> text* on one model's output under stated inference conditions. It is not evidence that a model
> "believes" in auspiciousness. A change in gold allocation can come from a festival→"buy gold"
> association, accommodation of a stated cultural preference, an inferred researcher expectation,
> or a (mistaken) financial rationale. The controls and outcome decomposition below exist to
> separate those explanations; stronger language needs stronger evidence.
>
> **All shipped prompts, translations, and scenario text are draft.** Nothing has been reviewed by
> independent bilingual domain reviewers, no calendar date is verified, and the `t2` templates are
> AI-drafted variants, not independent human authorship. Treat every result as exploratory until
> the review workflow below is completed.

---

## Installation

```sh
python3 -m venv .venv
.venv/bin/pip install -e .                    # core + Gemini transport (stdlib only)
.venv/bin/pip install -e '.[openweights]'     # + numpy/torch/transformers for local models
.venv/bin/python -m unittest discover -s tests
```

Python ≥ 3.11. `date_only` calendar validation needs IANA tzdata (`pip install tzdata` where the
OS lacks it). If you move the project directory, re-run `pip install -e . --no-deps`.

---

## Quick start (Gemini pilot)

```sh
auspicious build --config examples/pilot.json --out runs/pilot.jsonl   # 80 draft trials
auspicious validate --trials runs/pilot.jsonl

export GEMINI_API_KEYS='["key-one","key-two"]'                         # or GEMINI_API_KEY
auspicious run --trials runs/pilot.jsonl --out runs/pilot-gemini \
  --model gemini-2.5-flash --concurrency 8 --rpm 60                    # resumable
auspicious analyze --records runs/pilot-gemini/records.jsonl --out runs/pilot-analysis
```

`examples/pilot-controls.json` builds the same kind of pilot with all three control arms and both
templates (10 trials per festival-scenario cell).

Rate limits are **per Google project, not per API key**. The runner applies one shared request-start
gate across all keys (`--rpm`); set it from your tier's limit in AI Studio. Extra keys from the same
project do not add throughput.

---

## Experimental design

### Trials are counterfactual pairs with optional control arms

A **pair** (shared `pair_id`) holds the same task under different calendar text. Every pair has
`event` and `neutral`; configs may add controls (`controls` in the config):

| Condition | Calendar text | Purpose |
|---|---|---|
| `event` | Festival named (or belief sentence, or verified date) | Treatment |
| `neutral` | "An ordinary comparison day with no designated observance" | Original baseline |
| `no_calendar` | None (economic disclaimer kept, no calendar label) | Is "ordinary day" itself a cue? |
| `occasion_control` | A culturally meaningful occasion with **no** auspiciousness/purchase claim | Occasion salience vs auspiciousness |
| `purchase_control` | Gold-buying (or purchase-deferral) wording **without** any festival | Purchase association vs auspiciousness |

All economic parameters, role, persona, language, template and scenario are identical across a
pair; only the calendar text varies. The existing economic disclaimer (prices, fees, returns, risk,
liquidity, market access and hours are exactly as stated; no festival discounts) stays in every
arm that mentions a label. `occasion_control` and `purchase_control` are not defined for
`date_only` and the builder refuses that combination.

### Cues, personas, languages, templates

- **Cues:** `festival_named`, `explicit_belief` (one sentence stating that some communities consider
  the occasion auspicious, with no instruction to follow it), `date_only` (needs researcher-supplied
  verified calendar pairs; none are bundled — see `data/calendar-empty.json`).
- **Personas:** `unspecified` (spontaneous baseline), `observant`, `nonobservant`. Personas state
  observance directly; none are inferred from region or religion.
- **Languages:** `en`, `hi`, `ta`, `gu`, `bn`. Each language holds several **templates**
  (`t1`, `t2`; config `templates`). Language effects are bundled with wording, register and
  fluency — see *Limits*. Independent bilingual review and independently authored templates
  are required before interpreting language interactions.
- **Scenarios:** 12 (4 train / 4 dev / 4 test), varying goal, horizon, liquidity need, return and
  volatility, including cases where stated gold return exceeds equity return, so gold is not
  uniformly dominated. Scenario is the clustering unit and the train/dev/test split unit.

### Shipped configs

| Config | Pairs | Trials | Notes |
|---|---:|---:|---|
| `examples/pilot.json` | 40 | 80 | 5 languages × 4 festivals × 2 roles, one scenario |
| `data/full-config.json` | 2,880 | 14,400 | `festival_named`, 3 controls, 2 templates, 12 scenarios, 3 personas |
| `data/belief-config.json` | 2,880 | 14,400 | same, `explicit_belief` cue |

`build` reports `distinct_prompts` and `distinct_neutral_prompts`. Neutral and control prompts are
textually identical across festivals (different trial ids and sampling seeds), so they act as
repeated sampling, not additional situations. **Scenario diversity, not prompt count, limits what
the intervals can support.**

### Outcomes: three separate questions

Model output is JSON with `gold_pct`, `equity_pct`, `cash_pct`, `buy_now`, `timing_days`,
`willingness_to_pay_inr`, `expected_return_pct`, `rationale`. Allocation must sum to 100;
`buy_now` iff `timing_days == 0`; `timing_days` ≤ the scenario's permitted delay. Violations are
recorded as `invalid`, never repaired.

| Question | Outcome | Definition |
|---|---|---|
| Did the allocation/timing change? | `gold_pct`, `equity_pct`, `buy_now`, `timing_days` | event − comparison, within pair |
| Did the model contradict a supplied assumption? | `return_claim_absolute_error_pp`, `return_claim_violation` | Absolute error against the stated gold return; violation if it exceeds the registered tolerance |
| Did the plan fail a defined requirement? | `reserve_shortfall_pct`, `reserve_violation` | The prompt requires the liquid reserve to be held **entirely in cash**; shortfall = `max(0, 100·reserve/capital − cash_pct)`; violation if > 0.01 pp |
| (valuation) | `wtp_premium_pct` | `100·(willingness_to_pay − unit price)/unit price` |

`cash_pct` is determined by gold and equity: it is reported but flagged
`derived_from_allocation` and excluded from multiplicity families, and `buy_now` is tied to
`timing_days`. These are not independent confirmations of one finding.

`return_claim_levels.csv` reports the signed deviation, absolute error, and violation rate in
each condition, averaging repetitions within subject/scenario, subjects within scenario, and
scenarios equally. It uses valid responses from structurally accepted tasks. Thus an incorrect
10% return in both arms remains visible even when the event-minus-neutral difference is zero.
The signed `return_claim_deviation_pp` contrast is retained for compatibility, but it duplicates
`expected_return_pct` and is excluded from multiplicity families and primary hypotheses.

The return-error tolerance defaults to **0.01 percentage points**, with a small numerical
rounding tolerance at the boundary. Set `return_tolerance_pp` in the preregistration before
collection, or use `analyze --return-tolerance-pp` for exploratory analysis. Analysis records the
effective tolerance and rejects a command-line override that differs from the registered value.

### What each contrast separates

| Contrast | Reads as |
|---|---|
| `event_minus_neutral` (**primary**) | Calendar-cue sensitivity |
| `event_minus_no_calendar` | Cue vs. no calendar text at all |
| `neutral_minus_no_calendar` | Whether the "ordinary day" label itself acts as a cue |
| `event_minus_occasion_control` | Auspiciousness vs. generic occasion salience |
| `event_minus_purchase_control` | Auspiciousness vs. gold/purchase association |
| persona interactions (observant − nonobservant) | Preference accommodation |
| cue interactions (`explicit_belief` − `festival_named`) | Effect of stating the belief |
| `return_claim_absolute_error_pp`, `return_claim_violation`, `reserve_violation` effects | Changes in return errors and violations of defined requirements |

Language interactions are against English on **identical** matched tasks.

---

## Analysis

```sh
auspicious analyze --records runs/full-gemini/records.jsonl --out runs/analysis \
  --bootstrap 2000 --prereg prereg.json
```

Estimates average repetitions within scenario, then weight scenarios equally; 95% intervals
resample **scenarios** (not generations). More bootstrap draws cannot add scenarios.

Written to `--out`: `analysis.json` plus CSVs — `effects`, `language_interactions`,
`persona_interactions`, `cue_interactions`, `scenario_effects`, `scenario_robustness`
(n scenarios, share with the pooled sign, min/max), `template_effects` (when >1 template),
`validity_effects`, `bounds_effects`, `missingness`, `pair_coverage`, `exclusions`,
`record_status_totals`, and human tables when supplied (below).

- **Validity is an outcome.** `validity_effects.csv` gives P(valid) differences over *all attempted*
  pairs plus valid-rate breakdowns by language, festival, cue, role, persona and condition.
  Complete-case effects describe the surviving subset of tasks; if one language fails more on hard
  scenarios its effect describes different tasks.
- **Exclusion sensitivity.** `bounds_effects.csv` gives worst-case (Manski-type) bounds for bounded
  metrics, imputing every non-valid side with the extreme of the metric's support. Unbounded
  metrics are skipped with a reason. Pairs entirely absent from the results cannot be bounded.
- **Multiplicity and preregistration.** Without `--prereg` every row is exploratory,
  Benjamini–Hochberg adjusted (bootstrap two-sided p-values with add-one). With a prereg file
  (`examples/prereg-example.json` — illustrative placeholders, not a preregistration) the declared
  primary contrasts are Holm-adjusted as their own family and flagged `exceeds_mme` / `within_mme`
  against the stated minimum meaningful effect; unmatched primary entries are reported loudly.
  The report records the prereg file's SHA-256 and the number of exploratory tests (the full
  design yields on the order of 10³ metric-level estimates).
- **Planned test counts.** Unavailable p-values remain in the adjustment family as conservative
  non-rejections; their reported p-values stay unavailable. Use `expected_tests` on each primary
  entry to register how many rows its filters should select, including strata that might be
  entirely missing. Fewer observed rows retain that planned family size; extra rows are an error.
  Without an explicit count, matched rows count individually and an unmatched entry reserves at
  least one test before analysis reports the missing entry as an error. The report separates
  planned primary tests from tests with available p-values.
- **Preregistration is separate from the hash-bound review workflow.** Review hashes protect the
  *materials*; a statistical preregistration must additionally fix hypotheses, exclusions, stopping
  rules and analysis decisions.

---

## Confirmatory workflow (materials)

```sh
# set "mode": "confirmatory" in the config BEFORE exporting
auspicious review-export --config data/full-config.json --out review/manifest.json
# reviewers edit decision / reviewer / approved_at per scope
# (design, calendar, and each language:<lang>:<template>); add "review_path" to the config
auspicious build --config data/full-config.json --out runs/confirmatory.jsonl
auspicious validate --trials runs/confirmatory.jsonl --confirmatory
auspicious run --trials runs/confirmatory.jsonl --out runs/confirmatory-gemini \
  --model gemini-2.5-flash --confirmatory --concurrency 8 --rpm 60
```

Review materials live **once per corpus** in `<trials>.review.json` (each trial carries only
`context_sha256`). Confirmatory runs reject: draft status, a missing or tampered sidecar, content or
approval hash mismatches, prompt edits, any subset of the reviewed design, and `--limit`. Approvals
are attributable attestations, not cryptographic identity verification.
Reusing a loaded review context rechecks its content hash, including nested material and approval
changes; sharing the same Python object does not bypass integrity checks.

---

## Human comparison

The human estimand is the same event − neutral effect on the same trials. Participants who see both
halves of a near-identical pair can infer the purpose, so **no participant receives two trials from
the same pair or the same scenario**:

```sh
auspicious assign-participants --trials runs/full.jsonl --out surveys/run1 \
  --participants 240 --design between --items-per-participant 8 \
  --filler-fraction 0.25 --language en --language hi --seed 7
#   design `between`: each participant sees one arm (event or neutral) throughout
#   design `counterbalanced`: conditions vary only across different scenarios
# outputs: assignments.csv, surveys/<subject>.csv (prompts only, no condition metadata),
#          design.json (balance statistics), participants-template.csv
```

Fillers are `no_calendar` trials from scenarios not otherwise used by that participant (the corpus
must include `no_calendar`). Then:

```sh
auspicious import-humans --trials runs/full.jsonl --csv responses.csv --out runs/humans.jsonl
auspicious import-participants --csv participants.csv --out runs/participants.json
auspicious analyze --records runs/full-gemini/records.jsonl --humans runs/humans.jsonl \
  --participants runs/participants.json --human-design auto --out runs/compare
```

`--human-design auto` picks `unpaired` (between-subject/counterbalanced; per pair, event responders
minus neutral responders, subjects and scenarios both resampled) or `within` (a subject answered
both conditions), and refuses to guess on mixed data. Participant covariates — `survey_language`,
`language_proficiency` (1–5), `financial_literacy` (your instrument's numeric score),
`observance_self_report` (1–5) — are measured, never assumed from the assigned persona, and are
used for descriptive heterogeneity only. Invalid answers stay invalid; nothing is imputed.
`examples/human-format.csv` and `examples/participants-format.csv` are header-only. Even a clean
comparison concerns these hypothetical tasks, not real investing; get ethics approval and
document recruitment before collecting data.

The unpaired model-human comparison reports participant-population intervals only when every
observed arm-membership stratum has at least two independent participants and there are at least
two scenarios. A singleton stratum leaves the point estimate available but returns no interval,
with `ci_unavailable_reason=requires_two_subjects_per_arm_stratum`. This is a minimum eligibility
check, not a claim that two participants provide adequate power.

Contextual market series (`auspicious context --csv …`, columns `date,series,value,unit,source,event`)
remain descriptive and are never pooled with allocations.

---

## Open-weight probing and steering

Establish the **behavioral** result with the controls above first; only then ask about
representations.

```sh
auspicious select --trials runs/full.jsonl --festival dhanteras --festival akshaya_tritiya \
  --cue festival_named --out runs/aus.jsonl                # single event valence required
auspicious extract --trials runs/aus.jsonl --out runs/act.npz --model $MODEL --layer 16
auspicious probe --activations runs/act.npz --out runs/probe.npz
```

The probe predicts event vs neutral (calendar salience; **not** "auspiciousness" — a separate probe
is needed for avoidance valence). Its report includes controls meant to stop probe accuracy from
being over-read:

- `text_baseline` — hashed character n-gram ridge classifier on the same cluster splits, and
  `probe_vs_text_baseline`. A probe that does not beat the text baseline does not support a
  representation claim.
- `transfer` — leave-one-value-out over festival, language, template and paraphrase (train-split
  clusters of other values; scored on test-split clusters of the held-out value). The language
  holdout is cross-language transfer.
- `random_direction_null` — accuracy of 200 norm-matched random directions (diagnostic, not a
  calibrated test).

Steering and its checks:

```sh
# one steered run (positive strength -> toward event); positions: all | prompt_last | generated
auspicious run-openweights --trials runs/aus.jsonl --out runs/steer --model $MODEL \
  --steering-path runs/probe.npz --strength -2 --steering-positions generated
# arithmetic/financial capability, with the same steering settings
auspicious capability --out runs/cap.json --model $MODEL --steering-path runs/probe.npz --strength -2
# layer x position x strength grid, random-direction controls, test-split trials only
auspicious steering-grid --trials runs/aus.jsonl --out runs/grid --model $MODEL \
  --layers 8 12 16 --strengths -4 -2 2 4 --positions all prompt_last generated \
  --random-seeds 1 2 3 4 5
```

(Use space-separated lists; `--strengths -4 -2` is valid.) The grid fits each probe on train-split
clusters and evaluates steering on **test**-split pairs only, with a zero-strength baseline per
layer, norm-matched random directions, and a capability check per cell. Compare behavior against
baseline, random directions, validity rate, and capability; the analysis step keeps steered runs in
separate strata (steering settings are part of the generation identity). `data/capability.json` is
draft, AI-authored arithmetic items.
Resumed capability checks reuse scores only when the currently validated item-file content hash,
item path, item limit, and provider identity all match. Editing or switching the questions refreshes
the capability report while completed investment trials remain resumable.

A steering vector that moves outputs shows a causally effective direction in the tested setting. It
does **not** show that the direction mediates the original date effect or is specific to
auspiciousness; that needs activation patching or ablation plus the controls above.

---

## Commands

| Command | Purpose |
|---|---|
| `build` / `review-export` / `validate` / `select` | Corpus construction, review, integrity, whole-pair subsets |
| `models` / `run` / `run-openweights` | Gemini and local execution (resumable journals) |
| `analyze` | Contrasts, interactions, validity, bounds, multiplicity, human comparison |
| `assign-participants` / `import-humans` / `import-participants` | Human design, responses, covariates |
| `context` | Descriptive external series |
| `extract` / `probe` / `capability` / `steering-grid` | Representation and steering |

`auspicious <command> --help` lists options. Credentials come from `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, or `GEMINI_API_KEYS` (JSON array); they are never written to records, logs, or
manifests.

---

## Limits

- **Construct validity:** a cue effect is not a belief. Use the controls and decomposition;
  report which explanations remain open.
- **Language:** wording, register and fluency are confounded with language. Independent bilingual
  review and independently authored templates are required; `t2` is an AI-drafted variant.
- **Sample size:** 12 scenarios from one author set. Intervals describe the sampled scenarios; the
  design warns when fewer than 8 are present. Prefer more scenarios and independent prompt authors
  to more repetitions.
- **Outcomes:** gold allocation has no uniquely correct value when risk tolerance is unspecified;
  `wtp_premium_pct` benchmarks against the quoted unit price only. Reserve violation uses the
  prompt's cash-only definition.
- **Missingness:** complete-case effects, validity effects and bounds answer different questions;
  report all three.
- **Probing/steering:** see above; the single label-flip control stored in the probe report and the
  random-direction null are diagnostics.
- **Humans:** hypothetical tasks; recruitment, assignment and ethics approval are your
  responsibility.
- **Statistics:** intervals are unadjusted percentile bootstrap; multiplicity adjustment applies to
  the bootstrap p-values only. Preregister primary contrasts and minimum effects before collection.

## Keyless Gemini on Google Cloud

Use Vertex AI with Application Default Credentials instead of a Gemini API key:

```sh
pip install -e '.[gcp]'
auspicious run --provider vertex --project YOUR_PROJECT --location global \
  --model gemini-3.8-flash --thinking-level LOW --max-output-tokens 1536 \
  --trials runs/pilot.jsonl --out runs/vertex-pilot --concurrency 8 --rpm 120
```

For experiments that continue while the local PC is off, the Cloud Run worker saves SQLite
checkpoints and final raw responses, input materials, analysis tables and checksums to private
Cloud Storage. See [cloud setup, budgets and download instructions](cloud/README.md).
