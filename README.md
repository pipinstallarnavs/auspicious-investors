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
> or a (mistaken) financial rationale. The controls are robustness checks; their wording and norm confounds leave these explanations open.
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
auspicious analyze --records runs/pilot-gemini/records.jsonl --trials runs/pilot.jsonl \
  --split train --out runs/pilot-analysis                            # exploratory train-only pilot
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
| `occasion_control` | A culturally meaningful occasion with **no** auspiciousness/purchase claim | Occasion-wording robustness comparison |
| `purchase_control` | Gold-buying (or purchase-deferral) wording **without** any festival | Purchase-norm robustness comparison |

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
repeated sampling, not additional situations. **Scenario diversity, not prompt count, limits generalization.**

### Outcomes: separate questions

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

### Contrasts and robustness checks

| Contrast | Reads as |
|---|---|
| `event_minus_neutral` (**primary**) | Calendar-cue sensitivity |
| `event_minus_no_calendar` | Cue vs. no calendar text at all |
| `neutral_minus_no_calendar` | Whether the "ordinary day" label itself acts as a cue |
| `event_minus_occasion_control` | Event vs. civic-occasion wording (confounded robustness check) |
| `event_minus_purchase_control` | Event vs. purchase-norm wording (confounded robustness check) |
| persona interactions (observant − nonobservant) | Persona-dependent sensitivity, possibly preference accommodation |
| cue interactions (`explicit_belief` − `festival_named`) | Effect of stating the belief |
| `return_claim_absolute_error_pp`, `return_claim_violation`, `reserve_violation` effects | Changes in return errors and violations of defined requirements |

Language interactions are against English on **identical** matched tasks.

---

## Analysis

```sh
auspicious analyze --records runs/full-gemini/records.jsonl --trials runs/full.jsonl --out runs/analysis
# Explicit exploratory strata (kept separate): --split train --split dev --split test
```

Analysis defaults to **test only** and keeps `split` in effect strata. Estimates average
replicates/paraphrases within template, templates equally within subject/scenario, subjects within
scenario, then scenarios equally. Reported effects describe surviving complete pairs; invalid answers
are never replaced with zero. No default population confidence intervals, p-values or MME/equivalence
claims are produced. `--bootstrap` is retained for command compatibility and cannot enable inference.

Outputs include `analysis.json`, effects/interactions, per-scenario/template diagnostics, validity,
worst-case bounds, missingness, pair coverage, exclusions and human tables. Bounds include pairs with
**either** condition recorded; absent records and recorded nonvalid responses have separate counts.
Entirely absent model pairs are outside these bounds. `--trials` revalidates result metadata and input
hashes against the original roster; the report states planned and observed coverage.

Cross-language/persona/cue interactions require equal economics and material hashes. Shared hashes
are integrity checks, not evidence of translation equivalence. Existing occasion/purchase/no-calendar
contrasts are robustness comparisons, with wording, norm, length and economic-interpretation confounds.

Opt-in `--inference scenario_sign_flip` requires a collection-bound externally attested plan, explicit
independent sign symmetry, an approved original roster, immutable model/tokenizer identities, full
valid evaluation data, and at least 12 independent scenarios **per tested stratum**. The test is exact
through 16 scenarios, otherwise add-one Monte Carlo. Human/subgroup estimates stay descriptive. Fixed
primary identities/counts use Holm; missing tests remain in the family. No population CI is claimed.
The shipped four-scenario test split does not meet the threshold, and the threshold is not a power
justification. See [the audit response and inferential contract](docs/AUDIT_RESPONSE.md).

`examples/prereg-example.json` is a descriptive illustration, not proof of preregistration. A
confirmatory plan needs an external HTTPS registration URL, prior timezone-aware `registered_at`,
`timestamp_verified_by`, target population, frozen `evaluation_splits: ["test"]`, and
`template_weighting: "equal_planned_templates_complete_case"`. Each primary entry must register
one exact hypothesis (`expected_tests: 1`, all source/provider/design/split filters). Register the
exact bytes before collection; the software records the attestation and cannot authenticate it.

```sh
python scripts/calibrate_inference.py --out calibration.json
```

This reproducible synthetic null/power check does not validate the authored corpus's assumptions.

## Reviewed materials and prospective collection

Export review scopes, obtain attributable independent review, then build and validate the entire
reviewed design. `review-export` does not supply a review. No shipped materials have independent
approval or verified calendar dates.

```sh
# Set mode=confirmatory BEFORE exporting; reviewers complete genuine approval records.
auspicious review-export --config reviewed-config.json --out review/manifest.json
auspicious build --config reviewed-config.json --out runs/reviewed.jsonl
auspicious validate --trials runs/reviewed.jsonl --confirmatory
# Explicit immutable local model/tokenizer and externally registered plan:
auspicious run-openweights --trials runs/reviewed.jsonl --out runs/prospective \
  --model "$PINNED_LOCAL_MODEL" --confirmatory --prereg registered-plan.json
```

The review sidecar is `<trials>.review.json`. Collection rejects draft materials, tampered review
content, subsets of a reviewed design, limits, an incomplete prospective plan and unresolved model
or tokenizer identity. Hosted models that do not expose immutable identities cannot support this
confirmatory path. Each record binds the plan hash, source-file hashes, git revision and runtime
versions before the response is collected. Changing code/runtime/protocol requires a fresh run
directory; an old exploratory collection cannot become prospective retroactively.

## Human studies

Assignments are drafts until recruitment, language eligibility, comprehension, masking/suspicion,
coverage and the analysis plan have been piloted and independently checked. Existing `no_calendar`
fillers are financial tasks, not validated masking material.

```sh
auspicious assign-participants --trials runs/full.jsonl --out surveys/study1 \
  --study-id study1 --participants 240 --design between --items-per-participant 8 \
  --filler-fraction 0.25 --language en --language hi --seed 7 \
  --participant-roster eligible.csv
```

The optional roster has `subject_id,language,language_eligible,session_id`; every subject must be
eligible in the assigned language. Without it, language assignment is a drafting convenience and
eligibility remains unverified. Between-participant arms use randomized permuted blocks within
language rather than strict alternation. Conceal the master assignments and seed from recruiters.
Counterbalanced participants encounter both framings across distinct scenarios, so demand effects
remain possible. The preflight reports unexposed tasks and missing-arm coverage.

Distribute only `surveys/<study-subject>.csv`. Preserve `design.json` and `assignments.csv` unchanged.
Responses must carry `study_id,subject_id,trial_id,position` plus the answer columns in
`examples/human-format.csv`; include actual `submitted_at` timestamps with timezone to check order.

```sh
auspicious import-humans --trials runs/full.jsonl --csv responses.csv \
  --assignments surveys/study1/design.json --out runs/humans.jsonl
auspicious import-participants --csv participants.csv --out runs/participants.json \
  --assignments surveys/study1/design.json --instrument instrument.json
auspicious analyze --records runs/full-gemini/records.jsonl --humans runs/humans.jsonl \
  --trials runs/full.jsonl --assignments surveys/study1/design.json \
  --participants runs/participants.json --out runs/compare
```

Instrument JSON documents `id,version,source,scoring`; documentation is not validation. Covariates
are measured separately, never inferred from an assigned persona. Subject/language/task/position,
study and file hashes are verified. The report shows assigned ? submitted ? valid ? analysed,
nonresponse, order diagnostics and assignment-based outcome-support bounds. Tasks with no assigned
participant in one arm are flagged. Human effects, model-minus-human effects and subgroup mixtures
are descriptive; they do not receive population intervals or adjusted significance. Legacy import
requires explicit `--allow-unassigned`, with unknown nonresponse denominators.

External series imported with `auspicious context` remain descriptive and are never pooled with
allocations. The repository contains no participant or market observations.

## Open-weight condition probes and steering

```sh
auspicious select --trials runs/full.jsonl --festival dhanteras --festival akshaya_tritiya \
  --cue festival_named --out runs/aus.jsonl
auspicious extract --trials runs/aus.jsonl --out runs/act.npz --model "$MODEL" --layer 16
auspicious probe --activations runs/act.npz --out runs/probe.npz
```

The label is explicitly encoded in prompt text: accuracy means **condition decodability**, not
belief or a general auspiciousness representation. A text classifier can hit a 100% ceiling; beating
it is not a necessary or sufficient representation test. Text ridge is selected independently on
dev, with no test tuning. Reports include train-scenario refits, evaluation-scenario leave-one-out
ranges, cross-festival/language/template/paraphrase/cue transfer and a random-direction diagnostic.
Lexically disjoint independently authored cues and polarity counterfactuals are still required.
Stored trial-content and prompt hashes are rechecked on artifact load.

```sh
# Choose intervention behavior on dev; test scores are omitted from these probe reports.
auspicious steering-grid --trials runs/aus.jsonl --out runs/dev-grid --model "$MODEL" \
  --layers 8 12 16 --strengths -2 2 --positions all prompt_last generated --random-seeds 1 2 3
# Freeze one chosen completed dev cell, before inspecting test behavior.
auspicious freeze-steering --grid runs/dev-grid/grid.json --layer 12 --position prompt_last \
  --strength 2 --out runs/selection.json
# Exactly that setting, same corpus/model/generation/control settings, full test pairs.
auspicious steering-grid --trials runs/aus.jsonl --out runs/test-evaluation --model "$MODEL" \
  --layers 12 --strengths 2 --positions prompt_last --random-seeds 1 2 3 \
  --evaluation-split test --frozen-selection runs/selection.json
```

The local freeze binds dev artifacts, source/runtime, corpus and generation settings. It is an
integrity aid, not proof that nobody has inspected/reused test data; externally register it for
stronger claims. Reusing the same test data after selecting another cell remains exploratory.

Grid/CLI dose defaults to unit norm in diagonal-standardized **train** activation coordinates.
Strength is the size of that displacement; random directions match it. This is not full covariance
whitening or equal downstream disruption. Direct provider calls retain legacy raw-coefficient dose
for compatibility; CLI users must explicitly choose `--steering-dose raw_coefficient` for that mode.

Random controls and capability checks default on; `controls_complete` is separate from computation
`complete`. An uncontrolled grid can be explored but cannot be frozen for test evaluation. Each
cell reports paired validity/behavior changes versus baseline; random cells also compare against
learned cells, with paired answer/correctness transitions on capability items. The draft capability
set contains **15 unique authored problems**, repeated in five languages, not 75 independent items.
Cache reuse requires unchanged source/runtime, materials, item limit and provider settings.

An effective intervention direction does not establish mediation of the original calendar effect,
general reasoning preservation, or specificity to auspiciousness.

## Limits and audit follow-up

The target situation population, independent scenario authorship, bilingual validation, matched
control families, validated dates, additional avoidance occasions, human pilot and expanded
capability set remain research work. More crossed prompts cannot supply independent scenarios.
Recommendation sensitivity, compliance and hypothetical valuation are separate outcomes; no
belief/irrationality composite is defined. Uncoded rationales do not establish reasoning mechanisms.

[Audit response](docs/AUDIT_RESPONSE.md) tracks all 18 supplied findings, implemented safeguards,
remaining empirical work, compatibility changes and reproducible calibration evidence.
