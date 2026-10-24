# Research audit response

This change addresses the supplied `AUDIT.md` against main. It fixes computational failures and prevents unsupported inference. It does **not** create independent human reviews, verified calendar materials, participant data, or independent scenario authorship. The shipped study remains an exploratory measurement of sensitivity to particular prompt manipulations.

## Disposition of all 18 findings

| Audit item | Implemented change | Remaining empirical work |
|---|---|---|
| 1. Bootstrap inference | Removed ordinary-bootstrap tail p-values. Default outputs contain descriptive estimates without population intervals or significance. Opt-in sign flips require a prospective binding and explicit sign-symmetry assumption; suppress below 12 independent scenarios. | Justify the actual sampling/null model and choose adequate precision/power; 12 is an eligibility policy, not proof of sufficiency. |
| 2. Absent condition bounds | Include every accepted pair with either condition observed. Count absent records separately from recorded nonvalid answers. Regression tests cover widened bounds. | Entirely absent model pairs remain outside these bounds; use roster coverage and report this limitation. |
| 3. Cross-design matching | Require economics and material hashes to match. Recheck actual prompt hashes. Optional original roster checks every record against task metadata and input hashes; mandatory for inference. | A common hash does not establish bilingual equivalence. |
| 4. Human assignments | Namespaced study IDs, assignment CSV/content hashes, manifest-bound import and analysis, subject/task/position/arm/language checks and assigned → submitted → valid → analysed counts. | Maintain concealment and authentic administration records. |
| 5. Human uncertainty/attrition | Suppress human/subgroup/model-minus-human population intervals and p-values. Assignment-aware validity differences and worst-case bounds include subjects who never submitted. | A separately justified participant-population inferential model is still needed. |
| 6. Predictable alternation | Replace strict alternation with randomized permuted blocks within language; retain block/session IDs and timestamps. | Recruiters must not have access to future assignments or the seed. Fixed small blocks do not eliminate all predictability near block ends. |
| 7. Holdout protection | Behavioral analysis defaults to test, includes split in every effect stratum. Steering grids default to dev and omit test probe scores. Test behavior requires one hash-bound frozen dev selection. | Local files cannot prove a holdout was never inspected; externally register the selection and prohibit repeated test reuse administratively. |
| 8. Control validity | Existing contrasts are explicitly robustness checks with norm, occasion, length and economic-wording confounds. No unique auspiciousness identification claim. | Independently author multiple syntax/length-matched control families and orthogonal festival/norm/assertion manipulations. |
| 9. Language materials | Machine-readable report limitations and documentation retain AI-draft/unreviewed status. No invented approval. | Independent bilingual financial-domain review, independent variants and comprehension/manipulation checks. |
| 10. Dates/avoidance | Reports explicitly exclude unprompted date recognition and general avoidance-valence interpretations. | Verify dates/timezones and neutral dates, run recognition checks, sample additional avoidance occasions. |
| 11. Coverage/weighting | Count scenarios, retain split, equally weight templates before subjects/scenarios, and require complete planned evaluation outcomes for inference. Add reproducible null/power simulation. | Expand independent scenario authorship, define the situation population, and simulate the actual crossed design, attrition and multiplicity. |
| 12. Outcome interpretation | Report separates recommendation sensitivity, factual compliance, constraint compliance and hypothetical valuation; no belief/irrationality composite. | Validate preferences/WTP, incentive mechanisms and rationale coding if those become research aims. |
| 13. Human comparability | Optional eligible-language roster, coverage preflight, sequence/timestamp diagnostics and study-bound covariate/instrument provenance. Mark assignments as draft, not field ready. Subgroups are explicitly surviving-mixture descriptives. | Pilot recruitment, masking, suspicion, comprehension and prespecified pair/scenario/participant weighting. |
| 14. Probe claims | Describe condition decodability independent of whether probe beats a ceiling text classifier. Add cross-cue transfer; random directions are not lexical-shortcut tests. | Independently authored lexically disjoint cues and polarity counterfactuals. |
| 15. Steering dose/controls | Grid doses have unit norm in diagonal-standardized train activation space; random directions use the same scale. Random controls default on. `controls_complete` is separate from computation completion, and freezing requires random/capability controls. | Full covariance/output-disruption matching and broader specificity evidence. Raw coefficients remain an explicitly selectable exploratory compatibility mode. |
| 16. Probe/capability scope | Train-scenario refits and evaluation-scenario leave-one-out ranges; independently dev-tuned text penalty; distinguish unique authored items from translations; paired behavior/validity/correctness transitions against baseline and random. | Expand independently validated capability items. No general reasoning preservation or mediation claim. |
| 17. Prospective plan | Bind/archival-copy plan hash before calls; inference verifies binding on every record. Confirmatory protocol needs external timestamp/verifier attestation, target population, test split, weighting, and one fully specified primary identity per entry with `expected_tests=1`. No MME/equivalence flags from unadjusted CIs. | Software cannot authenticate a registration service or verify that the plan was unaltered before upload. Register the exact bytes externally and audit timestamps independently. |
| 18. Reproducibility | Collection fingerprints include source-file hashes, git revision, Python/platform and dependencies. Record separately resolved model/tokenizer commits. Fail confirmatory collection when unresolved. Recheck embedded trial-content/prompt hashes on probe load. Bind study/instrument provenance for covariates. | Independently attest remote hosted checkpoint provenance where the provider does not expose immutable revisions. |

## Inferential contract

The default is `descriptive`. Changing `--bootstrap` cannot create a p-value or population CI. Human and subgroup estimates remain descriptive under every supported inference mode.

`scenario_sign_flip` tests scenario means under joint invariance to independent sign changes. This is an explicit model assumption, **not** a treatment randomization implemented by these prompts. It requires `inference_assumption: "independent_sign_symmetric_scenario_effects_under_null"`, approved original materials/roster, immutable provider/model/tokenizer provenance, externally attested registration, collection-time plan hashes, and one valid observation per planned evaluation task. It enumerates all signs through 16 scenarios and otherwise uses 9,999 fixed-seed transformations with `(extreme + 1)/(draws + 1)`. Primary identities use Holm with unavailable tests retained as non-rejections. Population confidence intervals and interval-based MME/equivalence decisions remain unavailable.

The minimum of 12 scenarios does not make the shipped four-scenario test split inferentially adequate. Do not pool train/dev/test to evade that restriction. Even a test passing every software gate remains conditional on its unverified scientific assumptions.

Permutation validity depends on the transformation model; see [Hemerik and Goeman, Exact testing with random permutations](https://doi.org/10.1007/s11749-017-0571-1). Add-one Monte Carlo counting follows [Phipson and Smyth, Permutation P-values should never be zero](https://pubmed.ncbi.nlm.nih.gov/21044043/).

## Reproducible calibration

```sh
python scripts/calibrate_inference.py --out docs/inference-calibration.json
```

The checked-in JSON uses 1,000 independent Normal scenario-effect experiments per cell, seed 20261024. With 12 scenarios and zero mean, the raw test rejects 55/1,000 (5.5%; Monte Carlo Wilson interval 4.25–7.09%). A mean of 0.5 scenario SD gives 333/1,000 rejections (33.3% power), illustrating why passing a minimum cluster rule is insufficient. Two/four-scenario raw tests cannot reach 5% because of exact-test discreteness, and the analysis pipeline suppresses their p-values anyway. The `[1, 2]` diagnostic has raw exact p=0.5 regardless of requested bootstrap draws; the pipeline reports no p-value at that size.

These diagnostics are synthetic software checks under the stated model. They do not validate symmetry, independence, power or the situation population of the authored investment corpus. They also do not simulate participant sampling, selective invalidity, dependence among outcomes or multiplicity in the actual design.

## Data and compatibility

- Old result files remain usable descriptively, with unverifiable collection provenance reported. They cannot be upgraded to prospective data by attaching a plan afterward.
- Human import now requires `--assignments study/design.json`. `--allow-unassigned` is an explicit descriptive-only legacy path with unknown assignment/nonresponse denominators.
- Assigned-study analysis needs `--trials` and `--assignments`. Study-bound participant imports additionally document an instrument (id/version/source/scoring); its validity is still a research responsibility.
- Assignments contain participant and design metadata; distribute only the per-subject survey files. Preserve assignment files privately and record `submitted_at` with timezone for actual order diagnostics.
- A declared language-eligible roster is optional for drafting; absence of a roster does not certify eligibility and `field_ready` remains false. Coverage preflight exposes tasks missing an assigned arm.
- Effects average observed replicates within template, templates within subject/scenario, subjects within scenario, then scenarios equally. Descriptive survivor weights can still change when a template has no valid pair; inference therefore requires complete planned outcomes. Human assignment bounds target assigned-subject arm means per task; they are not population intervals.
- Source/runtime changes invalidate cached collections. Continue old runs using their original code/environment; start a new directory for changed software/protocols.
- Standalone local steering supports legacy raw-coefficient dosing. The grid and CLI default to calibrated activation-SD displacement. This is diagonal scaling, not full covariance whitening or equal downstream disruption.
- A grid's `complete` means computation finished. `controls_complete` records whether random controls and capability checks exist. Neither field certifies translation/control validity, general mechanism specificity, mediation or a preregistration.

## Before stronger claims

Archive independent authorship/review records, validated controls and dates, comprehension and masking pilots, recruitment/eligibility evidence, an expanded independent scenario/capability sample, and a design-specific precision/power analysis. Define the situation/participant populations, exact estimands and weights, fixed hypothesis identities, stopping/exclusion rules, holdout-use policy and external registration. No item in that list has been represented as completed by this code change.
