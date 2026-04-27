# SEAL/RCL Expert Diagnosis KB v6 Update Report

## Purpose

This update shifts the diagnosis-only SEAL/RCL assets from general case generation toward **infectious diseases expert formation**.

The first loop remains deliberately minimal:

1. Expert-lane weighted disease sampling.
2. Patient vignette generation.
3. Optional binary patient QC.
4. Diagnosis-only doctor answer.
5. Binary hidden judge.
6. One same-patient retry using a provisional reflection.
7. Save only successful cases and validated reflections.
8. Discard unrecovered failed patients and unvalidated reflections.

## Key design choice

This KB does **not** model general-population prevalence. Fields such as `general_population` remain useful host-context metadata, but they should not drive sampling.

## Files generated

- `infectious_diseases_seal_expert_rcl_kb_v6_compact.json`
- `infectious_diseases_seal_expert_rcl_conditions_v6.jsonl`
- `expert_curriculum_sampler_config_v1.json`
- `minimal_expert_rcl_codex_spec_v2.txt`
- `seal_expert_rcl_kb_update_report.md`

## Condition count

- Conditions retained: 716

## Expert lane counts

```json
{
  "core_syndrome_reasoning": 708,
  "common_infection_with_expert_trap": 373,
  "lookalike_differential_discrimination": 716,
  "travel_tropical_vector_zoonotic": 273,
  "sti_blood_borne_reproductive": 389,
  "high_consequence_must_not_miss": 327,
  "immunocompromised_or_healthcare_context": 177,
  "paediatric_perinatal_pregnancy": 96,
  "rare_tail_expert_recognition": 35
}
```

## Expert value counts

```json
{
  "medium": 198,
  "high": 518
}
```

## Sampling priority counts

```json
{
  "core": 197,
  "targeted": 180,
  "must_not_miss": 327,
  "rare_tail": 12
}
```

## Runtime recommendation

Use `infectious_diseases_seal_expert_rcl_kb_v6_compact.json` as the main runtime KB.

Use `expert_curriculum_sampler_config_v1.json` to keep the first smoke-test sampler simple and expert-oriented.

## Notes

- No treatment, IPC, AMS, source-control, or public-health tasks were added to the first loop.
- No drug doses or patient-specific clinical guidance were added.
- The new expert profiles are curriculum scaffolding, not clinical guidelines or epidemiological prevalence estimates.
