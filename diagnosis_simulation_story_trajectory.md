# Diagnosis Simulation Story Trajectory

This file shows a single illustrative trajectory through the diagnosis simulation:

1. Current hidden patient/condition data and the prompt used to create a patient case.
2. A quality-control example for the generated case.
3. The doctor-agent prompt using the case plus mock successful-case experience and validated reflections.

The example is self-contained and follows the prompt style used by the repository.

## 1. Current Patient Data And Prompt To Create Patient

In the simulation, the model first receives hidden condition metadata. This hidden data is used to generate a patient vignette, but the generated case must not reveal the disease name, pathogen name, or condition ID.

### Hidden Condition Data

```json
{
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "patient_generation": {
    "body_system": [
      "respiratory",
      "gastrointestinal",
      "multisystem_systemic"
    ],
    "time_course_hint": "acute 1 to 7 days",
    "symptoms_to_include": [
      "fever",
      "cough",
      "dyspnea",
      "diarrhea",
      "confusion"
    ],
    "syndrome_tags": [
      "febrile illness",
      "respiratory syndrome",
      "atypical pneumonia"
    ],
    "exposure_clues": [
      "hotel, spa, cooling tower, or building water-system exposure"
    ],
    "host_factor_clues": [
      "older adult",
      "smoking history or chronic lung disease when useful"
    ],
    "objective_or_discriminating_clues": [
      "pneumonia with gastrointestinal or neurologic features",
      "hyponatremia",
      "mild transaminitis",
      "negative routine respiratory viral testing",
      "water-system exposure separates this from routine community-acquired pneumonia"
    ],
    "minimum_case_requirements": [
      "age and sex",
      "chief complaint",
      "symptom timeline",
      "at least two condition-consistent symptoms",
      "one discriminating exposure, host factor, objective clue, or relevant negative",
      "no direct disease name, pathogen name, or condition ID"
    ]
  }
}
```

### Expert Curriculum Profile

```json
{
  "primary_training_lane": "lookalike_differential_discrimination",
  "curriculum_lanes": [
    "core_syndrome_reasoning",
    "common_infection_with_expert_trap",
    "lookalike_differential_discrimination"
  ],
  "expert_value": "high",
  "must_not_miss": false,
  "key_discriminators": [
    "water-system exposure",
    "pneumonia with diarrhea or confusion",
    "hyponatremia",
    "transaminitis",
    "negative respiratory viral testing"
  ],
  "lookalike_differentials": [
    "Community-acquired pneumonia",
    "Mycoplasma pneumonia",
    "Influenza pneumonia",
    "Aspiration pneumonia"
  ],
  "diagnostic_traps": [
    "calling it routine community-acquired pneumonia without using the water exposure",
    "over-anchoring on a respiratory viral syndrome",
    "ignoring gastrointestinal and neurologic clues in pneumonia"
  ],
  "expert_reason": "Trains discrimination between routine pneumonia and an atypical pneumonia pattern driven by exposure and extrapulmonary clues."
}
```

### Patient-Generation Prompt

```text
You are generating a synthetic patient vignette for a closed infectious disease diagnosis simulation.

The goal is to train infectious diseases expert diagnostic reasoning, not to reflect general-population prevalence.

Hidden condition:
{
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "patient_generation": {
    "body_system": [
      "respiratory",
      "gastrointestinal",
      "multisystem_systemic"
    ],
    "time_course_hint": "acute 1 to 7 days",
    "symptoms_to_include": [
      "fever",
      "cough",
      "dyspnea",
      "diarrhea",
      "confusion"
    ],
    "syndrome_tags": [
      "febrile illness",
      "respiratory syndrome",
      "atypical pneumonia"
    ],
    "exposure_clues": [
      "hotel, spa, cooling tower, or building water-system exposure"
    ],
    "host_factor_clues": [
      "older adult",
      "smoking history or chronic lung disease when useful"
    ],
    "objective_or_discriminating_clues": [
      "pneumonia with gastrointestinal or neurologic features",
      "hyponatremia",
      "mild transaminitis",
      "negative routine respiratory viral testing",
      "water-system exposure separates this from routine community-acquired pneumonia"
    ],
    "minimum_case_requirements": [
      "age and sex",
      "chief complaint",
      "symptom timeline",
      "at least two condition-consistent symptoms",
      "one discriminating exposure, host factor, objective clue, or relevant negative",
      "no direct disease name, pathogen name, or condition ID"
    ]
  }
}

Expert curriculum profile:
{
  "primary_training_lane": "lookalike_differential_discrimination",
  "curriculum_lanes": [
    "core_syndrome_reasoning",
    "common_infection_with_expert_trap",
    "lookalike_differential_discrimination"
  ],
  "expert_value": "high",
  "must_not_miss": false,
  "key_discriminators": [
    "water-system exposure",
    "pneumonia with diarrhea or confusion",
    "hyponatremia",
    "transaminitis",
    "negative respiratory viral testing"
  ],
  "lookalike_differentials": [
    "Community-acquired pneumonia",
    "Mycoplasma pneumonia",
    "Influenza pneumonia",
    "Aspiration pneumonia"
  ],
  "diagnostic_traps": [
    "calling it routine community-acquired pneumonia without using the water exposure",
    "over-anchoring on a respiratory viral syndrome",
    "ignoring gastrointestinal and neurologic clues in pneumonia"
  ],
  "expert_reason": "Trains discrimination between routine pneumonia and an atypical pneumonia pattern driven by exposure and extrapulmonary clues."
}

Generate a concise but diagnostically useful patient case.

Rules:
- Do not reveal the disease name.
- Do not reveal the pathogen name.
- Do not reveal the condition ID.
- Include age, sex, chief complaint, symptom timeline, relevant exposure or host factor, relevant negatives, and basic objective clues.
- Because this is the diagnosis task, include enough already-available examination, laboratory, imaging, microbiology, pathology, or other test results to support a diagnosis when they are relevant.
- Supportive test results may be specific, but they must not explicitly name the hidden disease, pathogen, or condition ID.
- Do not write meta-instructions to the doctor, such as "the team must distinguish" or "this points toward".
- Finish the vignette as a complete clinical note; do not end mid-sentence or with an unfinished diagnostic phrase.
- Include at least one key discriminator from the expert profile when possible.
- If lookalike differentials are listed, create a case where the gold diagnosis is distinguishable but not artificially obvious.
- Keep it focused and clinically dense; concise is better than exhaustive.
- Return plain text only.
```

### Example Generated Patient Case

```text
A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion. He returned 4 days ago from a conference held in an older hotel where he used the indoor spa and spent time near a humidified atrium. He has a 35-pack-year smoking history but no known immunocompromising condition. Temperature is 39.5 C, respiratory rate 28, oxygen saturation 89% on room air, and blood pressure 108/64. Chest radiograph shows a right lower-lobe consolidation. Laboratory testing shows sodium 126 mmol/L and mildly elevated AST/ALT. A multiplex respiratory viral panel is negative. He denies aspiration, recent hospitalization, tuberculosis exposure, and sick household contacts.
```

## 2. Quality Control Example

The generated case is then checked before it is shown to the doctor agent.

### Patient QC Prompt

```text
You are doing binary patient-case quality control for a closed infectious disease diagnosis simulation.

Hidden condition:
{
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "patient_generation": {
    "body_system": [
      "respiratory",
      "gastrointestinal",
      "multisystem_systemic"
    ],
    "time_course_hint": "acute 1 to 7 days",
    "symptoms_to_include": [
      "fever",
      "cough",
      "dyspnea",
      "diarrhea",
      "confusion"
    ],
    "syndrome_tags": [
      "febrile illness",
      "respiratory syndrome",
      "atypical pneumonia"
    ],
    "exposure_clues": [
      "hotel, spa, cooling tower, or building water-system exposure"
    ],
    "host_factor_clues": [
      "older adult",
      "smoking history or chronic lung disease when useful"
    ],
    "objective_or_discriminating_clues": [
      "pneumonia with gastrointestinal or neurologic features",
      "hyponatremia",
      "mild transaminitis",
      "negative routine respiratory viral testing",
      "water-system exposure separates this from routine community-acquired pneumonia"
    ],
    "minimum_case_requirements": [
      "age and sex",
      "chief complaint",
      "symptom timeline",
      "at least two condition-consistent symptoms",
      "one discriminating exposure, host factor, objective clue, or relevant negative",
      "no direct disease name, pathogen name, or condition ID"
    ]
  }
}

Expert curriculum profile:
{
  "primary_training_lane": "lookalike_differential_discrimination",
  "curriculum_lanes": [
    "core_syndrome_reasoning",
    "common_infection_with_expert_trap",
    "lookalike_differential_discrimination"
  ],
  "expert_value": "high",
  "must_not_miss": false,
  "key_discriminators": [
    "water-system exposure",
    "pneumonia with diarrhea or confusion",
    "hyponatremia",
    "transaminitis",
    "negative respiratory viral testing"
  ],
  "lookalike_differentials": [
    "Community-acquired pneumonia",
    "Mycoplasma pneumonia",
    "Influenza pneumonia",
    "Aspiration pneumonia"
  ],
  "diagnostic_traps": [
    "calling it routine community-acquired pneumonia without using the water exposure",
    "over-anchoring on a respiratory viral syndrome",
    "ignoring gastrointestinal and neurologic clues in pneumonia"
  ],
  "expert_reason": "Trains discrimination between routine pneumonia and an atypical pneumonia pattern driven by exposure and extrapulmonary clues."
}

Patient QC criteria:
{
  "leak_terms": [
    "Legionnaires' disease",
    "Legionella"
  ],
  "usable_case_criteria": [
    "does not directly reveal disease name, pathogen name, condition ID, or too-specific parent label",
    "vignette is broadly consistent with hidden condition",
    "contains enough diagnostic information to attempt a diagnosis",
    "does not contain impossible age/host/exposure contradictions",
    "not artificially obvious by naming the diagnosis in disguise"
  ],
  "discard_if": [
    "diagnosis or pathogen is leaked",
    "case is unrelated to hidden condition",
    "case contradicts required age/host/exposure context",
    "case lacks sufficient diagnostic clues"
  ]
}

Generated patient case:
A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion. He returned 4 days ago from a conference held in an older hotel where he used the indoor spa and spent time near a humidified atrium. He has a 35-pack-year smoking history but no known immunocompromising condition. Temperature is 39.5 C, respiratory rate 28, oxygen saturation 89% on room air, and blood pressure 108/64. Chest radiograph shows a right lower-lobe consolidation. Laboratory testing shows sodium 126 mmol/L and mildly elevated AST/ALT. A multiplex respiratory viral panel is negative. He denies aspiration, recent hospitalization, tuberculosis exposure, and sick household contacts.

Mark unusable if:
- diagnosis, pathogen, or condition ID is directly leaked
- the vignette is unrelated to the condition
- the vignette lacks enough diagnostic information
- the vignette contradicts the key host or exposure context
- the vignette ends mid-sentence or contains an unfinished phrase that implies the hidden answer

Do not mark unusable merely because the case includes supportive examination, laboratory,
imaging, microbiology, pathology, or other test results. In the diagnosis task, those
results are expected. Supportive results are only leakage if they explicitly name the
hidden disease, pathogen, or condition ID.

Return JSON only:
{
  "usable": 1 or 0,
  "reason": null or "short reason"
}
```

### Example QC Output

```json
{
  "usable": 1,
  "reason": null
}
```

### Example QC Failure For Contrast

If the generated patient case had said "urinary antigen is positive for Legionella" or named "Legionnaires' disease", QC should reject it:

```json
{
  "usable": 0,
  "reason": "The vignette directly leaks the pathogen or diagnosis, so the doctor would not need to infer the infectious diagnosis."
}
```

## 3. Case Example To Doctor Agent With Mock Experience And Reflections

After QC passes, the doctor agent receives the patient case plus retrieved memory. In a fresh run, memory may be empty. Later in the run, the prompt can contain successful cases and validated reflection rules. The example below uses mock memory to show the intended shape.

### Mock Retrieved Memory Context

```text
Successful cases:
- Case ID: success_000041
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  General rule: General rule: in respiratory syndrome presentations, favor Legionnaires' disease when pneumonia is paired with water-system exposure plus gastrointestinal or neurologic clues, and use this to separate the case from routine community-acquired pneumonia.
  Example case pattern: A 70-year-old smoker developed fever, dyspnea, diarrhea, and confusion after staying in a hotel with a spa. Imaging showed lobar pneumonia, sodium was low, liver enzymes were mildly elevated, and respiratory viral testing was negative.
  Example diagnosis: Diagnosis: Legionnaires' disease. Diagnostic justification: Hotel water-system exposure plus pneumonia with diarrhea, confusion, hyponatremia, and transaminitis makes this more specific than routine community-acquired pneumonia.

- Case ID: success_000052
  Condition: Mycoplasma pneumonia
  Training lane: common_infection_with_expert_trap
  General rule: General rule: in atypical pneumonia presentations, favor Mycoplasma when the case is subacute with prominent upper-respiratory symptoms or household/school clustering, and do not invent water-system exposure.
  Example case pattern: A young adult had 10 days of cough, low fever, sore throat, and diffuse interstitial infiltrates after several classmates were ill.
  Example diagnosis: Diagnosis: Mycoplasma pneumonia.

Validated reflection rules:
- Reflection ID: reflection_000014
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  Failure mode: missed_exposure
  Reusable rule: Do not label severe pneumonia as routine community-acquired pneumonia when a hotel, spa, humidifier, cooling tower, or other building water-system exposure is paired with diarrhea, confusion, hyponatremia, or transaminitis.

- Reflection ID: reflection_000019
  Condition: Influenza pneumonia
  Training lane: lookalike_differential_discrimination
  Failure mode: missed_discriminator
  Reusable rule: A negative respiratory viral panel and prominent water-system exposure should push away from influenza-like viral pneumonia when bacterial atypical pneumonia clues are present.
```

### Doctor-Agent Prompt

```text
You are an infectious diseases doctor-agent in a closed diagnosis simulation.

Your goal is expert diagnosis, not generic population-level guessing.

Previous successful cases and validated diagnostic reflections:
Successful cases:
- Case ID: success_000041
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  General rule: General rule: in respiratory syndrome presentations, favor Legionnaires' disease when pneumonia is paired with water-system exposure plus gastrointestinal or neurologic clues, and use this to separate the case from routine community-acquired pneumonia.
  Example case pattern: A 70-year-old smoker developed fever, dyspnea, diarrhea, and confusion after staying in a hotel with a spa. Imaging showed lobar pneumonia, sodium was low, liver enzymes were mildly elevated, and respiratory viral testing was negative.
  Example diagnosis: Diagnosis: Legionnaires' disease. Diagnostic justification: Hotel water-system exposure plus pneumonia with diarrhea, confusion, hyponatremia, and transaminitis makes this more specific than routine community-acquired pneumonia.

- Case ID: success_000052
  Condition: Mycoplasma pneumonia
  Training lane: common_infection_with_expert_trap
  General rule: General rule: in atypical pneumonia presentations, favor Mycoplasma when the case is subacute with prominent upper-respiratory symptoms or household/school clustering, and do not invent water-system exposure.
  Example case pattern: A young adult had 10 days of cough, low fever, sore throat, and diffuse interstitial infiltrates after several classmates were ill.
  Example diagnosis: Diagnosis: Mycoplasma pneumonia.

Validated reflection rules:
- Reflection ID: reflection_000014
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  Failure mode: missed_exposure
  Reusable rule: Do not label severe pneumonia as routine community-acquired pneumonia when a hotel, spa, humidifier, cooling tower, or other building water-system exposure is paired with diarrhea, confusion, hyponatremia, or transaminitis.

- Reflection ID: reflection_000019
  Condition: Influenza pneumonia
  Training lane: lookalike_differential_discrimination
  Failure mode: missed_discriminator
  Reusable rule: A negative respiratory viral panel and prominent water-system exposure should push away from influenza-like viral pneumonia when bacterial atypical pneumonia clues are present.

Current patient case:
A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion. He returned 4 days ago from a conference held in an older hotel where he used the indoor spa and spent time near a humidified atrium. He has a 35-pack-year smoking history but no known immunocompromising condition. Temperature is 39.5 C, respiratory rate 28, oxygen saturation 89% on room air, and blood pressure 108/64. Chest radiograph shows a right lower-lobe consolidation. Laboratory testing shows sodium 126 mmol/L and mildly elevated AST/ALT. A multiplex respiratory viral panel is negative. He denies aspiration, recent hospitalization, tuberculosis exposure, and sick household contacts.

Task:
Give the most likely infectious diagnosis and a short diagnostic justification.

Return using exactly these headings:

Diagnosis:
Differential:
Diagnostic justification:
Uncertainty:

Rules:
- Focus only on diagnosis.
- Do not provide treatment.
- Do not provide drug doses.
- Do not provide IPC, public-health, antimicrobial stewardship, source-control, or management plans.
- Use syndrome, time course, host context, exposure, and key discriminators.
- Treat prior cases and reflections as reusable diagnostic rules, not as exact templates to copy.
- Keep the answer concise.
```

### Example Doctor-Agent Output

```text
Diagnosis:
Legionnaires' disease.

Differential:
Routine community-acquired bacterial pneumonia, Mycoplasma pneumonia, influenza or other viral pneumonia, aspiration pneumonia, pulmonary tuberculosis.

Diagnostic justification:
The patient has acute pneumonia with prominent extrapulmonary features: watery diarrhea, confusion, hyponatremia, and mild transaminitis. The recent hotel spa/humidified atrium exposure is the key discriminator that pushes toward Legionnaires' disease rather than routine community-acquired pneumonia, Mycoplasma, or viral pneumonia.

Uncertainty:
The diagnosis would be strengthened by targeted testing and respiratory specimen evaluation, but the syndrome-plus-exposure pattern is the best fit among infectious causes.
```

### Hidden Judge Output For This Example

```json
{
  "correct": 1,
  "reflection": null
}
```

### If The Doctor Agent Were Initially Wrong

If the doctor answered "community-acquired pneumonia" and ignored the exposure and extrapulmonary clues, the hidden judge might return:

```json
{
  "correct": 0,
  "reflection": "Do not stop at routine community-acquired pneumonia when a pneumonia case includes building water-system exposure plus diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease."
}
```

The simulation would then retry the same patient with that reflection. If the retry succeeds, the case is saved as a successful case and the reflection is saved as validated memory.

## 4. Doctor With Wrong Answer

This is the same patient case, but here the doctor agent makes the common mistake of treating the case as routine community-acquired pneumonia and underusing the water-system exposure plus extrapulmonary clues.

### Wrong Doctor-Agent Output

```text
Diagnosis:
Community-acquired pneumonia.

Differential:
Influenza pneumonia, Mycoplasma pneumonia, aspiration pneumonia, pulmonary tuberculosis, atypical bacterial pneumonia.

Diagnostic justification:
The patient has fever, cough, hypoxemia, and right lower-lobe consolidation, which are most consistent with community-acquired pneumonia. The diarrhea and confusion may reflect systemic illness from pneumonia, and the negative viral panel makes bacterial pneumonia more likely.

Uncertainty:
The exact bacterial pathogen is uncertain without sputum culture, blood culture, or antigen testing.
```

## 5. Evaluation That Shows Incorrect

In the simulation, this is evaluated by the hidden diagnosis judge. The judge receives the hidden condition, the generated patient case, and the doctor answer. The judge is binary: correct or incorrect.

### Hidden Judge Prompt

```text
You are the hidden judge for a closed infectious disease diagnosis simulation.

The judge is binary and diagnosis-only.

Hidden condition:
{
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "gold_diagnosis": "Legionnaires' disease",
  "accepted_answers": [
    "Legionnaires' disease",
    "Legionella pneumonia",
    "Legionellosis"
  ],
  "parent_disease": "Atypical pneumonia",
  "subtype_or_variant": null,
  "correct_if": [
    "exact disease name or accepted synonym is given",
    "clinically acceptable parent/subtype match is given for this vignette",
    "gold diagnosis appears in the differential and justification tracks the right syndrome"
  ],
  "incorrect_if": [
    "answer anchors on routine community-acquired pneumonia without using the water-system exposure",
    "answer misses gastrointestinal or neurologic clues in pneumonia",
    "answer misses hyponatremia or transaminitis as discriminators",
    "answer gives a diagnosis that would materially derail the case"
  ],
  "expert_curriculum": {
    "primary_training_lane": "lookalike_differential_discrimination",
    "curriculum_lanes": [
      "core_syndrome_reasoning",
      "common_infection_with_expert_trap",
      "lookalike_differential_discrimination"
    ],
    "expert_value": "high",
    "must_not_miss": false,
    "key_discriminators": [
      "water-system exposure",
      "pneumonia with diarrhea or confusion",
      "hyponatremia",
      "transaminitis",
      "negative respiratory viral testing"
    ],
    "lookalike_differentials": [
      "Community-acquired pneumonia",
      "Mycoplasma pneumonia",
      "Influenza pneumonia",
      "Aspiration pneumonia"
    ],
    "diagnostic_traps": [
      "calling it routine community-acquired pneumonia without using the water exposure",
      "over-anchoring on a respiratory viral syndrome",
      "ignoring gastrointestinal and neurologic clues in pneumonia"
    ],
    "expert_reason": "Trains discrimination between routine pneumonia and an atypical pneumonia pattern driven by exposure and extrapulmonary clues."
  }
}

Generated patient case:
A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion. He returned 4 days ago from a conference held in an older hotel where he used the indoor spa and spent time near a humidified atrium. He has a 35-pack-year smoking history but no known immunocompromising condition. Temperature is 39.5 C, respiratory rate 28, oxygen saturation 89% on room air, and blood pressure 108/64. Chest radiograph shows a right lower-lobe consolidation. Laboratory testing shows sodium 126 mmol/L and mildly elevated AST/ALT. A multiplex respiratory viral panel is negative. He denies aspiration, recent hospitalization, tuberculosis exposure, and sick household contacts.

Doctor answer:
Diagnosis:
Community-acquired pneumonia.

Differential:
Influenza pneumonia, Mycoplasma pneumonia, aspiration pneumonia, pulmonary tuberculosis, atypical bacterial pneumonia.

Diagnostic justification:
The patient has fever, cough, hypoxemia, and right lower-lobe consolidation, which are most consistent with community-acquired pneumonia. The diarrhea and confusion may reflect systemic illness from pneumonia, and the negative viral panel makes bacterial pneumonia more likely.

Uncertainty:
The exact bacterial pathogen is uncertain without sputum culture, blood culture, or antigen testing.

Judging rules:
- Mark correct if the answer gives the exact diagnosis, accepted synonym, or clinically acceptable subtype or parent match for this generated case.
- Mark correct if the exact diagnosis is in the differential and the justification clearly tracks the correct syndrome and key discriminator.
- Mark incorrect if the answer follows the intended diagnostic trap, misses the key exposure or host clue, or gives a diagnosis that would derail the case.
- Do not judge treatment, IPC, public health, stewardship, source control, or management.
- If incorrect, the reflection must be a concise reusable diagnostic rule that identifies the missed discriminator or wrong lookalike.
- Do not include hidden chain-of-thought.

Return JSON only:
{
  "correct": 1 or 0,
  "reflection": null if correct, otherwise "one concise reusable diagnostic rule"
}
```

### Hidden Judge Output

```json
{
  "correct": 0,
  "reflection": "Do not stop at routine community-acquired pneumonia when pneumonia follows hotel, spa, humidifier, cooling-tower, or other building water-system exposure and is paired with diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease."
}
```

## 6. Reflection Creation And Prompt To Do So

There is not a separate reflection-generation prompt in the diagnosis loop. Reflection creation is part of the hidden judge prompt above. When the judge marks the answer incorrect, it is instructed to return one concise reusable diagnostic rule in the `reflection` field.

In other words, this judge output is the provisional reflection:

```json
{
  "correct": 0,
  "reflection": "Do not stop at routine community-acquired pneumonia when pneumonia follows hotel, spa, humidifier, cooling-tower, or other building water-system exposure and is paired with diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease."
}
```

At this point the reflection is not yet saved as validated memory. It becomes validated only if the same-patient retry succeeds.

## 7. Case With New Reflection

The retry prompt gives the doctor agent the same patient case, the previous wrong answer, and the provisional reflection. This is the same mechanism used by the diagnosis simulation loop.

### Retry Doctor-Agent Prompt

```text
You are an infectious diseases doctor-agent in a closed diagnosis simulation.

Your goal is expert diagnosis, not generic population-level guessing.

Previous successful cases and validated diagnostic reflections:
Successful cases:
- Case ID: success_000041
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  General rule: General rule: in respiratory syndrome presentations, favor Legionnaires' disease when pneumonia is paired with water-system exposure plus gastrointestinal or neurologic clues, and use this to separate the case from routine community-acquired pneumonia.
  Example case pattern: A 70-year-old smoker developed fever, dyspnea, diarrhea, and confusion after staying in a hotel with a spa. Imaging showed lobar pneumonia, sodium was low, liver enzymes were mildly elevated, and respiratory viral testing was negative.
  Example diagnosis: Diagnosis: Legionnaires' disease. Diagnostic justification: Hotel water-system exposure plus pneumonia with diarrhea, confusion, hyponatremia, and transaminitis makes this more specific than routine community-acquired pneumonia.

Validated reflection rules:
- Reflection ID: reflection_000014
  Condition: Legionnaires' disease
  Training lane: lookalike_differential_discrimination
  Failure mode: missed_exposure
  Reusable rule: Do not label severe pneumonia as routine community-acquired pneumonia when a hotel, spa, humidifier, cooling tower, or other building water-system exposure is paired with diarrhea, confusion, hyponatremia, or transaminitis.

Current patient case:
A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion. He returned 4 days ago from a conference held in an older hotel where he used the indoor spa and spent time near a humidified atrium. He has a 35-pack-year smoking history but no known immunocompromising condition. Temperature is 39.5 C, respiratory rate 28, oxygen saturation 89% on room air, and blood pressure 108/64. Chest radiograph shows a right lower-lobe consolidation. Laboratory testing shows sodium 126 mmol/L and mildly elevated AST/ALT. A multiplex respiratory viral panel is negative. He denies aspiration, recent hospitalization, tuberculosis exposure, and sick household contacts.

Previous answer for this same patient:
Diagnosis:
Community-acquired pneumonia.

Differential:
Influenza pneumonia, Mycoplasma pneumonia, aspiration pneumonia, pulmonary tuberculosis, atypical bacterial pneumonia.

Diagnostic justification:
The patient has fever, cough, hypoxemia, and right lower-lobe consolidation, which are most consistent with community-acquired pneumonia. The diarrhea and confusion may reflect systemic illness from pneumonia, and the negative viral panel makes bacterial pneumonia more likely.

Uncertainty:
The exact bacterial pathogen is uncertain without sputum culture, blood culture, or antigen testing.

Provisional reflection for this same patient:
Do not stop at routine community-acquired pneumonia when pneumonia follows hotel, spa, humidifier, cooling-tower, or other building water-system exposure and is paired with diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease.

Use the reflection to revise the diagnosis without adding treatment or management content.

Task:
Give the most likely infectious diagnosis and a short diagnostic justification.

Return using exactly these headings:

Diagnosis:
Differential:
Diagnostic justification:
Uncertainty:

Rules:
- Focus only on diagnosis.
- Do not provide treatment.
- Do not provide drug doses.
- Do not provide IPC, public-health, antimicrobial stewardship, source-control, or management plans.
- Use syndrome, time course, host context, exposure, and key discriminators.
- Treat prior cases and reflections as reusable diagnostic rules, not as exact templates to copy.
- Keep the answer concise.
```

## 8. Doctor Having Correct Answer With Reflection

The doctor now uses the reflection to revise the answer.

### Correct Retry Doctor-Agent Output

```text
Diagnosis:
Legionnaires' disease.

Differential:
Routine community-acquired bacterial pneumonia, Mycoplasma pneumonia, influenza or other viral pneumonia, aspiration pneumonia, pulmonary tuberculosis.

Diagnostic justification:
This is not just routine community-acquired pneumonia: the case combines acute pneumonia with hotel spa/humidified water-system exposure, watery diarrhea, confusion, hyponatremia, mild transaminitis, and negative respiratory viral testing. Those features are key discriminators for Legionnaires' disease over typical bacterial pneumonia, Mycoplasma, viral pneumonia, or aspiration.

Uncertainty:
The diagnosis would be confirmed by targeted testing and respiratory specimen evaluation, but the syndrome, exposure, and objective clues make Legionnaires' disease the most likely infectious diagnosis.
```

### Retry Hidden Judge Output

```json
{
  "correct": 1,
  "reflection": null
}
```

### Validated Reflection Record Created

Because the retry succeeded, the simulation can now save the provisional reflection as a validated reflection:

```json
{
  "reflection_id": "reflection_000020",
  "successful_case_id": "success_000061",
  "attempted_patient_id": "patient_000083",
  "task_type": "diagnosis",
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "training_lane": "lookalike_differential_discrimination",
  "expert_value": "high",
  "patient_case": "A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion...",
  "retrieval_text": "A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion...",
  "failure_mode": "missed_exposure",
  "reflection": "Do not stop at routine community-acquired pneumonia when pneumonia follows hotel, spa, humidifier, cooling-tower, or other building water-system exposure and is paired with diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease.",
  "validated_by": "same_patient_retry",
  "retrieval_tags": [
    "Legionnaires' disease",
    "Atypical pneumonia",
    "bacterial",
    "core syndrome reasoning",
    "common infection with expert trap",
    "lookalike differential discrimination",
    "respiratory syndrome",
    "water-system exposure"
  ]
}
```

### Successful Case Record Created

The corrected retry answer is also saved as a successful case:

```json
{
  "successful_case_id": "success_000061",
  "attempted_patient_id": "patient_000083",
  "task_type": "diagnosis",
  "condition_id": "inf_example_legionnaires_disease",
  "condition_name": "Legionnaires' disease",
  "training_lane": "lookalike_differential_discrimination",
  "expert_value": "high",
  "patient_case": "A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion...",
  "retrieval_text": "A 62-year-old male presents with 5 days of fever, worsening dry cough, shortness of breath, watery diarrhea, and new confusion...",
  "doctor_answer": "Diagnosis:\nLegionnaires' disease.\n\nDifferential:\nRoutine community-acquired bacterial pneumonia, Mycoplasma pneumonia, influenza or other viral pneumonia, aspiration pneumonia, pulmonary tuberculosis.\n\nDiagnostic justification:\nThis is not just routine community-acquired pneumonia: the case combines acute pneumonia with hotel spa/humidified water-system exposure, watery diarrhea, confusion, hyponatremia, mild transaminitis, and negative respiratory viral testing...",
  "solved_on": "retry",
  "validated_reflection_used": "Do not stop at routine community-acquired pneumonia when pneumonia follows hotel, spa, humidifier, cooling-tower, or other building water-system exposure and is paired with diarrhea, confusion, hyponatremia, or transaminitis; those discriminators should trigger Legionnaires' disease.",
  "expert_lesson": "General rule: in respiratory syndrome presentations, favor Legionnaires' disease when water-system exposure and extrapulmonary clues are present, and use them to separate the case from routine community-acquired pneumonia.",
  "retrieval_tags": [
    "Legionnaires' disease",
    "Atypical pneumonia",
    "bacterial",
    "core syndrome reasoning",
    "common infection with expert trap",
    "lookalike differential discrimination",
    "respiratory syndrome",
    "water-system exposure"
  ]
}
```
