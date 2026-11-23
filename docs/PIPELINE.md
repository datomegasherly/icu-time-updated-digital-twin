# Reproduction pathway (Windows / PowerShell)

All raw/derived patient-level files stay **outside** this Git repository. The commands below assume that an authorized researcher has locally prepared an existing DuckDB from the original restricted data and Stage 7C-expanded cohort files (which contain patient-level information). This release is not an automated one-command raw-to-paper reproduction and does not supply those files.

Example private local paths (replace on your machine):

```powershell
$repo = 'F:\Articles\1\project\ICU_Time_Updated_Digital_Twin_Code_Release_CANDIDATE'
$project = 'F:\Articles\1\project'
$db = "$project\outputs\stage3_rebuild\stage3_new.duckdb"
$config = "$repo\scripts\item_mapping_stage3c.json"
$mimicExpanded = "$project\outputs\stage5b_competing_discharge\mimic_landmark.csv"
$eicuExpanded  = "$project\outputs\stage5b_competing_discharge\eicu_landmark.csv"
$stage7cMetrics = "$project\outputs\stage7c_deterministic_v3\stage7c_rolling_metrics.csv"
```

**Verify these actual paths in your local run.** Do not use guessed placeholder files for a real reproduction; the stage scripts enforce reference gates.

Run the primary method once prerequisites are verified:

```powershell
python "$repo\scripts\icu_stage7e_unified_dynamic_audit.py" `
  --db $db --config $config `
  --expanded-mimic $mimicExpanded --expanded-eicu $eicuExpanded `
  --stage7c-metrics $stage7cMetrics `
  --out "$project\outputs\public_repro_stage7e" `
  --bootstrap 300 --seed 2026

python "$repo\scripts\icu_stage8a_lab_availability.py" `
  --db $db --config $config `
  --out "$project\outputs\public_repro_stage8a"

python "$repo\scripts\icu_stage8b_availability_aware.py" `
  --db $db --config $config `
  --expanded-mimic $mimicExpanded --expanded-eicu $eicuExpanded `
  --stage7e-metrics "$project\outputs\public_repro_stage7e\stage7e_metrics.csv" `
  --out "$project\outputs\public_repro_stage8b" `
  --bootstrap 300 --seed 2026

python "$repo\scripts\icu_stage8c_endpoint_adjudication.py" `
  --db $db --expanded-mimic $mimicExpanded --expanded-eicu $eicuExpanded `
  --stage7e-metrics "$project\outputs\public_repro_stage7e\stage7e_metrics.csv" `
  --out "$project\outputs\public_repro_stage8c"

python "$repo\scripts\icu_stage8d_endpoint_robust_validation.py" `
  --db $db --config $config `
  --stage8b-metrics "$project\outputs\public_repro_stage8b\stage8b_metrics.csv" `
  --stage8c-sensitivity "$project\outputs\public_repro_stage8c\stage8c_endpoint_sensitivity.csv" `
  --out "$project\outputs\public_repro_stage8d" `
  --bootstrap 300 --seed 2026
```

For the aggregate-only scientific cross-stage gate, point `icu_stage8e_scientific_evidence_lock.py` to the five corresponding output folders with `--s7e`, `--s8a`, `--s8b`, `--s8c`, `--s8d` and a **new** `--out` folder. Afterward, check the 8F future-protocol lock and 8G feasibility scripts, but do not mislabel those stages as newly independent validation.

**Important scientific and practical caveats**
- `stage7c_deterministic_v3` and `stage5b_competing_discharge` filenames and availability must be confirmed locally. The code will fail closed if reference checks do not reconcile.
- Earlier independently run Stage 7D suffered numerical drift; the revised unified Stage 7E/8B run uses same-run comparisons. The original historical code is included for provenance, not as a claim of identical output in every environment.
- `icu_stage7b_frozen_model.py` is a dependency module for later stages; its historical main entry point was a different experiment (fixed [24,48] window). Do not use it as the final article analysis by itself.
- The 8F future validation lock used a different conceptual boundary statement in some older documentation; the executed Stage 7C/8C/8D analyses use **(t,t+24h]**. Resolve prospectively with versioned protocol amendment.
- For MIMIC laboratory availability the proxy is `labevents.storetime`; for eICU it is `labresultrevisedoffset`, *not* verified first clinical visibility.
- Do not publicly commit the generated outputs unless each release file has separately passed disclosure review.
