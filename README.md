# A Time-Updated ICU Digital Twin for Mortality Prediction — Reproducibility Code (RELEASE CANDIDATE)

**Status:** unpublished author-review candidate; **not** an independent end-to-end replication or a newly validated model.

This repository candidate collects the actual Python analysis scripts and mapping configurations used in the retrospective, landmark-based ICU risk forecasting project. It is for reproducing the published analysis **only on datasets separately acquired through authorized PhysioNet access**. No patient records or fitted models are included. The code is not a clinical decision-support tool.

## Study and methods
- MIMIC-IV development and held-out internal evaluation; exploratory eICU-CRD transfer assessment.
- Landmark hours: **6, 12, 18, 24**; rolling **(t, t + 24 h]** ICU mortality as implemented.
- Three arms: vital signs, laboratory, integrated. Each fixed logistic regression is trained once on six-hour MIMIC data and reused at later landmarks.
- Two laboratory clocks: specimen/measurement timestamp and conservative stored/revised-result proxy. Their MIMIC model fits are **separate**, so timing comparisons are descriptive.
- eICU data informed **post-hoc** feature selection; external results are exploratory, not independent confirmatory validation.
- Patient-cluster bootstrap quantifies uncertainty in **aggregate** model comparison, not individual risk intervals.

## Repository contents
- `scripts/`: original analysis and audit scripts; mapped config files kept alongside the scripts because their Python imports use same-directory modules.
- `docs/PIPELINE.md`: execution instructions and prerequisites. Not all historical exploratory stages need to be repeated to reproduce the locked primary comparisons.
- `docs/DATA_ACCESS.md`: PhysioNet access conditions and citation guidance.
- `release_manifest.json`: SHA256 of every preserved original script/config and the seven Stage8F source fingerprints.
- `tools/verify_release.py`: standard-library-only release integrity, secrets and forbidden-file checker.
- `tools/smoke_test.py`: source syntax checks and synthetic tests where dependencies are available.
- `requirements.txt`: dependency names only; *exact old versions must be captured separately by the authors*.
- `requirements-smoke-tested-windows.txt`: pinned package versions **observed on the authors' Windows/Python 3.14 machine during the Nov 2026 static/synthetic test**; not claimed to be the original analytical environment.
- `docs/SMOKE_TEST_2026-11-23_FA.md`: author-supplied test log summary and remaining publication gates.

## Quick check (PowerShell, Windows)
```powershell
cd .\ICU_Time_Updated_Digital_Twin_Code_Release_CANDIDATE
python .\tools\verify_release.py
python -m pip install -r .\requirements.txt
python .\tools\smoke_test.py
python -m pip freeze > .\environment_authors.txt
```

For actual patient-data analysis, use a **separate private working copy**, pointing CLI arguments to datasets only accessible to authorized users. Do **not** commit outputs or the 7.3 GB derived DuckDB. Consult the step-by-step commands in `docs/PIPELINE.md`.

## Important publication prerequisites
1. Obtain authors' approval of this code and choose an explicit software license. No open-source license has been assigned yet.
2. Run `verify_release.py` and `smoke_test.py` in the authors' actual execution environment; compare outputs with locked results.
3. Confirm actual local MIMIC/eICU release numbers and document setup, dependency versions and acquisition access rights.
4. Deposit the vetted repository publicly (GitHub or another research-community-accessible repository); archive a versioned release in Zenodo if desired; obtain a permanent URL/DOI before finalizing the paper's **Code Availability** statement.
5. Resolve the discrepancy in future Stage8F endpoint-window documentation with a versioned amendment, *without retroactively editing the archived analyses*.

**No restricted patient data may be uploaded to this public repository.** See `docs/DATA_ACCESS.md`.
