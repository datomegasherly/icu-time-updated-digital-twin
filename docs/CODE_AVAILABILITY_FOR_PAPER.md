# Code availability wording

Public repository reported by owner (confirm that all extracted source files, rather than only a ZIP, are online before using this statement):
https://github.com/datomegasherly/icu-time-updated-digital-twin

## Suggested statement after online file verification

> Code for the reported cohort construction, time-aware feature processing, fixed-model landmark evaluation, and aggregate sensitivity analyses is publicly available at https://github.com/datomegasherly/icu-time-updated-digital-twin under the MIT License. Access to MIMIC-IV and eICU-CRD patient-level data remains subject to the respective PhysioNet credentialing and data-use requirements. No source or patient-level derived records are redistributed in the repository.

Only mention a Zenodo DOI or GitHub version tag once that release actually exists. Avoid claiming a fully independently replicated pipeline: existing tests cover source integrity and synthetic smoke tests only. The repository's MIT license does not apply to restricted patient data.
