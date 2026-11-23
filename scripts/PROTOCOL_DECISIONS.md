# PROTOCOL DECISIONS — pending final lock

**Current target:** landmark at 24h after ICU admission; ICU mortality in (24, 48]h.

**Observed handling:** ICU death during horizon = event; ICU stay until at least 48h without death during horizon = non-event; alive ICU discharge before 48h = excluded (not treated as non-event).

**eICU limitation:** unit discharge status/offset are operational proxies. They do not establish verified death timestamps.

**Research implications:** the estimand is conditional on survival to landmark and evaluable horizon under this cohort-selection rule. AUROC/AUPRC/Brier are not claims about all admissions. This is NOT a prospective deployment evaluation.

**Open decisions before final article:** sensitivity to excluding early-live discharges; source confirmation of hospital/unit outcome semantics; effects of first-health-system-stay vs first-person-stay; site-level variance and calibration; final temperature handling.

**Scientific caution:** do not add early-live discharge cases as negative without changing the estimand and accepting an explicitly different follow-up/competing-event strategy.
