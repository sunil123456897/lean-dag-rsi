# PR Tracker — `tejstead/heilbronn-site`

## 1. PR `#148` — `triangle-n13` & `triangle-n16` 80-digit KKT-Newton Refinements
- **PR URL:** https://github.com/tejstead/heilbronn-site/pull/148
- **Status:** Integrated by maintainer (`tejstead`) into release PR `#151` (https://github.com/tejstead/heilbronn-site/pull/151) with preserved authorship (`Ivan Staroselets`).
- **Branch:** `sunil123456897:kkt-refinement-triangle-n13-n16` -> `tejstead/heilbronn-site:main`
- **Entries updated:**
  - `data/sources/external/triangle-n13/` (`0.0265565289197755801383644112259`, `+3.7316e-14`, corank 1)
  - `data/sources/external/triangle-n16/` (`0.0183555463554611404123253636534`, `+5.7690e-14`, corank 1 + `15+1` interior floater)

## 2. Staged Mega-PR — 46 Genuine New World Records (`triangle-n37..n100`)
- **Package directory:** [`triangle_world_records/`](triangle_world_records/)
- **Summary & PR body:** [`SUMMARY_VERIFIED.md`](triangle_world_records/SUMMARY_VERIFIED.md), [`PR_DESCRIPTION.md`](triangle_world_records/PR_DESCRIPTION.md)
- **Standalone verifier:** `python heilbronn_pr_package/triangle_world_records/verify_records.py`
- **Credit:** `Ivan Staroselets, November 2026`
- **Blocker:** Waiting for [`tejstead/heilbronn-site#151`](https://github.com/tejstead/heilbronn-site/pull/151) (`OPEN / MERGEABLE`, raises `triangle` `N_MAX` from `36` to `100`) to merge into `upstream/main`.
- **Verification:** `46/46` passed `build/vendor/verify_exact.py` (strictly higher than `pr151` in exact `fractions.Fraction`) and `scripts/check_submission.py` (`EXIT CODE: 0`, `valid — feasible in exact arithmetic`).
- **Highlights:** `triangle-n70` (`+3.419%`), `triangle-n47` (`+2.029%`), `triangle-n48` (`+2.002%`), `triangle-n73` (`+1.880%`), `triangle-n41` (`+1.731%`), `triangle-n59` (`+1.485%`), `triangle-n100` (`+0.257%`).
- **Quick status check:**
  ```bash
  gh pr view 151 -R tejstead/heilbronn-site --json state,mergedAt,updatedAt
  ```
