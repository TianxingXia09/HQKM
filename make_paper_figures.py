#!/usr/bin/env python
"""Regenerate fig2-fig6 from the paper-profile results (not the pilot logs).

``code/make_figures.py`` renders the pilot-era figures from the smoke logs and
stamps them PILOT / RESULTS-PENDING. This script is its paper-profile
counterpart: it renders the same five data figures from
``results_paper.json`` -- the assembled output of the pre-registered E1-E5 run
-- through ``hqkm.figures.make_figures``, and stamps each one with the run
provenance and the honest scope limit (E4's real-data cells are environment
errors, so the model comparison is synthetic-only).

fig1 is a schematic with no data in it and is not regenerated here; it stays
the version produced by ``make_figures.py``.

Usage
-----
    python make_paper_figures.py                       # -> ../figures_paper
    python make_paper_figures.py --out-dir ../figures  # overwrite in place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hqkm.figures import make_figures

FOOTER = ("paper profile, pre-registered run (results_paper.json, "
          "2026-09-01); classically simulated; E4 panels cover the synthetic "
          "families S1-S5 only")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results_paper.json")
    p.add_argument("--out-dir", default="../figures_paper")
    p.add_argument("--no-footer", action="store_true")
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    results = (here / args.results).resolve()
    out_dir = (here / args.out_dir).resolve()
    blob = json.loads(results.read_text(encoding="utf-8"))
    counts = {s: len(v["rows"]) for s, v in blob["studies"].items()}
    made = make_figures(str(results), str(out_dir),
                        footer=None if args.no_footer else FOOTER)
    print(f"source {results.name} rows={counts}")
    for name in sorted(made):
        print(f"  {name} -> {out_dir.name}/{name}.pdf")
    missing = {"fig2_benefit_region", "fig3_scaling", "fig4_criterion_outcome",
               "fig5_ablation_grid", "fig6_beta_landscape"} - set(made)
    if missing:
        print(f"WARNING not rendered (no rows?): {sorted(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
