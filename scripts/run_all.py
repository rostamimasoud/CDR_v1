"""Regenerate every figure and table in the manuscript.

Run from the repository root:

    python3 scripts/run_all.py

Each stage writes its figures to ``figures/``, its tables to
``results/tables/`` and every scalar quoted in the manuscript to
``results/key_values.txt``, so the text and the computation stay in step.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

STAGES = [
    ("validation protocol", "validation"),
    ("figure 1, framework", "fig1_framework"),
    ("figure 2, durability", "fig2_durability"),
    ("figure 3, portfolio", "fig3_portfolio"),
    ("figure 4, uncertainty", "fig4_uncertainty"),
    ("figure 5, deployment timing", "fig5_dynamic"),
]


def main() -> int:
    total = time.time()
    for label, module in STAGES:
        print("[{}]".format(label))
        start = time.time()
        mod = __import__(module)
        mod.build()
        print("  done in {:.1f} s".format(time.time() - start))
    print("all stages complete in {:.1f} s".format(time.time() - total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
