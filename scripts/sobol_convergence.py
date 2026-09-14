"""Convergence check for the variance attribution.

The least cost outlay is the output of a constrained optimisation, so its
dependence on the parameters is strongly interactive: the set of measures held
at a capacity limit changes with the parameters. First order indices are
therefore small and are the hardest to estimate, while total order indices,
which depend only on differences of outputs, converge much faster. This script
computes both at two sample sizes so that the reported ranking is only as
precise as the estimator supports.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from common import save_table, write_values
from fig4_uncertainty import BASE_TH, SPECS
from scenarios import agriculture_measures, agriculture_pathway

from cipo import AR6, Portfolio
from cipo.uncertainty import portfolio_output, sobol_indices

SIZES = (128, 320)


def build():
    cm = AR6()
    port = Portfolio(
        cm, agriculture_measures(convex=True), agriculture_pathway(), horizon=BASE_TH
    )
    fn = portfolio_output(port, SPECS, quantity="cost")
    frames = []
    for n in SIZES:
        df = sobol_indices(fn, SPECS, n_base=n, seed=23)
        df["n_base"] = n
        frames.append(df)
        print(
            "  n_base={:4d}: sum S1 = {:+.3f}, sum ST = {:.3f}, "
            "max |S1| = {:.3f}".format(
                n, df["S1"].sum(), df["ST"].sum(), df["S1"].abs().max()
            )
        )
    out = pd.concat(frames, ignore_index=True)
    save_table(out, "table4d_sobol_convergence")

    fine = frames[-1]
    coarse = frames[0]
    shift = float(np.max(np.abs(fine["ST"].to_numpy() - coarse["ST"].to_numpy())))
    order = fine.sort_values("ST", ascending=False)["parameter"].tolist()
    write_values(
        {
            "sobol_sum_ST": float(fine["ST"].sum()),
            "sobol_sum_S1": float(fine["S1"].sum()),
            "sobol_max_abs_S1": float(fine["S1"].abs().max()),
            "sobol_ST_shift_between_sizes": shift,
            "sobol_n_base_final": SIZES[-1],
            "sobol_rank1": order[0],
            "sobol_rank2": order[1],
            "sobol_rank3": order[2],
            "sobol_ST_rank1": float(fine["ST"].max()),
        }
    )
    print("  largest change in total order between sizes: {:.3f}".format(shift))
    print("  ranking by total order: " + ", ".join(order))
    return out


if __name__ == "__main__":
    build()
