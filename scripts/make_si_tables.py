"""Render the computed tables as LaTeX for the Supplementary Information.

Every table written by the analysis scripts is converted here, so that the
Supplementary Information contains the full numerical output of the study
rather than a selection of it. The formatting rules are per column: counts
stay integers, fractions and shares are given to three decimals, and physical
quantities that span many orders of magnitude are given in scientific
notation.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from common import ROOT, TAB_DIR

OUT = os.path.join(ROOT, "manuscript", "si_tables")
os.makedirs(OUT, exist_ok=True)

#: Table file, caption, and an optional column subset with display names.
SPEC = [
    (
        "table_val1_ar6_metrics",
        "Validation against the assessed emission metrics of the Sixth "
        "Assessment Report. Absolute values are in watt per square metre year "
        "per kilogram; relative metrics are dimensionless.",
        None,
    ),
    (
        "table_val2_he_benchmark",
        "Reproduction of published compensation ratios for temporary removal "
        "offsetting a methane pulse over a hundred year horizon. The release "
        "parameter of the source is converted to the mean storage time used "
        "here before evaluation.",
        None,
    ),
    (
        "table_val3_optimality",
        "Optimality and limiting case checks. Each computed value is compared "
        "against an independent closed form reference.",
        None,
    ),
    (
        "table_val4_structural",
        "Sensitivity of the headline quantities to the choice of thermal "
        "response function.",
        None,
    ),
    (
        "table1_compensation_ratios",
        "Compensation ratios in kilograms of carbon dioxide stored per "
        "kilogram of gas emitted, for representative measures, two gases and "
        "three horizons, with the worst horizon sensitivity over twenty to "
        "five hundred years.",
        None,
    ),
    (
        "table2_durability",
        "Optimal durability, irreducible sensitivity floor, and admissible "
        "durability interval for each profile family and gas.",
        None,
    ),
    (
        "table3_portfolio_by_horizon",
        "Least cost neutral portfolio for the agricultural pathway as a "
        "function of the policy horizon. Deployment is in gigatonnes.",
        None,
    ),
    (
        "table3b_portfolio_detail",
        "Composition of the least cost neutral portfolio for the agricultural "
        "pathway at a hundred year horizon.",
        None,
    ),
    (
        "table4_sobol",
        "First and total order Sobol indices for the least cost outlay of the "
        "agricultural portfolio.",
        None,
    ),
    (
        "table6_atlas",
        "Compensation ratio in kilograms of carbon dioxide stored per kilogram "
        "of gas emitted, for four representative storage times and three "
        "accounting horizons, for each of the eight gases of the atlas. "
        "Storage times are mean storage times of a first order release store.",
        None,
    ),
    (
        "table4d_sobol_convergence",
        "Convergence of the variance attribution with sample size.",
        None,
    ),
    (
        "table4b_robustness",
        "Cost of a guaranteed neutral portfolio against the confidence "
        "demanded, with the attainable ceiling.",
        None,
    ),
    (
        "table4c_monte_carlo_summary",
        "Summary statistics of the Monte Carlo ensemble.",
        None,
    ),
    (
        "table5_schedule",
        "Optimal deployment schedule for the national net zero pathway, in "
        "kilograms started in each decision year.",
        None,
    ),
    (
        "table5b_netzero_detail",
        "Composition of the least cost neutral portfolio for the national net "
        "zero pathway at a hundred year horizon.",
        None,
    ),
]


def _fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return "{:d}".format(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "unbounded" if x > 0 else "n/a"
        a = abs(x)
        if a == 0:
            return "0"
        if a >= 1e5 or a < 1e-3:
            s = "{:.3e}".format(x)
            mant, exp = s.split("e")
            return "${} \\times 10^{{{}}}$".format(mant, int(exp))
        if a >= 100:
            return "{:.1f}".format(x)
        return "{:.3f}".format(x)
    return str(x).replace("_", " ")


#: Long machine readable column names shortened for the printed tables.
HEADERS = {
    "horizon_yr": "horizon (yr)",
    "mean_storage_time_yr": "mean storage (yr)",
    "optimal_storage_time_yr": "optimum (yr)",
    "interval_lower_yr": "interval lower (yr)",
    "interval_upper_yr": "interval upper (yr)",
    "sensitivity_at_threshold": "sensitivity",
    "compensation_ratio_at_threshold": "ratio at threshold",
    "sensitivity_floor": "floor",
    "sensitivity_durable_limit": "durable limit",
    "horizon_sensitivity": "sensitivity",
    "difference_pct": "difference (\\%)",
    "this_study": "this study",
    "relative_error": "rel. error",
    "equilibrium_sensitivity": "eq. sensitivity",
    "thermal_response": "thermal response",
    "release_parameter_yr": "release par. (yr)",
    "premium_fraction": "premium",
    "confidence_max": "max confidence",
    "expected_cooling": "exp. cooling",
    "cooling_std": "cooling s.d.",
    "cooling_per_unit": "cooling per unit",
    "cooling_delivered": "cooling delivered",
    "marginal_cost": "marginal cost",
    "marginal_cost_per_cooling": "marg. cost per cooling",
    "neutrality_residual": "residual",
    "alpha_CH4_TH20": "$\\alpha$ CH$_4$ 20",
    "alpha_CH4_TH100": "$\\alpha$ CH$_4$ 100",
    "alpha_CH4_TH500": "$\\alpha$ CH$_4$ 500",
    "alpha_N2O_TH20": "$\\alpha$ N$_2$O 20",
    "alpha_N2O_TH100": "$\\alpha$ N$_2$O 100",
    "alpha_N2O_TH500": "$\\alpha$ N$_2$O 500",
}


def _header(col: str) -> str:
    if col in HEADERS:
        return HEADERS[col]
    out = col.replace("_", " ")
    out = out.replace("deploy Gt ", "").replace("cooling share ", "")
    out = out.replace("alpha ", "")
    return out


def render(name: str, caption: str, columns=None) -> str:
    path = os.path.join(TAB_DIR, name + ".csv")
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    if columns:
        df = df[list(columns)]
    df = df.applymap(_fmt)
    header = [_header(c) for c in df.columns]
    align = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{" + caption + "}",
        "\\label{tab:" + name + "}",
        # Shrink only when the natural width overflows the text block, so
        # narrow tables keep the body font size.
        "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def build():
    written = []
    for name, caption, cols in SPEC:
        body = render(name, caption, cols)
        if not body:
            print("  skipped (not found): {}".format(name))
            continue
        out = os.path.join(OUT, name + ".tex")
        with open(out, "w") as fh:
            fh.write(body)
        written.append(name)
        print("  si table -> manuscript/si_tables/{}.tex".format(name))
    return written


if __name__ == "__main__":
    build()
