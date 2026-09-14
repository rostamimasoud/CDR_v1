"""Figure 4: uncertainty, attribution and robust design.

Panels
------
a  Distribution of the least cost outlay over parameter draws, with the
   central estimate marked. The spread is the honest uncertainty on the
   headline cost.
b  Sobol indices attributing the variance of that cost to each parameter.
   First order and total order bars are shown together, so the gap between
   them measures interaction.
c  Cost of a neutrality guarantee against the confidence demanded. The
   premium is the price of insisting that neutrality hold under an
   adversarial distribution with the same first two moments.
d  Composition of the expected value portfolio against the robust portfolio.
   Robustness shifts deployment towards measures whose cooling is least
   uncertain.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import pandas as pd
from common import (
    COLOURS,
    DOUBLE,
    SEQUENCE,
    configure,
    light_grid,
    panel_label,
    save,
    save_table,
    write_values,
)
from scenarios import GT, agriculture_measures, agriculture_pathway

from cipo import AR6, Portfolio
from cipo.uncertainty import (
    ParameterSpec,
    cooling_moments,
    monte_carlo,
    portfolio_output,
    attainable_confidence,
    max_guaranteed_confidence,
    robust_portfolio,
    robustness_premium,
    sobol_indices,
)

N_MC = 512
N_SOBOL = 128
BASE_TH = 100.0
CONFIDENCES = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675]

SPECS = [
    ParameterSpec("Climate sensitivity", "thermal", 0.70, 1.30),
    ParameterSpec("CO$_2$ permanent fraction", "a0", 0.85, 1.15),
    ParameterSpec("Carbon feedback strength", "gamma", 0.50, 1.50),
    ParameterSpec("CH$_4$ radiative efficiency", "re", 0.86, 1.14, "CH4"),
    ParameterSpec("CH$_4$ lifetime", "tau", 0.88, 1.12, "CH4"),
    ParameterSpec("Soil carbon durability", "storage", 0.50, 2.00, 0),
    ParameterSpec("Biochar durability", "storage", 0.50, 2.00, 1),
]


def build():
    cm = AR6()
    configure()
    pathway = agriculture_pathway()
    measures = agriculture_measures(convex=True)
    port = Portfolio(cm, measures, pathway, horizon=BASE_TH)
    central = port.minimise_cost(neutral=True)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.66 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # -- a: Monte Carlo distribution of cost --------------------------------
    mc = monte_carlo(port, SPECS, n_samples=N_MC, seed=11)
    good = mc[mc["success"]].copy()
    cost_tn = good["cost"].to_numpy() / 1e12
    ax_a.hist(
        cost_tn, bins=34, color=COLOURS["exp_mid"], edgecolor="white", linewidth=0.3
    )
    q05, q50, q95 = np.percentile(cost_tn, [5, 50, 95])
    ax_a.axvline(central.cost / 1e12, color=COLOURS["accent"], lw=1.1)
    ax_a.axvline(q05, color=COLOURS["permanent"], lw=0.8, ls=(0, (3, 2)))
    ax_a.axvline(q95, color=COLOURS["permanent"], lw=0.8, ls=(0, (3, 2)))
    ax_a.text(
        central.cost / 1e12, ax_a.get_ylim()[1] * 0.97,
        " central estimate", fontsize=6.0, color=COLOURS["accent"], va="top",
    )
    ax_a.set_xlabel("Least cost outlay (trillion USD)")
    ax_a.set_ylabel("Number of\nparameter draws")
    light_grid(ax_a, "y")
    panel_label(ax_a, "a", dx=-0.03, dy=1.14)

    # -- b: Sobol attribution -----------------------------------------------
    fn = portfolio_output(port, SPECS, quantity="cost")
    sob = sobol_indices(fn, SPECS, n_base=N_SOBOL, seed=23)
    order = np.argsort(sob["ST"].to_numpy())
    y = np.arange(len(SPECS))
    ax_b.barh(
        y + 0.19, sob["ST"].to_numpy()[order], height=0.36,
        color=COLOURS["exp_long"], label="Total order", lw=0,
    )
    ax_b.barh(
        y - 0.19, sob["S1"].to_numpy()[order], height=0.36,
        color=COLOURS["exp_short"], label="First order", lw=0,
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([SPECS[i].name for i in order], fontsize=6.0)
    ax_b.set_xlabel("Share of variance in least cost outlay")
    ax_b.set_xlim(0, max(0.75, float(sob["ST"].max()) * 1.12))
    ax_b.legend(loc="lower right")
    light_grid(ax_b, "x")
    panel_label(ax_b, "b", dx=-0.03, dy=1.14)

    # -- c: robustness premium ----------------------------------------------
    mean_b, cov_b, mean_w, var_w = cooling_moments(port, SPECS, n_samples=256, seed=31)
    prem = robustness_premium(
        port, mean_b, cov_b, mean_w, var_w, confidences=CONFIDENCES
    )
    eta_max = attainable_confidence(port, mean_b, cov_b, mean_w, var_w)
    cov_limit = max_guaranteed_confidence(port, mean_b, cov_b)
    feas = prem["feasible"].to_numpy().astype(bool)
    ax_c.plot(
        np.array(CONFIDENCES)[feas] * 100.0,
        prem["cost"].to_numpy()[feas] / 1e12,
        color=COLOURS["permanent"], marker="o", ms=2.6,
    )
    ax_c.axvline(eta_max * 100.0, color=COLOURS["accent"], lw=0.9, ls=(0, (3, 2)))
    ax_c.text(
        0.035, 0.97,
        "guarantee unattainable\nbeyond {:.0f}%".format(eta_max * 100.0),
        transform=ax_c.transAxes, fontsize=6.0, color=COLOURS["accent"],
        ha="left", va="top",
    )
    ax_c.set_xlabel("Confidence that neutrality is met (%)")
    ax_c.set_ylabel("Guaranteed cost\n(trillion USD)")
    light_grid(ax_c)
    panel_label(ax_c, "c", dx=-0.03, dy=1.14)

    # -- d: expected against robust composition ------------------------------
    eta_show = min(0.65, max(0.5, eta_max - 0.01))
    rob = robust_portfolio(port, mean_b, cov_b, mean_w, var_w, confidence=eta_show)
    exp_alpha = robust_portfolio(
        port, mean_b, cov_b, mean_w, var_w, confidence=0.50
    )["alpha"]
    x = np.arange(port.n)
    ax_d.bar(
        x - 0.20, exp_alpha / GT, width=0.38, color=COLOURS["exp_short"],
        label="Neutral in expectation", lw=0,
    )
    ax_d.bar(
        x + 0.20, rob["alpha"] / GT, width=0.38, color=COLOURS["exp_long"],
        label="Guaranteed at {:.0f}% confidence".format(eta_show * 100),
        lw=0,
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(
        [iv.display.replace(" with ", "\nwith ") for iv in measures],
        rotation=32, ha="right", fontsize=5.6,
    )
    ax_d.set_ylabel("Deployment\n(Gt CO$_2$, or Gt CH$_4$)")
    ax_d.set_ylim(0, 1.42 * float(np.max(np.r_[exp_alpha, rob["alpha"]]) / GT))
    ax_d.legend(loc="upper right", handlelength=1.2, borderaxespad=0.4)
    light_grid(ax_d, "y")
    panel_label(ax_d, "d", dx=-0.03, dy=1.14)

    fig.subplots_adjust(hspace=0.80, wspace=0.46)
    save(fig, "fig4_uncertainty")

    save_table(sob, "table4_sobol")
    save_table(prem, "table4b_robustness")
    save_table(
        mc[["cost", "deviation", "peak", "success"]].describe().reset_index(),
        "table4c_monte_carlo_summary",
    )

    rel_sd = np.sqrt(np.diag(cov_b)) / mean_b
    vals = {
        "mc_n_samples": N_MC,
        "mc_success_fraction": float(mc["success"].mean()),
        "mc_cost_median_trillion": float(q50),
        "mc_cost_p05_trillion": float(q05),
        "mc_cost_p95_trillion": float(q95),
        "mc_cost_relative_range": float((q95 - q05) / q50),
        "sobol_n_evaluations": int(sob.attrs.get("n_evaluations", 0)),
        "sobol_top_parameter": str(sob.loc[sob["ST"].idxmax(), "parameter"]),
        "sobol_top_ST": float(sob["ST"].max()),
        "sobol_interaction_max": float(sob["interaction"].max()),
        "robust_cost_50_trillion": float(prem["cost"].iloc[0] / 1e12),
        "robust_cost_high_trillion": float(rob["cost"] / 1e12),
        "robust_confidence_shown": float(eta_show),
        "robust_premium_high_pct": 100.0
        * (rob["cost"] / prem["cost"].iloc[0] - 1.0),
        "attainable_confidence": float(eta_max),
        "covariance_confidence_limit": float(cov_limit["confidence_max"]),
        "kappa_max": float(cov_limit["kappa_max"]),
        "cooling_relative_sd_max": float(np.max(rel_sd)),
        "cooling_relative_sd_min": float(np.min(rel_sd)),
        "BE_mean": mean_w,
        "BE_sd": float(np.sqrt(var_w)),
    }
    for nm, s in zip(port.names, rel_sd):
        vals["cooling_rel_sd_" + nm] = float(s)
    for _, row in sob.iterrows():
        vals["sobol_ST_" + str(row["parameter"]).replace(" ", "_")] = float(row["ST"])
    write_values(vals)
    return sob, prem, vals


if __name__ == "__main__":
    sob, prem, vals = build()
    print(
        "  cost 5-95%: ${:.2f} to ${:.2f} tn (median ${:.2f})".format(
            vals["mc_cost_p05_trillion"],
            vals["mc_cost_p95_trillion"],
            vals["mc_cost_median_trillion"],
        )
    )
    print("  dominant parameter: {} (total order {:.2f})".format(
        vals["sobol_top_parameter"], vals["sobol_top_ST"]
    ))
    print("  attainable confidence {:.1f}% (covariance limit {:.1f}%)".format(
        vals["attainable_confidence"] * 100, vals["covariance_confidence_limit"] * 100
    ))
    print("  premium at {:.0f}% confidence: {:.1f}%".format(
        vals["robust_confidence_shown"] * 100, vals["robust_premium_high_pct"]
    ))
