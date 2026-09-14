"""Figure 2: compensation ratios and the durability question.

This is the figure that overturns the premise of a durability threshold.

Panels
------
a  Compensation ratio against horizon for representative storage durabilities,
   offsetting methane. The ratio varies by a factor of a few across the
   conventional horizons, and for a temporary store it is not monotone in the
   horizon, since the cumulative cooling it is credited with itself peaks and
   declines.
b  The same for nitrous oxide, whose longer lifetime shifts the curves without
   changing their shape.
c  Worst horizon sensitivity against mean storage time. The curve is U shaped
   with an interior minimum and a strictly positive floor, so no durability
   makes the compensation ratio horizon free, and increasing durability past
   the optimum makes it worse. Species matched removal, shown as a reference
   line at zero, is the only exactly invariant case.
d  Admissible durability intervals against the tolerance. Each interval is
   bounded above as well as below, and all vanish below the floor.
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
    configure,
    light_grid,
    panel_label,
    save,
    save_table,
    write_values,
)
from scenarios import FAMILIES, REPRESENTATIVE

from cipo import AR6, ExponentialRelease, PermanentRemoval, SpeciesRemoval
from cipo.threshold import (
    compensation_ratio,
    durability_interval,
    durability_table,
    log_sensitivity,
    max_log_sensitivity,
    optimal_durability,
    sensitivity_profile,
)

TH_MIN, TH_MAX = 20.0, 500.0


def build():
    cm = AR6()
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.66 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    th = np.geomspace(10.0, 1000.0, 400)
    shades = [COLOURS["exp_short"], COLOURS["exp_mid"], COLOURS["exp_long"], COLOURS["linear"]]

    # -- a, b: compensation ratios -------------------------------------------
    for ax, species, label in (
        (ax_a, "CH4", "CH$_4$"),
        (ax_b, "N2O", "N$_2$O"),
    ):
        for (name, tau_bar), colour in zip(sorted(REPRESENTATIVE.items(), key=lambda kv: kv[1]), shades):
            iv = ExponentialRelease(tau=tau_bar)
            ax.plot(
                th,
                compensation_ratio(cm, iv, species, th),
                color=colour,
                label="{} ($\\bar\\tau$ = {:g} yr)".format(name, tau_bar),
            )
        ax.plot(
            th,
            compensation_ratio(cm, PermanentRemoval(), species, th),
            color=COLOURS["permanent"],
            ls=(0, (4, 2)),
            lw=0.9,
            label="Permanent removal",
        )
        for marker in (20.0, 100.0, 500.0):
            ax.axvline(marker, color=COLOURS["grid"], lw=0.5, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time horizon (yr)")
        ax.set_ylabel(
            "Compensation ratio\n(kg CO$_2$ stored per kg {})".format(label)
        )
        ax.set_xlim(10, 1000)
        light_grid(ax, "y")
    ax_a.legend(loc="lower left", ncol=1)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    # -- c: horizon sensitivity against durability ---------------------------
    taus = np.geomspace(2.0, 3.0e4, 90)
    colours = {
        "Exponential release": COLOURS["exp_mid"],
        "Delayed pulse": COLOURS["delayed"],
        "Constant-rate release": COLOURS["linear"],
        "Afforestation": COLOURS["afforest"],
    }
    optima = {}
    for name, fam in FAMILIES.items():
        prof = sensitivity_profile(cm, fam, "CH4", taus, TH_MIN, TH_MAX, n_horizon=121)
        ax_c.plot(prof["storage_time"], prof["sensitivity"], color=colours[name], label=name)
        opt = optimal_durability(cm, fam, "CH4", TH_MIN, TH_MAX)
        optima[name] = opt
        ax_c.plot(
            [opt["optimal_storage_time"]],
            [opt["sensitivity_floor"]],
            "o",
            ms=3.0,
            color=colours[name],
            zorder=6,
        )
    perm_s = max_log_sensitivity(cm, PermanentRemoval(), "CH4", TH_MIN, TH_MAX)
    ax_c.axhline(perm_s, color=COLOURS["permanent"], lw=0.9, ls=(0, (4, 2)))
    ax_c.text(
        2.6, perm_s + 0.015, "Permanent removal", ha="left", va="bottom",
        fontsize=5.8, color=COLOURS["permanent"],
    )
    ax_c.axhline(0.0, color=COLOURS["removal"], lw=0.9, ls=(0, (1, 1.6)))
    ax_c.text(
        2.6, 0.022, "CH$_4$ removal, horizon invariant", ha="left", va="bottom",
        fontsize=5.8, color=COLOURS["removal"],
    )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Mean storage time $\\bar\\tau$ (yr)")
    ax_c.set_ylabel(
        "Horizon sensitivity\n$\\max_{TH}\\,|\\partial\\ln\\alpha/\\partial\\ln TH|$"
    )
    ax_c.set_xlim(2, 3.0e4)
    # The curves and both reference lines stay below 0.85, so the band above
    # is free and the legend sits there without overlapping anything.
    ax_c.set_ylim(-0.05, 1.36)
    ax_c.legend(
        loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.4,
        borderaxespad=0.25,
    )
    light_grid(ax_c, "y")
    panel_label(ax_c, "c")

    # -- d: admissible intervals --------------------------------------------
    fam = FAMILIES["Exponential release"]
    floor = optima["Exponential release"]["sensitivity_floor"]
    tolerances = np.linspace(floor * 1.001, 0.85, 40)
    lower, upper = [], []
    for tol in tolerances:
        out = durability_interval(
            cm, fam, "CH4", float(tol), TH_MIN, TH_MAX,
            n_horizon=81, optimum=optima["Exponential release"],
        )
        lower.append(out["lower"])
        upper.append(out["upper"])
    lower = np.array(lower)
    upper = np.array(upper)
    ax_d.fill_betweenx(
        tolerances, lower, upper, color=COLOURS["exp_mid"], alpha=0.30, lw=0
    )
    ax_d.plot(lower, tolerances, color=COLOURS["exp_long"], lw=1.0)
    ax_d.plot(upper, tolerances, color=COLOURS["exp_long"], lw=1.0)
    ax_d.axhline(floor, color=COLOURS["accent"], lw=0.9, ls=(0, (3, 2)))
    ax_d.text(
        2.6, floor - 0.012, "irreducible floor {:.2f}".format(floor),
        fontsize=6.0, color=COLOURS["accent"], va="top",
    )
    ax_d.plot(
        [optima["Exponential release"]["optimal_storage_time"]], [floor],
        "o", ms=3.0, color=COLOURS["accent"], zorder=6,
    )
    ax_d.set_xscale("log")
    ax_d.set_xlabel("Mean storage time $\\bar\\tau$ (yr)")
    ax_d.set_ylabel("Sensitivity tolerance $\\epsilon$")
    ax_d.set_xlim(2, 3.0e4)
    ax_d.set_ylim(floor - 0.06, 0.86)
    ax_d.text(
        3.2, 0.80, "admissible durabilities", fontsize=6.2,
        color=COLOURS["exp_long"], va="top",
    )
    light_grid(ax_d)
    panel_label(ax_d, "d")

    fig.subplots_adjust(hspace=0.44, wspace=0.34)
    save(fig, "fig2_durability")

    # -- tables and recorded values -----------------------------------------
    tab = durability_table(
        cm, FAMILIES, ["CH4", "N2O"], tolerances=(0.35, 0.50), horizon_min=TH_MIN,
        horizon_max=TH_MAX,
    )
    save_table(tab, "table2_durability")

    rows = []
    for name, tau_bar in sorted(REPRESENTATIVE.items(), key=lambda kv: kv[1]):
        iv = ExponentialRelease(tau=tau_bar)
        row = {"measure": name, "mean_storage_time_yr": tau_bar}
        for sp in ("CH4", "N2O"):
            for h in (20.0, 100.0, 500.0):
                row["alpha_{}_TH{:g}".format(sp, h)] = float(
                    compensation_ratio(cm, iv, sp, h)
                )
        row["horizon_sensitivity"] = max_log_sensitivity(cm, iv, "CH4", TH_MIN, TH_MAX)
        rows.append(row)
    perm = {"measure": "Permanent removal", "mean_storage_time_yr": np.inf}
    for sp in ("CH4", "N2O"):
        for h in (20.0, 100.0, 500.0):
            perm["alpha_{}_TH{:g}".format(sp, h)] = float(
                compensation_ratio(cm, PermanentRemoval(), sp, h)
            )
    perm["horizon_sensitivity"] = perm_s
    rows.append(perm)
    save_table(pd.DataFrame(rows), "table1_compensation_ratios")

    vals = {"sensitivity_permanent_CH4": perm_s, "threshold_horizon_window": "20 to 500 yr"}
    for name, opt in optima.items():
        key = name.replace(" ", "_").replace("-", "_")
        vals["optimal_durability_" + key] = opt["optimal_storage_time"]
        vals["sensitivity_floor_" + key] = opt["sensitivity_floor"]
        vals["sensitivity_durable_limit_" + key] = opt["sensitivity_durable_limit"]
    for sp in ("CH4", "N2O"):
        for h in (20.0, 100.0, 500.0):
            vals["alpha_perm_{}_TH{:g}".format(sp, h)] = float(
                compensation_ratio(cm, PermanentRemoval(), sp, h)
            )
            vals["alpha_exp50_{}_TH{:g}".format(sp, h)] = float(
                compensation_ratio(cm, ExponentialRelease(tau=50.0), sp, h)
            )
    ratio_20 = float(compensation_ratio(cm, ExponentialRelease(tau=50.0), "CH4", 20.0))
    ratio_500 = float(compensation_ratio(cm, ExponentialRelease(tau=50.0), "CH4", 500.0))
    vals["alpha_exp50_CH4_fold_change_20_to_500"] = ratio_20 / ratio_500
    write_values(vals)
    return optima, perm_s, tab


if __name__ == "__main__":
    optima, perm_s, tab = build()
    for name, opt in optima.items():
        print(
            "  {:<24s} optimum {:7.1f} yr  floor {:.3f}  durable limit {:.3f}".format(
                name,
                opt["optimal_storage_time"],
                opt["sensitivity_floor"],
                opt["sensitivity_durable_limit"],
            )
        )
    print("  permanent removal sensitivity {:.3f}".format(perm_s))
