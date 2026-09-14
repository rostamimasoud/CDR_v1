"""Figure 5: deployment timing and the national net zero case.

Panels
------
a  The net zero emission pathway being offset, by gas. The retained carbon
   dioxide residual is what makes this case qualitatively harder than the
   agricultural one.
b  Optimal deployment schedule. Deploying everything at once front loads the
   cooling and produces a transient over cooling excursion; the schedule
   spreads deployment and removes it.
c  Net warming under immediate deployment against the optimal schedule, both
   meeting the same cumulative neutrality constraint.
d  Portfolio composition for the two scenarios side by side, showing how a
   permanent emission component forces permanent removal into the mix.
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
from scenarios import (
    GT,
    MT,
    agriculture_measures,
    agriculture_pathway,
    net_zero_measures,
    net_zero_pathway,
)

from cipo import AR6, Portfolio
from cipo.dynamic import DynamicPortfolio

BASE_TH = 100.0
DECISION_TIMES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def build():
    cm = AR6()
    configure()
    pathway = net_zero_pathway()
    measures = net_zero_measures(convex=True)
    port = Portfolio(cm, measures, pathway, horizon=BASE_TH)
    static = port.minimise_cost(neutral=True)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.68 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # -- a: the pathway ------------------------------------------------------
    t = np.linspace(0.0, 150.0, 1501)
    ax_a.plot(
        t, pathway.emissions["CO2"].eval(t) / MT, color=COLOURS["permanent"],
        label="CO$_2$ (left axis)",
    )
    ax_a.set_xlabel("Year from present")
    ax_a.set_ylabel("CO$_2$ (Mt yr$^{-1}$)", color=COLOURS["permanent"])
    ax_a.tick_params(axis="y", colors=COLOURS["permanent"])
    ax_a.set_xlim(0, 150)
    ax_a.set_ylim(0, 430)
    twin = ax_a.twinx()
    twin.plot(
        t, pathway.emissions["CH4"].eval(t) / MT, color=COLOURS["removal"],
        label="CH$_4$ (right axis)",
    )
    twin.plot(
        t, pathway.emissions["N2O"].eval(t) / MT * 10.0,
        color=COLOURS["afforest"], label="N$_2$O $\\times$ 10 (right axis)",
    )
    twin.set_ylabel("CH$_4$, N$_2$O (Mt yr$^{-1}$)")
    twin.set_ylim(0, 5.6)
    twin.spines["top"].set_visible(False)
    lines = ax_a.get_lines() + twin.get_lines()
    ax_a.legend(lines, [l.get_label() for l in lines], loc="upper right")
    light_grid(ax_a, "y")
    panel_label(ax_a, "a", dx=-0.03, dy=1.12)

    # -- b, c: dynamic schedule ---------------------------------------------
    dyn = DynamicPortfolio(
        cm, measures, pathway, horizon=BASE_TH, decision_times=DECISION_TIMES
    )
    sched = dyn.optimise(objective="deviation", neutral=True)
    dep = sched.deployment
    bottom = np.zeros(len(DECISION_TIMES))
    for i, iv in enumerate(measures):
        ax_b.bar(
            np.array(DECISION_TIMES),
            dep[i] / GT,
            bottom=bottom,
            width=7.0,
            color=SEQUENCE[i % len(SEQUENCE)],
            label=iv.display,
            lw=0,
        )
        bottom += dep[i] / GT
    ax_b.set_xlabel("Year deployment begins")
    ax_b.set_ylabel("Deployment started\n(Gt CO$_2$, or Gt CH$_4$)")
    ax_b.set_xlim(-6, 66)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.6)
    light_grid(ax_b, "y")
    panel_label(ax_b, "b", dx=-0.03, dy=1.12)

    t2 = np.linspace(0.0, 200.0, 2001)
    ax_c.plot(
        t2, pathway.temperature(cm, t2) * 1e3, color=COLOURS["exp_long"],
        label="No action",
    )
    ax_c.plot(
        t2, port.net_temperature(static.alpha, t2) * 1e3,
        color=COLOURS["delayed"], ls=(0, (4, 2)),
        label="Immediate deployment",
    )
    ax_c.plot(
        t2, dyn.net_temperature(dep.ravel(), t2) * 1e3, color=COLOURS["net"],
        label="Optimal schedule",
    )
    ax_c.axhline(0.0, color="black", lw=0.5)
    ax_c.axvline(BASE_TH, color=COLOURS["grid"], lw=0.6, ls=":")
    ax_c.set_xlabel("Year from present")
    ax_c.set_ylabel("Net warming (mK)")
    ax_c.set_xlim(0, 200)
    ax_c.legend(loc="lower right")
    light_grid(ax_c, "y")
    panel_label(ax_c, "c", dx=-0.03, dy=1.12)

    # -- d: composition, both scenarios -------------------------------------
    agri_port = Portfolio(
        cm, agriculture_measures(convex=True), agriculture_pathway(), horizon=BASE_TH
    )
    agri = agri_port.minimise_cost(neutral=True)
    agri_share = agri.alpha * agri_port.cooling_vector()
    agri_share = agri_share / agri_share.sum()
    nz_share = static.alpha * port.cooling_vector()
    nz_share = nz_share / nz_share.sum()

    names = [iv.name for iv in measures]
    agri_map = dict(zip(agri_port.names, agri_share))
    left = np.array([agri_map.get(n, 0.0) for n in names])
    x = np.arange(len(names))
    ax_d.bar(
        x - 0.20, left, width=0.38, color=COLOURS["exp_short"],
        label="Agricultural sector", lw=0,
    )
    ax_d.bar(
        x + 0.20, nz_share, width=0.38, color=COLOURS["exp_long"],
        label="National net zero", lw=0,
    )
    ax_d.set_xticks(x)
    # Wrapping these names onto several lines made each label wide enough to
    # meet its neighbour. A single line rotated further offsets them
    # diagonally instead, so they stay apart.
    ax_d.set_xticklabels(
        [iv.display for iv in measures],
        rotation=38, ha="right", rotation_mode="anchor", fontsize=5.2,
    )
    ax_d.set_ylabel("Share of cumulative\ncooling")
    # Afforestation is the tallest bar and sits mid axis, so the upper right,
    # above methane removal, is the free corner.
    ax_d.set_ylim(0, 0.60)
    ax_d.legend(loc="upper right", handlelength=1.3, borderaxespad=0.35)
    light_grid(ax_d, "y")
    panel_label(ax_d, "d", dx=-0.03, dy=1.12)

    fig.subplots_adjust(hspace=0.80, wspace=0.62)
    save(fig, "fig5_dynamic")

    # -- tables and values ---------------------------------------------------
    save_table(sched.frame(port.names).reset_index(), "table5_schedule")
    nz_detail = port.summary_frame(static.alpha)
    nz_detail["deploy_Gt"] = nz_detail["alpha"] / GT
    save_table(nz_detail, "table5b_netzero_detail")

    static_traj = port.net_temperature(static.alpha, t2)
    dyn_traj = dyn.net_temperature(dep.ravel(), t2)
    com = sched.centre_of_mass()
    vals = {
        "caseB_BE_100": port.cumulative_warming(),
        "caseB_cost_trillion": static.cost / 1e12,
        "caseB_peak_noaction_mK": float(
            np.max(np.abs(pathway.temperature(cm, t2))) * 1e3
        ),
        "caseB_peak_immediate_mK": float(np.max(np.abs(static_traj)) * 1e3),
        "caseB_peak_scheduled_mK": float(np.max(np.abs(dyn_traj)) * 1e3),
        "caseB_overcool_immediate_mK": float(-np.min(static_traj) * 1e3),
        "caseB_overcool_scheduled_mK": float(-np.min(dyn_traj) * 1e3),
        "caseB_deviation_immediate": port.deviation(static.alpha),
        "caseB_deviation_scheduled": sched.deviation,
        "caseB_deviation_reduction_pct": 100.0
        * (1.0 - sched.deviation / port.deviation(static.alpha)),
        "caseB_schedule_cost_trillion": sched.cost / 1e12,
        "caseB_schedule_success": str(sched.success),
        "caseB_n_measures_deployed": int(np.count_nonzero(static.alpha > 1e-9)),
        "caseB_coverage": port.attainability()["coverage"],
    }
    for nm, s in zip(port.names, nz_share):
        vals["caseB_cooling_share_" + nm] = float(s)
    for nm, c in zip(port.names, com):
        if np.isfinite(c):
            vals["caseB_deploy_centre_of_mass_" + nm] = float(c)
    write_values(vals)
    return sched, vals


if __name__ == "__main__":
    sched, vals = build()
    print("  immediate deployment over-cools by {:.2f} mK".format(
        vals["caseB_overcool_immediate_mK"]
    ))
    print("  optimal schedule over-cools by {:.2f} mK".format(
        vals["caseB_overcool_scheduled_mK"]
    ))
    print("  residual warming reduced {:.1f}% by timing alone".format(
        vals["caseB_deviation_reduction_pct"]
    ))
