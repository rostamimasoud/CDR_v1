"""Figure 3: the optimal portfolio for the agricultural case study.

Panels
------
a  The emission pathway being offset, by gas.
b  Net warming with no action, with a cost minimising neutral portfolio, and
   with the cheapest single measure sized to the same neutrality constraint.
   Neutrality equalises the cumulative effect, so the residual trajectory is
   what distinguishes the portfolios.
c  Portfolio composition against the policy horizon. The mix shifts towards
   durable and permanent measures as the horizon lengthens, which is the
   quantitative content of the durability requirement.
d  Cost against residual warming, the attainable frontier. The portfolio
   dominates every single measure strategy, and the gap is the value of
   diversity.
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
from scenarios import GT, MT, UNIT_COSTS, agriculture_measures, agriculture_pathway

from cipo import AR6, Portfolio

HORIZONS = [20.0, 50.0, 100.0, 200.0, 300.0, 500.0]
BASE_TH = 100.0


def build():
    cm = AR6()
    configure()
    pathway = agriculture_pathway()
    measures = agriculture_measures(convex=True)
    port = Portfolio(cm, measures, pathway, horizon=BASE_TH)
    base = port.minimise_cost(neutral=True)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.66 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # -- a: the emission pathway --------------------------------------------
    t = np.linspace(0.0, 150.0, 1501)
    ax_a.plot(
        t, pathway.emissions["CH4"].eval(t) / MT, color=COLOURS["removal"],
        label="CH$_4$ (left axis)",
    )
    ax_a.set_xlabel("Year from present")
    ax_a.set_ylabel("CH$_4$ (Mt yr$^{-1}$)", color=COLOURS["removal"])
    ax_a.tick_params(axis="y", colors=COLOURS["removal"])
    ax_a.set_ylim(0, 9)
    ax_a.set_xlim(0, 150)
    twin = ax_a.twinx()
    twin.plot(
        t, pathway.emissions["N2O"].eval(t) / MT, color=COLOURS["afforest"],
        label="N$_2$O (right axis)",
    )
    twin.set_ylabel("N$_2$O (Mt yr$^{-1}$)", color=COLOURS["afforest"])
    twin.tick_params(axis="y", colors=COLOURS["afforest"])
    twin.set_ylim(0, 0.34)
    twin.spines["top"].set_visible(False)
    lines = ax_a.get_lines() + twin.get_lines()
    ax_a.legend(lines, [l.get_label() for l in lines], loc="lower left")
    light_grid(ax_a, "y")
    panel_label(ax_a, "a", dx=-0.03, dy=1.12)

    # -- b: net warming trajectories ----------------------------------------
    t2 = np.linspace(0.0, 200.0, 2001)
    ax_b.plot(
        t2, pathway.temperature(cm, t2) * 1e3, color=COLOURS["exp_long"],
        label="No action",
    )
    ax_b.plot(
        t2, port.net_temperature(base.alpha, t2) * 1e3, color=COLOURS["net"],
        label="Optimal portfolio",
    )
    # Cheapest single measure meeting the same constraint.
    b = port.cooling_vector()
    marg = np.array([iv.marginal_cost(0.0) for iv in measures])
    ratio = np.where(b > 0, marg / b, np.inf)
    order = np.argsort(ratio)
    single = None
    for idx in order:
        need = port.cumulative_warming() / b[idx]
        cap = measures[idx].max_scale
        if cap is None or need <= cap:
            single = idx
            break
    if single is not None:
        alpha_single = np.zeros(port.n)
        alpha_single[single] = port.cumulative_warming() / b[single]
        ax_b.plot(
            t2, port.net_temperature(alpha_single, t2) * 1e3,
            color=COLOURS["delayed"], ls=(0, (4, 2)),
            label="Single measure ({})".format(measures[single].display.lower()),
        )
    ax_b.axhline(0.0, color="black", lw=0.5)
    ax_b.axvline(BASE_TH, color=COLOURS["grid"], lw=0.6, ls=":")
    ax_b.text(BASE_TH + 3, ax_b.get_ylim()[1] * 0.94, "$TH$", fontsize=6.0)
    ax_b.set_xlabel("Year from present")
    ax_b.set_ylabel("Net warming (mK)")
    ax_b.set_xlim(0, 200)
    ax_b.legend(loc="upper left")
    light_grid(ax_b, "y")
    panel_label(ax_b, "b", dx=-0.03, dy=1.12)

    # -- c: composition against horizon -------------------------------------
    rows = []
    shares = np.zeros((len(HORIZONS), port.n))
    costs = []
    for k, th in enumerate(HORIZONS):
        p = Portfolio(cm, agriculture_measures(convex=True), pathway, horizon=th)
        r = p.minimise_cost(neutral=True)
        deliv = r.alpha * p.cooling_vector()
        total = float(np.sum(deliv))
        shares[k] = deliv / total if total > 0 else deliv
        costs.append(r.cost)
        row = {"horizon_yr": th, "cost_usd": r.cost, "success": r.success}
        for nm, a, d in zip(p.names, r.alpha, deliv):
            row["deploy_Gt_" + nm] = a / GT
            row["cooling_share_" + nm] = d / total if total else 0.0
        rows.append(row)
    bottom = np.zeros(len(HORIZONS))
    x = np.arange(len(HORIZONS))
    for i, iv in enumerate(measures):
        ax_c.bar(
            x, shares[:, i], bottom=bottom, width=0.68,
            color=SEQUENCE[i % len(SEQUENCE)], label=iv.display, lw=0,
        )
        bottom += shares[:, i]
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["{:g}".format(h) for h in HORIZONS])
    ax_c.set_xlabel("Policy time horizon (yr)")
    ax_c.set_ylabel("Share of cumulative\ncooling delivered")
    ax_c.set_ylim(0, 1)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    panel_label(ax_c, "c", dx=-0.03, dy=1.12)

    # -- d: cost against residual warming -----------------------------------
    front = port.pareto_frontier(n_points=22, neutral=False)
    ok = np.isfinite(front["deviation"])
    ax_d.plot(
        front["cost"][ok] / 1e12, front["deviation"][ok] * 1e6,
        color=COLOURS["permanent"], marker="o", ms=2.2, label="Portfolio frontier",
    )
    for i, iv in enumerate(measures):
        cost_i, dev_i = [], []
        for frac in np.linspace(0.05, 1.0, 14):
            a = np.zeros(port.n)
            cap = iv.max_scale if iv.max_scale is not None else port.scales()["alpha"]
            a[i] = frac * cap
            cost_i.append(port.cost(a) / 1e12)
            dev_i.append(port.deviation(a) * 1e6)
        # Coloured as in panel c, whose legend serves the whole figure, so
        # these carry no legend entry of their own and the panel legend stays
        # small enough to sit clear of every curve.
        ax_d.plot(
            cost_i, dev_i, color=SEQUENCE[i % len(SEQUENCE)], lw=0.8,
            ls=(0, (2, 1.4)),
        )
    ax_d.set_xlabel("Deployment cost (trillion USD)")
    ax_d.set_ylabel(
        "Residual warming\n$\\int_0^{TH}\\Delta T_{\\rm net}^2\\,dt$ ($\\mu$K$^2$ yr)"
    )
    ax_d.set_yscale("log")
    ax_d.set_xlim(left=0)
    proxy = plt.Line2D(
        [], [], color="#7a7a7a", lw=0.8, ls=(0, (2, 1.4)),
        label="Single measures, coloured as in c",
    )
    handles, labels = ax_d.get_legend_handles_labels()
    ax_d.legend(
        handles + [proxy], labels + [proxy.get_label()],
        loc="upper right", fontsize=5.4, handlelength=1.6, borderaxespad=0.35,
    )
    light_grid(ax_d)
    panel_label(ax_d, "d", dx=-0.03, dy=1.12)

    fig.subplots_adjust(hspace=0.62, wspace=0.62)
    save(fig, "fig3_portfolio")

    # -- tables and values ---------------------------------------------------
    save_table(pd.DataFrame(rows), "table3_portfolio_by_horizon")
    summary = port.summary_frame(base.alpha)
    summary["deploy_Gt"] = summary["alpha"] / GT
    save_table(summary, "table3b_portfolio_detail")

    perm_unit = UNIT_COSTS["beccs"] / 1000.0
    perm_only = (
        port.cumulative_warming()
        / float(measures[4].cooling(cm, BASE_TH))
        * perm_unit
    )
    at = port.attainability()
    kkt = port.kkt_report(base.alpha)
    vals = {
        "caseA_BE_100": port.cumulative_warming(),
        "caseA_peak_noaction_mK": port.peak_deviation(np.zeros(port.n)) * 1e3,
        "caseA_peak_portfolio_mK": base.peak * 1e3,
        "caseA_cost_trillion": base.cost / 1e12,
        "caseA_cost_permanent_only_trillion": perm_only / 1e12,
        "caseA_saving_vs_permanent_pct": 100.0 * (1.0 - base.cost / perm_only),
        "caseA_coverage": at["coverage"],
        "caseA_kkt_multiplier": kkt["multiplier"],
        "caseA_kkt_relative_spread": kkt["relative_spread"],
        "caseA_gram_condition": port.condition_number(),
        "caseA_n_measures_deployed": int(np.count_nonzero(base.alpha > 1e-9)),
        "caseA_residual_fraction": abs(base.neutrality_residual)
        / port.cumulative_warming(),
    }
    for nm, a, d in zip(port.names, base.alpha, base.alpha * port.cooling_vector()):
        vals["caseA_deploy_Gt_" + nm] = a / GT
        vals["caseA_cooling_share_" + nm] = d / float(
            np.sum(base.alpha * port.cooling_vector())
        )
    if single is not None:
        vals["caseA_single_measure"] = measures[single].name
        vals["caseA_single_peak_mK"] = port.peak_deviation(alpha_single) * 1e3
        vals["caseA_single_cost_trillion"] = port.cost(alpha_single) / 1e12
        vals["caseA_single_deviation_ratio"] = port.deviation(
            alpha_single
        ) / port.deviation(base.alpha)
    write_values(vals)
    return base, port, vals


if __name__ == "__main__":
    base, port, vals = build()
    print("  cost ${:.2f} tn, permanent-only ${:.2f} tn, saving {:.1f}%".format(
        vals["caseA_cost_trillion"],
        vals["caseA_cost_permanent_only_trillion"],
        vals["caseA_saving_vs_permanent_pct"],
    ))
    print("  peak warming: no action {:.2f} mK, portfolio {:.2f} mK".format(
        vals["caseA_peak_noaction_mK"], vals["caseA_peak_portfolio_mK"]
    ))
