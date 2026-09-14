"""Figure 1: intervention profiles, their responses, and the physical asymmetry.

Panels
------
a  Stored carbon stock for the four profile families at a common mean storage
   time, which is the invariant used to compare shapes.
b  Temperature response of one unit of each measure. Every temporary measure
   crosses zero: cooling while the carbon is held, then residual warming once
   it returns.
c  Cumulative cooling against the horizon. For temporary measures this peaks
   and then declines, so a longer horizon credits them less.
d  The asymmetry the framework rests on. The cumulative temperature effect of
   a short lived gas saturates; that of carbon dioxide grows without bound,
   because a fraction of a pulse never decays.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from common import (
    COLOURS,
    DOUBLE,
    configure,
    gradient_fill,
    halo,
    light_grid,
    panel_label,
    save,
    spine_style,
    write_values,
)

from cipo import (
    AR6,
    Afforestation,
    DelayedPulse,
    ExponentialRelease,
    LinearRelease,
    PermanentRemoval,
)

TAU_BAR = 60.0


def build():
    cm = AR6()
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.62 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    from scenarios import _afforestation_for

    families = [
        ("Exponential release", ExponentialRelease(tau=TAU_BAR), COLOURS["exp_mid"]),
        ("Delayed pulse", DelayedPulse(tau=TAU_BAR), COLOURS["delayed"]),
        ("Constant-rate release", LinearRelease(release_time=2 * TAU_BAR), COLOURS["linear"]),
        ("Afforestation", _afforestation_for(TAU_BAR), COLOURS["afforest"]),
    ]

    # -- a: stored stock -----------------------------------------------------
    t = np.linspace(0.0, 260.0, 4001)
    for label, iv, colour in families:
        stock = -iv.flux().definite(t)
        gradient_fill(ax_a, t, stock, colour, alpha=0.13, zorder=1)
        ax_a.plot(
            t, stock, color=colour, label=label, zorder=3,
            path_effects=halo(2.0),
        )
    ax_a.axhline(1.0, color=COLOURS["permanent"], lw=0.8, ls=(0, (4, 2)))
    ax_a.text(
        6, 1.025, "Permanent removal", ha="left", va="bottom",
        fontsize=6.0, color=COLOURS["permanent"],
    )
    ax_a.set_xlabel("Time since deployment (yr)")
    ax_a.set_ylabel("Carbon held in store\n(kg CO$_2$ per kg deployed)")
    ax_a.set_xlim(0, 260)
    ax_a.set_ylim(0, 1.12)
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
    light_grid(ax_a, "y")
    panel_label(ax_a, "a")

    # -- b: temperature response --------------------------------------------
    t2 = np.linspace(0.0, 400.0, 6001)
    ax_b.axhspan(0.0, 0.20, color=COLOURS["removal"], alpha=0.07, lw=0, zorder=0)
    ax_b.text(
        247, 0.012, "net warming", ha="right", va="bottom", fontsize=5.6,
        color=COLOURS["removal"], zorder=4,
    )
    for label, iv, colour in families:
        ax_b.plot(
            t2, iv.response(cm, t2) * 1e15, color=colour, label=label,
            zorder=3, path_effects=halo(2.0),
        )
    ax_b.plot(
        t2,
        PermanentRemoval().response(cm, t2) * 1e15,
        color=COLOURS["permanent"],
        ls=(0, (4, 2)),
        lw=0.9,
        label="Permanent removal",
    )
    ax_b.annotate(
        "Permanent removal",
        xy=(208, -0.424), xytext=(140, -0.318),
        fontsize=5.8, color=COLOURS["permanent"],
        arrowprops=dict(arrowstyle="-", lw=0.5, color=COLOURS["permanent"]),
    )
    ax_b.axhline(0.0, color="black", lw=0.5)
    ax_b.set_xlabel("Time since deployment (yr)")
    ax_b.set_ylabel("Temperature response\n(fK per kg CO$_2$ deployed)")
    ax_b.set_xlim(0, 250)
    ax_b.set_ylim(-0.60, 0.155)
    light_grid(ax_b, "y")
    panel_label(ax_b, "b")

    zero_crossings = {}
    for label, iv, _ in families:
        r = iv.response(cm, t2)
        sign = np.sign(r)
        idx = np.where(np.diff(sign) > 0)[0]
        zero_crossings[label] = float(t2[idx[0]]) if idx.size else np.nan

    # -- c: cumulative cooling against horizon -------------------------------
    th = np.geomspace(5.0, 5000.0, 600)
    peaks = {}
    for label, iv, colour in families:
        b = iv.cooling(cm, th)
        ax_c.plot(
            th, b * 1e15, color=colour, label=label, zorder=3,
            path_effects=halo(2.0),
        )
        i = int(np.argmax(b))
        peaks[label] = (float(th[i]), float(b[i]))
        ax_c.plot([th[i]], [b[i] * 1e15], "o", ms=2.4, color=colour, zorder=5)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Time horizon (yr)")
    ax_c.set_ylabel("Cumulative cooling delivered\n(fK yr per kg CO$_2$)")
    ax_c.set_xlim(5, 5000)
    # Capped so that the peaks of the temporary families, which fall between
    # 25 and 29 fK yr, are readable. The panel carries the temporary families
    # alone; the unbounded growth of permanent removal is shown in panel d
    # through the cumulative warming of a carbon dioxide pulse.
    ax_c.set_ylim(0.28, 30.0)
    # The curves all sit above 1.5 fK yr, so the foot of the panel is free.
    ax_c.legend(loc="lower left", handlelength=1.3, borderaxespad=0.3)
    light_grid(ax_c)
    panel_label(ax_c, "c")

    # -- d: the asymmetry ----------------------------------------------------
    th2 = np.geomspace(5.0, 5000.0, 600)
    norm_ch4 = float(cm.iagtp_limit("CH4"))
    ax_d.plot(
        th2,
        cm.iagtp("CH4", th2) / norm_ch4,
        color=COLOURS["removal"],
        label="CH$_4$ pulse (saturates)",
        zorder=3,
        path_effects=halo(2.0),
    )
    ax_d.plot(
        th2,
        cm.iagtp("N2O", th2) / float(cm.iagtp_limit("N2O")),
        color=COLOURS["afforest"],
        label="N$_2$O pulse (saturates)",
        zorder=3,
        path_effects=halo(2.0),
    )
    ax_d.plot(
        th2,
        cm.iagtp("CO2", th2) / norm_ch4,
        color=COLOURS["permanent"],
        label="CO$_2$ pulse (unbounded)",
        zorder=3,
        path_effects=halo(2.0),
    )
    ax_d.axhline(1.0, color=COLOURS["grid"], lw=0.6, ls=":")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Time horizon (yr)")
    ax_d.set_ylabel("Cumulative warming\n(fraction of CH$_4$ asymptote)")
    ax_d.set_xlim(5, 5000)
    ax_d.legend(loc="lower right")
    light_grid(ax_d)
    panel_label(ax_d, "d")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        spine_style(ax)
    fig.subplots_adjust(hspace=0.42, wspace=0.36)
    save(fig, "fig1_framework")

    vals = {
        "fig1_tau_bar": TAU_BAR,
        "co2_permanent_fraction": cm.carbon.permanent_fraction,
        "iagtp_ch4_limit": norm_ch4,
        "iagtp_n2o_limit": float(cm.iagtp_limit("N2O")),
    }
    for label, t0 in zero_crossings.items():
        vals["zero_crossing_" + label.replace(" ", "_").replace("-", "_")] = t0
    for label, (tp, bp) in peaks.items():
        key = label.replace(" ", "_").replace("-", "_")
        vals["cooling_peak_horizon_" + key] = tp
        vals["cooling_peak_value_" + key] = bp
    write_values(vals)
    return zero_crossings, peaks


if __name__ == "__main__":
    zc, pk = build()
    print("  zero crossings (yr):", {k: round(v, 1) for k, v in zc.items()})
    print("  cooling peaks (yr):", {k: round(v[0], 1) for k, v in pk.items()})
