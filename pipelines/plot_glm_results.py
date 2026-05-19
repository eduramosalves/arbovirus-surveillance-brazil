"""
================================================================================
 PLOT GLM RESULTS — Poisson / quasi-Poisson / Negative Binomial
================================================================================
 Reads the CSVs produced by glm_common.run_models / export_results and
 generates publication-style figures of the climate IRRs.

 Outputs (PNG, 300 dpi) -> outputs/:
   glm_fig1_pooled_climate_irr.png   : pooled-FE IRR bars (Temp, Rain) by virus
                                        and model family
   glm_fig2_per_state_heatmap.png    : per-state IRR heatmap (negbin), Temp/Rain
   glm_fig3_lag_profile.png          : median IRR by lag (0/1/2) for Temp & Rain
   glm_fig4_aic_comparison.png       : model-fit AIC across family x spec x virus
   glm_fig5_forest_significant.png   : forest plot of significant per-state IRRs
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

VIRUS_ORDER  = ["Dengue", "Chikungunya", "Zika"]
VIRUS_COLOR  = {"Dengue": "#d7263d", "Chikungunya": "#1b998b", "Zika": "#2e86ab"}
FAMILY_ORDER = ["poisson", "quasi_poisson", "negbin"]
FAMILY_LABEL = {"poisson": "Poisson",
                "quasi_poisson": "Quasi-Poisson",
                "negbin": "Negative Binomial"}
LAG_TERMS_TEMP = ["Temp_Avg", "Temp_Lag1", "Temp_Lag2"]
LAG_TERMS_RAIN = ["Rain_mm",  "Rain_Lag1", "Rain_Lag2"]
LAG_LABELS     = ["Lag 0", "Lag 1", "Lag 2"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _read(family, spec):
    fp = os.path.join(OUT, f"glm_{family}_{spec}.csv")
    return pd.read_csv(fp)


# -----------------------------------------------------------------------------
# Figure 1 — Pooled-FE climate IRRs (Temp & Rain) by virus x model family
# -----------------------------------------------------------------------------
def fig1_pooled_climate_irr():
    rows = []
    for fam in FAMILY_ORDER:
        df = _read(fam, "pooled_fe")
        clim = df[df["Term"].isin(["Temp_Avg", "Rain_mm"])].copy()
        clim["Family"] = fam
        rows.append(clim)
    data = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, term, title in zip(
        axes, ["Temp_Avg", "Rain_mm"],
        ["Mean temperature (IRR per +1 °C)", "Rainfall (IRR per +1 mm)"]
    ):
        sub = data[data["Term"] == term]
        x = np.arange(len(VIRUS_ORDER))
        width = 0.26
        for i, fam in enumerate(FAMILY_ORDER):
            f = sub[sub["Family"] == fam].set_index("Virus").reindex(VIRUS_ORDER)
            irr = f["IRR"].values
            lo  = irr - f["IRR_CI_L"].values
            hi  = f["IRR_CI_U"].values - irr
            bars = ax.bar(x + (i - 1) * width, irr, width,
                          label=FAMILY_LABEL[fam],
                          edgecolor="black", linewidth=0.6,
                          color=plt.cm.viridis(0.2 + 0.3 * i))
            ax.errorbar(x + (i - 1) * width, irr, yerr=[lo, hi],
                        fmt="none", ecolor="black", capsize=3, lw=0.8)
            for j, (v, b, sig) in enumerate(zip(VIRUS_ORDER, bars, f["Significant"].values)):
                if bool(sig):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                            "*", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(VIRUS_ORDER)
        ax.set_ylabel("Incidence Rate Ratio (IRR)")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Pooled GLM with state fixed effects — climate IRRs (95% CI; * p<0.05)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT, "glm_fig1_pooled_climate_irr.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 2 — Per-state IRR heatmap (NegBin), Temp_Avg & Rain_mm
# -----------------------------------------------------------------------------
def fig2_per_state_heatmap():
    df = _read("negbin", "per_state")
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, term, title in zip(
        axes, ["Temp_Avg", "Rain_mm"],
        ["Mean temperature  (IRR per +1 °C)", "Rainfall  (IRR per +1 mm)"]
    ):
        sub = df[df["Term"] == term]
        # Pivot: rows = states (alphabetical), cols = viruses
        mat  = sub.pivot(index="Location_Name", columns="Virus", values="IRR")
        sig  = sub.pivot(index="Location_Name", columns="Virus", values="Significant")
        mat  = mat[VIRUS_ORDER]
        sig  = sig.reindex(columns=VIRUS_ORDER)
        log_mat = np.log2(mat.values)
        vmax = float(np.nanmax(np.abs(log_mat)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(log_mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(VIRUS_ORDER)))
        ax.set_xticklabels(VIRUS_ORDER, rotation=0)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        ax.set_title(title)
        # Annotate IRR + significance marker
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                marker = "*" if bool(sig.values[i, j]) else ""
                ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(log_mat[i, j]) > vmax * 0.55 else "black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("log₂(IRR)")
    fig.suptitle("Per-state Negative-Binomial GLM — climate IRRs by state and virus\n"
                 "(* significant at p<0.05; red = positive association, blue = negative)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT, "glm_fig2_per_state_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 3 — Lag profile (median IRR + IQR across states) by virus
# -----------------------------------------------------------------------------
def fig3_lag_profile():
    df = _read("negbin", "lagged")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    panels = [
        (axes[0], LAG_TERMS_TEMP, "Temperature lag profile",
         "IRR per +1 °C"),
        (axes[1], LAG_TERMS_RAIN, "Rainfall lag profile",
         "IRR per +1 mm"),
    ]
    for ax, terms, title, ylab in panels:
        for v in VIRUS_ORDER:
            med = []
            lo, hi = [], []
            for t in terms:
                irr_vals = df[(df["Virus"] == v) & (df["Term"] == t)]["IRR"].values
                irr_vals = irr_vals[np.isfinite(irr_vals)]
                if len(irr_vals) == 0:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                med.append(np.median(irr_vals))
                lo.append(np.percentile(irr_vals, 25))
                hi.append(np.percentile(irr_vals, 75))
            x = np.arange(len(terms))
            med = np.array(med); lo = np.array(lo); hi = np.array(hi)
            ax.plot(x, med, "-o", color=VIRUS_COLOR[v], label=v, lw=2, markersize=7)
            ax.fill_between(x, lo, hi, color=VIRUS_COLOR[v], alpha=0.15)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(terms)))
        ax.set_xticklabels(LAG_LABELS)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("Per-state NegBin GLM with lagged climate predictors — "
                 "median IRR across states (shaded = IQR)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT, "glm_fig3_lag_profile.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 4 — AIC model comparison (per_state / lagged / pooled_fe)
# -----------------------------------------------------------------------------
def fig4_aic_comparison():
    rows = []
    for fam in FAMILY_ORDER:
        for spec in ("per_state", "lagged", "pooled_fe"):
            df = _read(fam, spec)
            # one AIC per (Virus, Location_Name); take unique combos
            uniq = df[["Virus", "Location_Name", "AIC"]].drop_duplicates()
            uniq = uniq[np.isfinite(uniq["AIC"])]
            for v, g in uniq.groupby("Virus"):
                rows.append({"Family": fam, "Spec": spec, "Virus": v,
                             "AIC_median": g["AIC"].median(),
                             "AIC_n": len(g)})
    agg = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, v in zip(axes, VIRUS_ORDER):
        sub = agg[agg["Virus"] == v]
        specs = ["per_state", "lagged", "pooled_fe"]
        x = np.arange(len(specs))
        width = 0.26
        for i, fam in enumerate(FAMILY_ORDER):
            vals = (sub[sub["Family"] == fam].set_index("Spec")
                    .reindex(specs)["AIC_median"].values)
            ax.bar(x + (i - 1) * width, vals, width,
                   label=FAMILY_LABEL[fam],
                   edgecolor="black", linewidth=0.6,
                   color=plt.cm.viridis(0.2 + 0.3 * i))
        ax.set_xticks(x)
        ax.set_xticklabels(["per-state", "lagged", "pooled FE"])
        ax.set_ylabel("Median AIC  (log scale; lower = better fit)")
        ax.set_yscale("log")
        ax.set_title(v, color=VIRUS_COLOR[v])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Model fit comparison — median AIC across states (log scale)\n"
                 "Negative Binomial fits dengue/chikv/zika monthly counts "
                 "vastly better than Poisson due to overdispersion",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT, "glm_fig4_aic_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 5 — Forest plot of significant per-state climate IRRs (NegBin)
# -----------------------------------------------------------------------------
def fig5_forest_significant():
    df = _read("negbin", "per_state")
    df = df[df["Term"].isin(["Temp_Avg", "Rain_mm"]) & df["Significant"]].copy()
    if df.empty:
        print("  no significant per-state rows; skipping forest plot")
        return
    df["log_irr"]   = np.log2(df["IRR"])
    df["log_lo"]    = np.log2(df["IRR_CI_L"])
    df["log_hi"]    = np.log2(df["IRR_CI_U"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 11))
    for ax, term, title in zip(
        axes, ["Temp_Avg", "Rain_mm"],
        ["Significant TEMPERATURE associations", "Significant RAINFALL associations"]
    ):
        sub = df[df["Term"] == term].copy()
        if sub.empty:
            ax.set_title(f"{title}\n(no significant states)")
            ax.axis("off")
            continue
        sub["label"] = sub["Location_Name"] + "  (" + sub["Virus"] + ")"
        sub = sub.sort_values(["Virus", "log_irr"])
        y = np.arange(len(sub))
        colors = [VIRUS_COLOR[v] for v in sub["Virus"]]
        ax.errorbar(sub["log_irr"], y,
                    xerr=[sub["log_irr"] - sub["log_lo"],
                          sub["log_hi"] - sub["log_irr"]],
                    fmt="o", ecolor="gray", elinewidth=0.8, capsize=2,
                    markersize=5, mfc="white", mec="black", lw=0)
        for xi, yi, c in zip(sub["log_irr"], y, colors):
            ax.scatter(xi, yi, color=c, s=30, zorder=3, edgecolor="black", lw=0.4)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=7)
        ax.set_xlabel("log₂(IRR)   (right = positive,  left = negative)")
        ax.set_title(title)
    # Single legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=VIRUS_COLOR[v], markeredgecolor="black",
                          markersize=8, label=v) for v in VIRUS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Forest plot of significant per-state climate effects "
                 "(Negative-Binomial GLM, p<0.05)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = os.path.join(OUT, "glm_fig5_forest_significant.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    print("Plotting GLM results ...")
    fig1_pooled_climate_irr()
    fig2_per_state_heatmap()
    fig3_lag_profile()
    fig4_aic_comparison()
    fig5_forest_significant()
    print("Done.")
