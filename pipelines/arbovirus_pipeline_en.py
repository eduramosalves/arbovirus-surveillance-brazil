"""
================================================================================
 PYTHON PIPELINE — ARBOVIRUS ANALYSIS (NORMALIZED DB version)
================================================================================
 Identical methodology to ../arbovirus_pipeline_en.py but reads the normalized
 CSV schema via data_loader.load_pipeline_datasets() and writes outputs to
 ./outputs/ (relative to this script).
================================================================================
"""

import os, sys, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import pymannkendall as mk
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran
import mapclassify

from data_loader import load_pipeline_datasets, UF_REGION

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05

COLORS = {"Dengue": "#C0392B", "Chikungunya": "#E67E22", "Zika": "#2980B9"}
REGION_COLORS = {
    "Norte": "#1B7837", "Nordeste": "#762A83",
    "Sudeste": "#E08214", "Sul": "#4393C3", "Centro-Oeste": "#D6604D",
}

STATE_CENTROIDS = {
    "Rondonia": (-63.0,-10.8), "Acre": (-70.5,-9.0),
    "Amazonas": (-64.0,-3.5),  "Roraima": (-61.4, 2.0),
    "Para": (-52.0,-3.4),      "Amapa": (-51.0, 1.0),
    "Tocantins": (-48.3,-9.0), "Maranhao": (-44.5,-4.9),
    "Piaui": (-42.8,-7.7),     "Ceara": (-39.3,-5.5),
    "Rio Grande do Norte": (-36.7,-5.8), "Paraiba": (-36.7,-7.1),
    "Pernambuco": (-37.8,-8.4),"Alagoas": (-36.6,-9.7),
    "Sergipe": (-37.1,-10.6),  "Bahia": (-41.7,-12.5),
    "Minas Gerais": (-44.7,-18.5), "Espirito Santo": (-40.3,-20.0),
    "Rio de Janeiro": (-43.2,-22.9),"Sao Paulo": (-48.5,-22.0),
    "Parana": (-51.6,-24.9),   "Santa Catarina": (-50.0,-27.2),
    "Rio Grande do Sul": (-53.1,-30.0),
    "Mato Grosso do Sul": (-54.4,-20.5),
    "Mato Grosso": (-56.1,-13.0),
    "Goias": (-49.6,-15.8),    "Distrito Federal": (-47.9,-15.8),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.grid": True,            "grid.alpha": 0.3,
    "figure.facecolor": "white",
})


# ==============================================================================
# 2. DESCRIPTIVE — Annual Incidence
# ==============================================================================
def compute_annual(datasets):
    annual = {}
    for virus, df in datasets.items():
        ann = (df.groupby(["Location_Name", "UF_Code", "Region", "Year"])
                 .agg(Total_Cases=("Cases", "sum"),
                      Population=("Population", "first"))
                 .reset_index())
        ann["Annual_Inc_100k"] = (ann["Total_Cases"] / ann["Population"]) * 1e5
        annual[virus] = ann
    return annual


def plot_descriptive(annual, datasets):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.38})
    fig.suptitle("Annual Arbovirus Incidence in Brazil (per 100,000 inhabitants)",
                 fontsize=14, fontweight="bold", y=1.02)
    for ax, (virus, ann) in zip(axes, annual.items()):
        c = COLORS[virus]
        nat = (ann.groupby("Year")
                  .apply(lambda g: np.average(g["Annual_Inc_100k"], weights=g["Population"]))
                  .reset_index(name="Inc"))
        ax.fill_between(nat["Year"], nat["Inc"], alpha=0.12, color=c)
        ax.plot(nat["Year"], nat["Inc"], marker="o", color=c, linewidth=2.2,
                markersize=5, label="Brazil", zorder=5)
        for region, rg in ann.groupby("Region"):
            rt = (rg.groupby("Year")
                    .apply(lambda g: np.average(g["Annual_Inc_100k"], weights=g["Population"]))
                    .reset_index(name="Inc"))
            ax.plot(rt["Year"], rt["Inc"], linewidth=0.9, linestyle="--",
                    color=REGION_COLORS.get(region, "grey"), alpha=0.75, label=region)
        ax.set_title(virus, fontsize=12, fontweight="bold", color=c, pad=6)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Incidence per 100,000 inhabitants", fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ax == axes[0]:
            ax.legend(fontsize=7, frameon=True, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_01_annual_incidence.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_01_annual_incidence.png")

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), gridspec_kw={"wspace": 0.5})
    fig.suptitle("Annual Incidence by State (per 100,000 inhabitants)",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, (virus, ann) in zip(axes, annual.items()):
        pivot = ann.pivot_table(index="Location_Name", columns="Year",
                                values="Annual_Inc_100k", aggfunc="mean")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.25, linecolor="white",
                    cbar_kws={"label": "Inc./100k", "shrink": 0.7})
        ax.set_title(virus, fontsize=11, fontweight="bold", color=COLORS[virus])
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_02_heatmap_state_year.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_02_heatmap_state_year.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.4})
    fig.suptitle("Monthly Incidence Distribution by Region",
                 fontsize=13, fontweight="bold", y=1.01)
    order = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
    y_max = max(df.dropna(subset=["Monthly_Inc_100k"])["Monthly_Inc_100k"].quantile(0.99)
                for df in datasets.values())
    for ax, (virus, df) in zip(axes, datasets.items()):
        df_c = df.dropna(subset=["Monthly_Inc_100k", "Region"])
        sns.boxplot(data=df_c, x="Region", y="Monthly_Inc_100k", order=order,
                    palette=REGION_COLORS,
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.4}, ax=ax)
        ax.set_title(virus, fontsize=11, fontweight="bold", color=COLORS[virus])
        ax.set_xlabel(""); ax.set_ylabel("Monthly Incidence / 100k", fontsize=9)
        ax.set_ylim(0, y_max * 1.05)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_03_boxplot_regional.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_03_boxplot_regional.png")

    all_data = pd.concat(
        [df.dropna(subset=["Monthly_Inc_100k"]).assign(Virus=virus)
         for virus, df in datasets.items()], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(data=all_data, x="Virus", y="Monthly_Inc_100k",
                palette=COLORS, order=list(COLORS.keys()),
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.4}, ax=ax)
    ax.set_title("Monthly Incidence Comparison by Disease\n(per 100,000 inhabitants)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Monthly Incidence / 100k", fontsize=10)
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_03b_boxplot_disease_comparison.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_03b_boxplot_disease_comparison.png")


# ==============================================================================
# 3. MANN-KENDALL
# ==============================================================================
def run_mann_kendall(annual):
    results = {}
    for virus, ann in annual.items():
        rows = []
        for uf, grp in ann.groupby("Location_Name"):
            series = grp.sort_values("Year")["Annual_Inc_100k"].values
            if len(series) < 4:
                continue
            res = mk.original_test(series, alpha=ALPHA)
            rows.append({
                "Virus": virus, "Location_Name": uf, "Region": grp["Region"].iloc[0],
                "Trend": res.trend, "Tau": res.Tau, "p_value": res.p, "S": res.s,
                "Sen_Slope": res.slope, "Sen_Intercept": res.intercept,
                "Significant": res.p < ALPHA, "n_years": len(series),
            })
        df_res = pd.DataFrame(rows)
        results[virus] = df_res
        n_sig = df_res["Significant"].sum()
        n_inc = (df_res["Trend"] == "increasing").sum()
        n_dec = (df_res["Trend"] == "decreasing").sum()
        print(f"  [{virus:12s}] Significant: {n_sig}/27 | Increasing: {n_inc} | Decreasing: {n_dec}")
    return results


def plot_mann_kendall(mk_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), gridspec_kw={"wspace": 0.5})
    fig.suptitle("Mann-Kendall Test — Kendall's Tau by State\n(coloured bars = p < 0.05)",
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, (virus, df) in zip(axes, mk_results.items()):
        df_s = df.sort_values("Tau"); c = COLORS[virus]
        bar_colors = [c if sig else "#CCCCCC" for sig in df_s["Significant"]]
        ax.barh(df_s["Location_Name"], df_s["Tau"], color=bar_colors,
                edgecolor="white", linewidth=0.4, height=0.75)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xlabel("Kendall's Tau", fontsize=9); ax.set_xlim(-1.1, 1.1)
        ax.tick_params(axis="y", labelsize=7)
        n_sig = df_s["Significant"].sum()
        ax.set_title(f"{virus}\n({n_sig}/27 significant states)",
                     fontsize=10, fontweight="bold", color=c)
        ax.legend(handles=[mpatches.Patch(color=c, label=f"p < {ALPHA}"),
                           mpatches.Patch(color="#CCCCCC", label="Not significant")],
                  fontsize=8, frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_04_mannkendall_tau.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_04_mannkendall_tau.png")

    all_mk = pd.concat(mk_results.values())
    pivot = all_mk.pivot(index="Location_Name", columns="Virus", values="Sen_Slope")
    pivot = pivot.reindex(all_mk.groupby("Location_Name")["Sen_Slope"].mean()
                              .sort_values().index)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pivot)); w = 0.28
    for i, (virus, col) in enumerate(COLORS.items()):
        vals = pivot.get(virus, pd.Series(dtype=float))
        ax.bar(x + i * w, vals, width=w, color=col, alpha=0.85,
               label=virus, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x + w)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Sen's Slope (cases/100k per year)", fontsize=9)
    ax.set_title("Sen's Slope by State — Rate of Change in Incidence",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_05_sens_slope.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_05_sens_slope.png")


# ==============================================================================
# 4. SPEARMAN
# ==============================================================================
def run_spearman(datasets):
    lag_map = {("Temp",0):"Temp_Avg",("Temp",1):"Temp_Lag1",("Temp",2):"Temp_Lag2",
               ("Rain",0):"Rain_mm",("Rain",1):"Rain_Lag1",("Rain",2):"Rain_Lag2"}
    results = {}
    for virus, df in datasets.items():
        rows = []
        for uf, grp in df.groupby("Location_Name"):
            gc = grp.dropna(subset=["Monthly_Inc_100k"])
            for (var, lag), col in lag_map.items():
                if col not in gc.columns:
                    continue
                sub = gc.dropna(subset=[col])
                if len(sub) < 8:
                    continue
                rho, pval = stats.spearmanr(sub["Monthly_Inc_100k"], sub[col])
                rows.append({"Virus": virus, "Location_Name": uf,
                             "Region": gc["Region"].iloc[0], "Variable": var,
                             "Lag": lag, "Rho": rho, "p_value": pval,
                             "Significant": pval < ALPHA, "n": len(sub)})
        df_res = pd.DataFrame(rows); results[virus] = df_res
        print(f"  [{virus:12s}] {len(df_res)} pairs tested | "
              f"{df_res['Significant'].sum()} significant (α = {ALPHA})")
    return results


def plot_spearman(spearman):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"wspace": 0.45})
    fig.suptitle("Spearman Correlation: Monthly Incidence × Climate\n"
                 "(mean Rho across states | * = ≥ 50% of states with p < 0.05)",
                 fontsize=12, fontweight="bold", y=1.04)
    for ax, (virus, df) in zip(axes, spearman.items()):
        summary = (df.groupby(["Variable","Lag"])
                     .agg(Mean_Rho=("Rho","mean"), Prop_Sig=("Significant","mean"))
                     .reset_index())
        pivot_rho = summary.pivot(index="Variable", columns="Lag", values="Mean_Rho")
        pivot_sig = summary.pivot(index="Variable", columns="Lag", values="Prop_Sig")
        im = ax.imshow(pivot_rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["Lag 0","Lag 1","Lag 2"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Rainfall\n(mm)","Temperature\n(°C)"])
        for i in range(pivot_rho.shape[0]):
            for j in range(pivot_rho.shape[1]):
                rho = pivot_rho.values[i,j]
                star = "*" if pivot_sig.values[i,j] >= 0.5 else ""
                ax.text(j, i, f"{rho:.2f}{star}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="white" if abs(rho) > 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rho")
        ax.set_title(virus, fontsize=10, fontweight="bold", color=COLORS[virus])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_06_spearman_summary.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_06_spearman_summary.png")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), gridspec_kw={"wspace":0.45,"hspace":0.5})
    fig.suptitle("Spearman's Rho by State — Lag with Highest Absolute Correlation\n"
                 "(opaque bar = p < 0.05 | L0/L1/L2 = optimal lag)",
                 fontsize=13, fontweight="bold", y=1.02)
    for ci, (virus, df) in enumerate(spearman.items()):
        for ri, var in enumerate(["Temp","Rain"]):
            ax = axes[ri][ci]
            sub = df[df["Variable"] == var].copy()
            best = sub.loc[sub.groupby("Location_Name")["Rho"]
                              .apply(lambda x: x.abs().idxmax())]
            best = best.sort_values("Rho")
            for ii, (_, row) in enumerate(best.iterrows()):
                ax.barh(row["Location_Name"], row["Rho"],
                        color=("#C0392B" if row["Rho"] > 0 else "#2980B9"),
                        alpha=(1.0 if row["Significant"] else 0.28),
                        height=0.72, edgecolor="white", linewidth=0.3)
                xpos = row["Rho"] + (0.03 if row["Rho"] >= 0 else -0.03)
                ax.text(xpos, ii, f"L{row['Lag']}", va="center",
                        ha=("left" if row["Rho"] >= 0 else "right"),
                        fontsize=5.5, color="black")
            ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlim(-1.15, 1.15); ax.tick_params(axis="y", labelsize=6)
            ax.set_title(f"{virus} — {'Temperature' if var=='Temp' else 'Rainfall'}",
                         fontsize=9, fontweight="bold", color=COLORS[virus])
            ax.set_xlabel("Spearman's Rho", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_07_spearman_state_bestlag.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_07_spearman_state_bestlag.png")

    all_sp = pd.concat(spearman.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"wspace": 0.4})
    fig.suptitle("Lag Profile: Mean Rho by Region and Time Lag",
                 fontsize=12, fontweight="bold")
    for ax, var in zip(axes, ["Temp","Rain"]):
        sub = all_sp[all_sp["Variable"]==var]
        reg_lag = sub.groupby(["Region","Lag","Virus"])["Rho"].mean().reset_index()
        styles = {"Dengue":"-","Chikungunya":"--","Zika":"-."}
        for (region, virus_v), g in reg_lag.groupby(["Region","Virus"]):
            ax.plot(g["Lag"], g["Rho"], marker="o", linewidth=1.2,
                    linestyle=styles.get(virus_v,"-"),
                    color=REGION_COLORS.get(region,"grey"), alpha=0.82,
                    label=f"{region} ({virus_v})")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["Lag 0","Lag 1","Lag 2"])
        ax.set_ylabel("Spearman's Rho (mean)", fontsize=9)
        ax.set_title("Temperature" if var=="Temp" else "Rainfall",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(-0.65, 0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0.28, 1, 1])
    fig.legend(handles[:12], labels[:12], loc="lower center", ncol=6,
               fontsize=6.5, bbox_to_anchor=(0.5, 0.02), frameon=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "py_08_lag_profile_region.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_08_lag_profile_region.png")


def plot_spearman_rho_only(spearman):
    lag_order = ["Temp Lag0","Temp Lag1","Temp Lag2","Rain Lag0","Rain Lag1","Rain Lag2"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 10), gridspec_kw={"wspace": 0.55})
    fig.suptitle("Spearman's Rho by State — All Variables and Lags",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, (virus, df) in zip(axes, spearman.items()):
        df2 = df.copy()
        df2["Col"] = df2["Variable"] + " Lag" + df2["Lag"].astype(str)
        pivot = df2.pivot_table(index="Location_Name", columns="Col",
                                values="Rho", aggfunc="mean")
        cols = [c for c in lag_order if c in pivot.columns]
        pivot = pivot[cols]
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, annot=True,
                    fmt=".2f", annot_kws={"size":6.5}, linewidths=0.25,
                    linecolor="white", cbar_kws={"label":"Rho","shrink":0.65})
        ax.set_title(virus, fontsize=11, fontweight="bold", color=COLORS[virus])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(axis="x", labelsize=8, rotation=35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_06b_spearman_rho_only.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_06b_spearman_rho_only.png")


def plot_spearman_rain_significant(spearman):
    sig_data = {}
    for virus, df in spearman.items():
        rain = df[df["Variable"]=="Rain"].copy()
        sig_states = rain[rain["Significant"]]["Location_Name"].unique()
        if len(sig_states) == 0:
            sig_data[virus] = pd.DataFrame(); continue
        sig_only = rain[rain["Significant"] & rain["Location_Name"].isin(sig_states)]
        best = sig_only.loc[sig_only.groupby("Location_Name")["Rho"]
                              .apply(lambda x: x.abs().idxmax())]
        sig_data[virus] = best.sort_values("Rho").reset_index(drop=True)

    n_panels = sum(1 for df in sig_data.values() if not df.empty)
    if n_panels == 0:
        print("    (no significant rain-incidence correlations found)"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={"wspace": 0.5})
    fig.suptitle("States with Significant Spearman Correlation: Rainfall × Incidence\n"
                 "(best significant lag shown | p < 0.05)",
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, (virus, df) in zip(axes, sig_data.items()):
        c = COLORS[virus]
        if df.empty:
            ax.set_title(f"{virus}\n(no significant states)", fontsize=10,
                         fontweight="bold", color=c); ax.axis("off"); continue
        for ii, (_, row) in enumerate(df.iterrows()):
            ax.barh(row["Location_Name"], row["Rho"],
                    color=(c if row["Rho"]>0 else "#2980B9"),
                    height=0.72, edgecolor="white", linewidth=0.3)
            xpos = row["Rho"] + (0.03 if row["Rho"]>=0 else -0.03)
            ax.text(xpos, ii, f"L{int(row['Lag'])}", va="center",
                    ha=("left" if row["Rho"]>=0 else "right"),
                    fontsize=7, color="black")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-1.15, 1.15); ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("Spearman's Rho", fontsize=9)
        ax.set_title(f"{virus}\n({len(df)} significant states)",
                     fontsize=10, fontweight="bold", color=c)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_06d_rain_significant_states.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_06d_rain_significant_states.png")


def run_spearman_rain_lag2(datasets):
    results = {}
    for virus, df in datasets.items():
        rows = []
        for uf, grp in df.groupby("Location_Name"):
            grp_s = grp.sort_values("Date").copy()
            rain = grp_s["Rain_mm"].values
            inc = grp_s["Monthly_Inc_100k"].values
            if len(rain) < 10: continue
            rain_t = rain[:-2]; inc_tp2 = inc[2:]
            valid = np.isfinite(rain_t) & np.isfinite(inc_tp2)
            if valid.sum() < 8: continue
            rho, pval = stats.spearmanr(inc_tp2[valid], rain_t[valid])
            rows.append({"Virus": virus, "Location_Name": uf,
                         "Region": grp["Region"].iloc[0], "Rho": rho,
                         "p_value": pval, "Significant": pval < ALPHA,
                         "n": int(valid.sum())})
        results[virus] = pd.DataFrame(rows)
        print(f"  [{virus:12s}] Rain Lag-2 | {len(rows)} states | "
              f"{results[virus]['Significant'].sum()} significant (α = {ALPHA})")
    return results


def plot_spearman_rain_lag2(rain_lag2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), gridspec_kw={"wspace": 0.5})
    fig.suptitle("Spearman Correlation: Rainfall (month t) × Incidence (month t+2)\n"
                 "(opaque bar = p < 0.05)",
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, (virus, df) in zip(axes, rain_lag2.items()):
        df_s = df.sort_values("Rho"); c = COLORS[virus]
        for ii, (_, row) in enumerate(df_s.iterrows()):
            ax.barh(row["Location_Name"], row["Rho"],
                    color=(c if row["Rho"]>0 else "#2980B9"),
                    alpha=(1.0 if row["Significant"] else 0.28),
                    height=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-1.1, 1.1); ax.tick_params(axis="y", labelsize=7)
        n_sig = int(df_s["Significant"].sum())
        ax.set_title(f"{virus}\n({n_sig} states p < 0.05)",
                     fontsize=10, fontweight="bold", color=c)
        ax.set_xlabel("Spearman's Rho", fontsize=9)
        ax.legend(handles=[mpatches.Patch(color=c, label=f"p < {ALPHA}"),
                           mpatches.Patch(color="#CCCCCC", label="Not significant")],
                  fontsize=8, frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_06c_spearman_rain_lag2.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_06c_spearman_rain_lag2.png")


# ==============================================================================
# 5. SPATIAL
# ==============================================================================
def build_spatial(annual):
    gdf_base = gpd.GeoDataFrame(
        {"Location_Name": list(STATE_CENTROIDS.keys()),
         "lon": [v[0] for v in STATE_CENTROIDS.values()],
         "lat": [v[1] for v in STATE_CENTROIDS.values()]},
        geometry=gpd.points_from_xy(
            [v[0] for v in STATE_CENTROIDS.values()],
            [v[1] for v in STATE_CENTROIDS.values()]), crs="EPSG:4326")
    spatial = {}
    for virus, ann in annual.items():
        mean_inc = (ann.groupby("Location_Name")
                       .apply(lambda g: np.average(g["Annual_Inc_100k"], weights=g["Population"]))
                       .reset_index(name="Mean_Inc"))
        gdf = gdf_base.merge(mean_inc, on="Location_Name", how="left")
        gdf["Region"] = gdf["Location_Name"].map(UF_REGION)
        spatial[virus] = gdf
    return spatial


def run_moran(spatial):
    moran_results = {}
    for virus, gdf in spatial.items():
        gdf_c = gdf.dropna(subset=["Mean_Inc"])
        if len(gdf_c) < 5: continue
        coords = np.column_stack([gdf_c.geometry.x, gdf_c.geometry.y])
        w = KNN.from_array(coords, k=4); w.transform = "r"
        m = Moran(gdf_c["Mean_Inc"].values, w, permutations=999)
        moran_results[virus] = {"I": m.I, "EI": m.EI, "z_sim": m.z_sim,
                                "p_sim": m.p_sim, "Significant": m.p_sim < ALPHA}
        print(f"  [{virus:12s}] I = {m.I:.4f} | p = {m.p_sim:.4f} | "
              f"{'Significant' if m.p_sim < ALPHA else 'NS'}")
    return moran_results


def plot_spatial(spatial, moran_results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Mean Incidence by State — Natural Breaks (Jenks) Classification",
                 fontsize=13, fontweight="bold", y=1.01)
    cmap_j = plt.cm.YlOrRd
    for ax, (virus, gdf) in zip(axes, spatial.items()):
        gdf_c = gdf.dropna(subset=["Mean_Inc"]).copy()
        vals = gdf_c["Mean_Inc"].values; k = min(5, len(np.unique(vals)))
        try:
            clf = mapclassify.NaturalBreaks(vals, k=k)
            bins = np.concatenate([[-np.inf], clf.bins])
        except Exception:
            bins = np.percentile(vals, np.linspace(0,100,k+1)); bins[0] = -np.inf
        gdf_c["Class"] = np.digitize(vals, bins[1:], right=True)
        ax.set_facecolor("#D6EAF8"); ax.set_xlim(-74,-33); ax.set_ylim(-34,6)
        for cls in sorted(gdf_c["Class"].unique()):
            sub = gdf_c[gdf_c["Class"]==cls]
            ax.scatter(sub["lon"], sub["lat"],
                       s=sub["Mean_Inc"]/vals.max()*650 + 60,
                       c=[cmap_j(cls/k)]*len(sub),
                       edgecolors="white", linewidths=0.5, alpha=0.88, zorder=5)
        for _, row in gdf_c.iterrows():
            abbr = row["Location_Name"].split()[-1][:2].upper()
            ax.text(row["lon"], row["lat"], abbr, fontsize=4.5,
                    ha="center", va="center", fontweight="bold", zorder=6)
        mr = moran_results.get(virus, {})
        sub_t = (f"Moran's I = {mr['I']:.3f} | p = {mr['p_sim']:.3f}" if mr else "")
        ax.set_title(f"{virus}\n{sub_t}", fontsize=10, fontweight="bold", color=COLORS[virus])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        try:
            clf2 = mapclassify.NaturalBreaks(vals, k=k)
            bdry = np.concatenate([[0], clf2.bins])
            handles_j = [mpatches.Patch(color=cmap_j(ci/k),
                          label=f"{bdry[ci]:.1f}–{bdry[ci+1]:.1f}") for ci in range(k)]
        except Exception: handles_j = []
        ax.legend(handles=handles_j, title="Inc./100k\n(Jenks)", fontsize=6.5,
                  title_fontsize=7, loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_09_map_jenks.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_09_map_jenks.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.4})
    fig.suptitle("Moran Scatterplot — Spatial Autocorrelation",
                 fontsize=12, fontweight="bold", y=1.02)
    for ax, (virus, gdf) in zip(axes, spatial.items()):
        gdf_c = gdf.dropna(subset=["Mean_Inc"]).copy()
        vals = gdf_c["Mean_Inc"].values
        if len(vals) < 5: continue
        coords = np.column_stack([gdf_c.geometry.x, gdf_c.geometry.y])
        w = KNN.from_array(coords, k=4); w.transform = "r"
        z = (vals - vals.mean()) / vals.std()
        wz = np.array([
            sum(w.weights[i][j] * z[list(w.neighbors[i])[j]]
                for j in range(len(w.neighbors[i])))
            for i in range(len(z))])
        ax.scatter(z, wz, c=COLORS[virus], alpha=0.72, s=45,
                   edgecolors="white", linewidths=0.4, zorder=4)
        m_fit, b_fit = np.polyfit(z, wz, 1)
        xl = np.linspace(z.min(), z.max(), 100)
        ax.plot(xl, m_fit*xl + b_fit, color="black", linewidth=1.2, zorder=3)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
        mr = moran_results.get(virus, {})
        note = (f"I = {mr['I']:.3f}, p = {mr['p_sim']:.3f}" if mr else "")
        ax.set_title(f"{virus}\n{note}", fontsize=10, fontweight="bold", color=COLORS[virus])
        ax.set_xlabel("z (standardised incidence)", fontsize=9)
        ax.set_ylabel("Wz (spatial lag)", fontsize=9)
        for _, row in gdf_c.iterrows():
            zi = (row["Mean_Inc"] - vals.mean()) / vals.std()
            ii = list(gdf_c["Location_Name"]).index(row["Location_Name"])
            ax.annotate(row["Location_Name"][:3], (zi, wz[ii]), fontsize=4.8, alpha=0.72)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "py_10_moran_scatterplot.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> py_10_moran_scatterplot.png")


# ==============================================================================
# 6. EXPORT
# ==============================================================================
def export_results(annual, mk_results, sp_results, moran_results, rain_lag2_results=None):
    path = os.path.join(OUTPUT_DIR, "py_statistical_results.xlsx")
    annual_all = pd.concat([df.assign(Virus=v) for v,df in annual.items()], ignore_index=True)
    mk_all = pd.concat(mk_results.values(), ignore_index=True)
    sp_all = pd.concat(sp_results.values(), ignore_index=True)
    moran_df = pd.DataFrame([{"Virus":v, **{k:res[k] for k in res}}
                             for v,res in moran_results.items()])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        annual_all.to_excel(writer, sheet_name="Annual_Incidence", index=False)
        mk_all.to_excel(writer, sheet_name="Mann_Kendall", index=False)
        sp_all.to_excel(writer, sheet_name="Spearman_Lag", index=False)
        if not moran_df.empty:
            moran_df.to_excel(writer, sheet_name="Global_Moran", index=False)
        if rain_lag2_results:
            rl2 = pd.concat(rain_lag2_results.values(), ignore_index=True)
            rl2.to_excel(writer, sheet_name="Rain_Lag2_Incidence", index=False)
    print(f"    Saved -> {os.path.basename(path)}")


def main():
    sep = "=" * 62
    print(f"\n{sep}")
    print("  ARBOVIRUS PIPELINE — NORMALIZED DB version")
    print(f"{sep}")

    print("\n[1/5] Data loading from normalized DB")
    datasets = load_pipeline_datasets()

    print("\n[2/5] Descriptive — Annual incidence")
    annual = compute_annual(datasets)
    plot_descriptive(annual, datasets)

    print("\n[3/5] Mann-Kendall trend")
    mk_results = run_mann_kendall(annual)
    plot_mann_kendall(mk_results)

    print("\n[4/5] Spearman (lag 0–2)")
    sp_results = run_spearman(datasets)
    plot_spearman(sp_results)

    print("\n[4b/5] Spearman — rho-only / sig rain")
    plot_spearman_rho_only(sp_results)
    plot_spearman_rain_significant(sp_results)

    print("\n[4c/5] Rain Lag-2 (explicit)")
    rain_lag2_results = run_spearman_rain_lag2(datasets)
    plot_spearman_rain_lag2(rain_lag2_results)

    print("\n[5/5] Spatial — Moran + Jenks")
    spatial = build_spatial(annual)
    moran_results = run_moran(spatial)
    plot_spatial(spatial, moran_results)

    print("\n  Exporting...")
    export_results(annual, mk_results, sp_results, moran_results, rain_lag2_results)
    print(f"\n{sep}\n  Done! -> {OUTPUT_DIR}\n{sep}\n")


if __name__ == "__main__":
    main()
