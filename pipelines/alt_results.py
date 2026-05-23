"""
Alternative visualizations selected for the final figure set.

Produces the 5 charts that were chosen from the original 16-figure
exploration and copied into figures/. Reads the same source data as the
main pipeline (db/ and outputs/).

Outputs (PNG, 300 dpi) -> alt_results/:
  trend01_slope_first_last.png       slope chart of first vs last year per state
  trend03_sen_dotbar.png             sorted Sen-slope dot-and-bar per state
  spat01_choropleth_facet.png        faceted choropleth of mean annual incidence
  spat02_lisa_facet.png              faceted LISA-category map
  glm02_caterpillar_temp_irr.png     per-state Temp IRR caterpillar (NegBin)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db"
OUT_MAIN = ROOT / "outputs"
OUT = ROOT / "alt_results"
OUT.mkdir(exist_ok=True)

VIRUS_ORDER = ["Dengue", "Chikungunya", "Zika"]
VIRUS_COLOR = {"Dengue": "#d7263d", "Chikungunya": "#1b998b", "Zika": "#2e86ab"}

IBGE_CODE_TO_NAME = {
    "11": "Rondonia", "12": "Acre", "13": "Amazonas", "14": "Roraima",
    "15": "Para", "16": "Amapa", "17": "Tocantins", "21": "Maranhao",
    "22": "Piaui", "23": "Ceara", "24": "Rio Grande do Norte", "25": "Paraiba",
    "26": "Pernambuco", "27": "Alagoas", "28": "Sergipe", "29": "Bahia",
    "31": "Minas Gerais", "32": "Espirito Santo", "33": "Rio de Janeiro",
    "35": "Sao Paulo", "41": "Parana", "42": "Santa Catarina",
    "43": "Rio Grande do Sul", "50": "Mato Grosso do Sul", "51": "Mato Grosso",
    "52": "Goias", "53": "Distrito Federal",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
def annual_incidence() -> pd.DataFrame:
    return pd.read_excel(OUT_MAIN / "py_statistical_results.xlsx", sheet_name="Annual_Incidence")


def mann_kendall() -> pd.DataFrame:
    return pd.read_excel(OUT_MAIN / "py_statistical_results.xlsx", sheet_name="Mann_Kendall")


def lisa() -> pd.DataFrame:
    return pd.read_excel(OUT_MAIN / "geo_spatial_stats.xlsx", sheet_name="LISA_Clusters")


def mean_inc() -> pd.DataFrame:
    return pd.read_excel(OUT_MAIN / "geo_spatial_stats.xlsx", sheet_name="Mean_Incidence")


def glm(family: str, spec: str) -> pd.DataFrame:
    return pd.read_csv(OUT_MAIN / f"glm_{family}_{spec}.csv")


def load_states_gdf() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DB / "brazil_states.geojson")
    gdf["Location_Name"] = gdf["codarea"].astype(str).map(IBGE_CODE_TO_NAME)
    gdf = gdf[gdf["Location_Name"].notna()].copy()
    gdf = gdf.dissolve(by="Location_Name", as_index=False)
    return gdf


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_slope_chart(ai: pd.DataFrame, mk: pd.DataFrame) -> Path:
    """Slope chart 2016 vs 2025 per state, faceted by disease."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), sharey=False)
    for ax, virus in zip(axes, VIRUS_ORDER):
        sub = ai[ai["Virus"] == virus]
        first_year = sub["Year"].min()
        last_year = sub["Year"].max()
        first = sub[sub["Year"] == first_year].set_index("Location_Name")["Annual_Inc_100k"]
        last = sub[sub["Year"] == last_year].set_index("Location_Name")["Annual_Inc_100k"]
        common = first.index.intersection(last.index)
        sig = set(mk[(mk["Virus"] == virus) & mk["Significant"]]["Location_Name"])
        for state in common:
            y0, y1 = first[state], last[state]
            highlight = state in sig
            color = VIRUS_COLOR[virus] if highlight else "#bcbcbc"
            lw = 1.8 if highlight else 0.6
            alpha = 1.0 if highlight else 0.6
            ax.plot([0, 1], [y0, y1], color=color, lw=lw, alpha=alpha)
            ax.scatter([0, 1], [y0, y1], color=color, s=18 if highlight else 6,
                       zorder=3, alpha=alpha)
            if highlight:
                ax.text(1.02, y1, state, va="center", ha="left", fontsize=8,
                        color=color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([str(first_year), str(last_year)])
        ax.set_xlim(-0.1, 1.4)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(virus, color=VIRUS_COLOR[virus])
        ax.set_ylabel("Annual incidence per 100k" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Slope chart — first vs last year of surveillance, "
                 "coloured states are MK-significant (p<0.05)",
                 fontweight="bold")
    out = OUT / "trend01_slope_first_last.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_sen_dotbar(mk: pd.DataFrame) -> Path:
    """Sorted Sen-slope dot-and-bar per state, faceted by disease."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharex=False)
    for ax, virus in zip(axes, VIRUS_ORDER):
        sub = mk[mk["Virus"] == virus].copy().sort_values("Sen_Slope")
        y = np.arange(len(sub))
        colors = []
        for _, r in sub.iterrows():
            if not r["Significant"]:
                colors.append("#cccccc")
            elif r["Sen_Slope"] > 0:
                colors.append("#d7263d")
            else:
                colors.append("#2e86ab")
        ax.hlines(y, 0, sub["Sen_Slope"], color=colors, lw=1.6)
        ax.scatter(sub["Sen_Slope"], y, c=colors, s=28, edgecolor="white", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Location_Name"], fontsize=7.5)
        ax.set_title(virus, color=VIRUS_COLOR[virus])
        ax.set_xlabel("Sen slope (cases per 100k per year)")
        ax.grid(axis="x", alpha=0.25)
    handles = [
        Patch(color="#d7263d", label="↑ significant"),
        Patch(color="#2e86ab", label="↓ significant"),
        Patch(color="#cccccc", label="ns"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Mann-Kendall Sen slopes by state — sorted dot-and-bar",
                 fontweight="bold", y=1.06)
    out = OUT / "trend03_sen_dotbar.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_choropleth_facet(mean: pd.DataFrame) -> Path:
    """Faceted choropleth of mean annual incidence per disease."""
    gdf = load_states_gdf()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
    for ax, virus in zip(axes, VIRUS_ORDER):
        sub = mean[mean["Virus"] == virus]
        merged = gdf.merge(sub, on="Location_Name", how="left")
        cmap = LinearSegmentedColormap.from_list(
            "alt", ["#f5f5f5", VIRUS_COLOR[virus]])
        merged.plot(column="Mean_Inc", ax=ax, cmap=cmap, linewidth=0.4,
                    edgecolor="white", legend=True,
                    legend_kwds={"shrink": 0.6, "label": "Mean annual incidence /100k"},
                    missing_kwds={"color": "#e0e0e0"})
        ax.set_title(virus, color=VIRUS_COLOR[virus])
        ax.set_axis_off()
    fig.suptitle("Choropleth small-multiples — mean annual incidence per 100k",
                 fontweight="bold")
    out = OUT / "spat01_choropleth_facet.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_lisa_grid(lisa_df: pd.DataFrame) -> Path:
    """LISA cluster grid — categorical map per disease."""
    gdf = load_states_gdf()
    cat_color = {
        "HH": "#b2182b", "LL": "#2166ac", "HL": "#ef8a62", "LH": "#67a9cf",
        "NS": "#dddddd"
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
    for ax, virus in zip(axes, VIRUS_ORDER):
        sub = lisa_df[lisa_df["Virus"] == virus]
        merged = gdf.merge(sub, on="Location_Name", how="left")
        merged["color"] = merged["LISA_Cat"].map(cat_color).fillna("#dddddd")
        merged.plot(color=merged["color"], ax=ax, linewidth=0.4, edgecolor="white")
        ax.set_title(virus, color=VIRUS_COLOR[virus])
        ax.set_axis_off()
    handles = [Patch(color=c, label=k) for k, c in cat_color.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LISA cluster categories — faceted by disease",
                 fontweight="bold")
    out = OUT / "spat02_lisa_facet.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_caterpillar_per_state(per_state_nb: pd.DataFrame) -> Path:
    """Per-state Temp IRR caterpillar, sorted, faceted by disease."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharex=False)
    for ax, virus in zip(axes, VIRUS_ORDER):
        sub = per_state_nb[(per_state_nb["Virus"] == virus) &
                           (per_state_nb["Term"] == "Temp_Avg")].copy()
        sub = sub.sort_values("IRR")
        y = np.arange(len(sub))
        for i, r in enumerate(sub.itertuples()):
            color = VIRUS_COLOR[virus] if r.Significant else "#bbbbbb"
            ax.hlines(y[i], r.IRR_CI_L, r.IRR_CI_U, color=color, lw=1.4)
            ax.plot(r.IRR, y[i], "o", color=color, ms=5, markeredgecolor="white", mew=0.6)
        ax.axvline(1.0, color="black", lw=0.6, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Location_Name"], fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("Temperature IRR (95% CI) — NegBin per-state")
        ax.set_title(virus, color=VIRUS_COLOR[virus])
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Caterpillar plot — per-state temperature IRR (NegBin), sorted",
                 fontweight="bold")
    out = OUT / "glm02_caterpillar_temp_irr.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ai = annual_incidence()
    mk = mann_kendall()
    li = lisa()
    mi = mean_inc()
    per_state_nb = glm("negbin", "per_state")

    written = [
        fig_slope_chart(ai, mk),
        fig_sen_dotbar(mk),
        fig_choropleth_facet(mi),
        fig_lisa_grid(li),
        fig_caterpillar_per_state(per_state_nb),
    ]
    print(f"Wrote {len(written)} figures to {OUT}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
