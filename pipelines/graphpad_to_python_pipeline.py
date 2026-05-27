"""
================================================================================
 GRAPHPAD PRISM (.pzfx) -> PYTHON PLOTTING PIPELINE
================================================================================
 Reads the three GraphPad Prism files for Brazilian arbovirus surveillance
 (Dengue 2013–2025, Chikungunya 2017–2025, Zika 2016–2025) and reproduces the
 same set of graphs in Python with GraphPad-Prism-like styling.

 Plot recipes per table type
 ---------------------------
   Monthly YYYY-YYYY              -> grouped bar chart  (Month × Year)
                                  + accompanying heat-map
   Sex / ELISA / IgM / Hosp /
   Serotype / Evolution /
   Criteria / Classification /
   Pregnancy                      -> 100 %-stacked bar chart over the years
                                  + absolute-counts grouped bar chart
   Age group YYYY-YYYY            -> stacked bar chart over the years
   Federative unit YYYY-YYYY      -> horizontal bar chart of cumulative totals
                                    (colored by Brazilian macro-region)
   General Data (Dengue only)     -> exported to CSV (data dictionary only)

 Output: PNG files in /mnt/user-data/outputs, all prefixed with the virus.
================================================================================
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(_HERE, "db", "graphpad_data")
OUTPUT_DIR = os.path.join(_HERE, "outputs")  # normalized_results/outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "Dengue"      : "GPad_Dengue_analysis.pzfx",
    "Chikungunya" : "GPad_chikungunya_analysis.pzfx",
    "Zika"        : "GPad_zika_analysis.pzfx",
}

# Virus signature colors (consistent with the previous epidemiological pipeline)
VIRUS_COLOR = {"Dengue": "#C0392B", "Chikungunya": "#E67E22", "Zika": "#2980B9"}

# Brazilian macro-regions for the Federative-unit plots
UF_REGION = {
    "Rondonia":"Norte","Acre":"Norte","Amazonas":"Norte","Roraima":"Norte",
    "Para":"Norte","Amapa":"Norte","Tocantins":"Norte",
    "Maranhao":"Nordeste","Piaui":"Nordeste","Ceara":"Nordeste",
    "Rio Grande do Norte":"Nordeste","Paraiba":"Nordeste","Pernambuco":"Nordeste",
    "Alagoas":"Nordeste","Sergipe":"Nordeste","Bahia":"Nordeste",
    "Minas Gerais":"Sudeste","Espirito Santo":"Sudeste",
    "Rio de Janeiro":"Sudeste","Sao Paulo":"Sudeste",
    "Parana":"Sul","Santa Catarina":"Sul","Rio Grande do Sul":"Sul",
    "Mato Grosso do Sul":"Centro-Oeste","Mato Grosso":"Centro-Oeste",
    "Goias":"Centro-Oeste","Distrito Federal":"Centro-Oeste",
}
REGION_COLOR = {
    "Norte":"#1B7837", "Nordeste":"#762A83", "Sudeste":"#E08214",
    "Sul":"#4393C3", "Centro-Oeste":"#D6604D",
}

# ── GraphPad-Prism look ───────────────────────────────────────────────────────
def set_prism_style():
    """Apply a clean GraphPad-Prism-like matplotlib style."""
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "font.size"        : 11,
        "axes.titlesize"   : 13,
        "axes.titleweight" : "bold",
        "axes.labelsize"   : 11,
        "axes.labelweight" : "bold",
        "axes.linewidth"   : 1.4,
        "axes.edgecolor"   : "#000000",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "xtick.major.size" : 5,
        "xtick.major.width": 1.4,
        "xtick.direction"  : "out",
        "ytick.major.size" : 5,
        "ytick.major.width": 1.4,
        "ytick.direction"  : "out",
        "legend.frameon"   : False,
        "legend.fontsize"  : 9,
        "figure.facecolor" : "white",
        "savefig.dpi"      : 150,
        "savefig.bbox"     : "tight",
    })


# ==============================================================================
# 1. PARSE THE .pzfx FILES
# ==============================================================================
def parse_pzfx(path: str) -> List[Dict]:
    """Parse a GraphPad Prism .pzfx file into a list of table dicts."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns_match = re.match(r"\{.*\}", root.tag)
    ns = ns_match.group(0) if ns_match else ""

    tables = []
    for t in root.findall(f"{ns}Table"):
        title_el = t.find(f"{ns}Title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        # Row labels (years or categories on the X-axis)
        row_titles: List[str] = []
        rt = t.find(f"{ns}RowTitlesColumn")
        if rt is not None:
            sub = rt.find(f"{ns}Subcolumn")
            if sub is not None:
                row_titles = [(d.text or "").strip()
                              for d in sub.findall(f"{ns}d")]

        # YColumns (one per series)
        columns = []
        for yc in t.findall(f"{ns}YColumn"):
            tt = yc.find(f"{ns}Title")
            col_title = (tt.text or "").strip() if tt is not None else ""
            sub = yc.find(f"{ns}Subcolumn")
            values: List[float] = []
            if sub is not None:
                for d in sub.findall(f"{ns}d"):
                    txt = (d.text or "").strip()
                    try:
                        values.append(float(txt))
                    except ValueError:
                        values.append(np.nan)
            columns.append({"title": col_title, "values": values})

        tables.append({
            "id"        : t.get("ID"),
            "title"     : title,
            "row_titles": row_titles,
            "columns"   : columns,
        })
    return tables


def table_to_df(tbl: Dict) -> pd.DataFrame:
    """Convert a parsed table into a DataFrame: rows = row_titles, cols = series."""
    data = {c["title"]: c["values"] for c in tbl["columns"]}
    df = pd.DataFrame(data, index=tbl["row_titles"])
    df.index.name = "Year"
    return df


# ==============================================================================
# 2. PLOT RECIPES
# ==============================================================================
def _save(fig, fname: str) -> None:
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out)
    plt.close(fig)
    print(f"    Saved -> {fname}")


def plot_monthly(df: pd.DataFrame, virus: str, table_title: str) -> None:
    """Heatmap of monthly cases (years × months)."""
    month_order = ["JAN","FEV","MAR","APR","MAY","JUN",
                   "JUL","AUG","SEP","OCT","NOV","DEC"]
    df = df[[m for m in month_order if m in df.columns]]
    years = df.index.tolist()
    months = df.columns.tolist()

    # ── Heat-map ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(years) + 2))
    # Build a virus-colored colormap that runs from white -> virus color
    cmap_v = LinearSegmentedColormap.from_list(
        f"{virus}_cmap", ["#FFFFFF", VIRUS_COLOR[virus]], N=256)
    im = ax.imshow(df.values, aspect="auto", cmap=cmap_v)
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months)
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(f"{virus} — {table_title} (heat-map)",
                 color=VIRUS_COLOR[virus])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cases", rotation=90)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}".replace(",", " ")))
    # Annotate each cell when the matrix is reasonably small
    if df.size <= 13 * 12:
        vmax = np.nanmax(df.values)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i, j]
                if not np.isnan(v) and v > 0:
                    color = "white" if v > 0.55 * vmax else "black"
                    ax.text(j, i, f"{int(v):,}".replace(",", " "),
                            ha="center", va="center",
                            color=color, fontsize=7)
    _save(fig, f"{virus}_01b_monthly_heatmap.png")


def plot_stacked_and_grouped(df: pd.DataFrame, virus: str,
                             table_title: str, slug: str) -> None:
    """100%-stacked bar chart (proportions per year)."""
    years = df.index.tolist()
    cats = df.columns.tolist()
    palette = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20",
                           len(cats))
    colors = [palette(i) for i in range(len(cats))]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    totals = df.sum(axis=1).replace(0, np.nan)
    pct = df.div(totals, axis=0).fillna(0) * 100
    bottom = np.zeros(len(years))
    x = np.arange(len(years))
    for i, c in enumerate(cats):
        ax.bar(x, pct[c].values, bottom=bottom, color=colors[i],
               label=c, edgecolor="white", linewidth=0.4, width=0.78)
        bottom += pct[c].values
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{virus} — {table_title}",
                 color=VIRUS_COLOR[virus], fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, title=table_title.split()[0])

    fig.tight_layout()
    _save(fig, f"{virus}_{slug}.png")


# Numeric ranges for each age bin (used to reconstruct a continuous age axis
# from binned counts).  "<1 year" -> [0,1); "80 e +" capped at 100.
AGE_BIN_RANGES = {
    "<1 year": (0.0,   1.0),
    "1-4"    : (1.0,   5.0),
    "5-9"    : (5.0,  10.0),
    "10-14"  : (10.0, 15.0),
    "15-19"  : (15.0, 20.0),
    "20-39"  : (20.0, 40.0),
    "40-59"  : (40.0, 60.0),
    "60-64"  : (60.0, 65.0),
    "65-69"  : (65.0, 70.0),
    "70-79"  : (70.0, 80.0),
    "80 e +" : (80.0, 100.0),
}
N_AGE_SAMPLES = 6000   # samples per year used to build the violin


def plot_age_group(df: pd.DataFrame, virus: str, table_title: str) -> None:
    """Violin plot: reconstructed age distribution per year.

    The Prism table reports the number of cases per age *bin* per year.
    To draw a violin, for each year we draw ``N_AGE_SAMPLES`` synthetic ages:
    each sample is assigned to a bin with probability proportional to that
    bin's case count, then jittered uniformly inside the bin's age range.
    The resulting violin width at age *a* is therefore proportional to the
    number of cases of age *a* reported that year.
    'Ign/Blank' is dropped because the age is unknown.
    """
    rng = np.random.default_rng(seed=42)
    bins = list(AGE_BIN_RANGES.keys())
    ranges = list(AGE_BIN_RANGES.values())
    # Restrict to bins actually present in this dataframe
    available = [b for b in bins if b in df.columns]
    if not available:
        print(f"    (no age bins found, skipping {table_title})")
        return
    ranges = [AGE_BIN_RANGES[b] for b in available]

    records = []
    medians = {}
    for year in df.index:
        weights = df.loc[year, available].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        total = weights.sum()
        if total <= 0:
            continue
        probs = weights / total
        bin_idx = rng.choice(len(available), size=N_AGE_SAMPLES, p=probs)
        lows  = np.array([ranges[i][0] for i in bin_idx])
        highs = np.array([ranges[i][1] for i in bin_idx])
        ages = rng.uniform(lows, highs)
        medians[year] = float(np.median(ages))
        for a in ages:
            records.append({"Year": str(year), "Age": a})

    plot_df = pd.DataFrame.from_records(records)
    if plot_df.empty:
        print(f"    (no data, skipping {table_title})")
        return

    # Preserve year order on the X-axis
    year_order = [str(y) for y in df.index if str(y) in medians]

    fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(year_order) + 3), 6))

    # Violins
    import seaborn as sns
    sns.violinplot(
        data=plot_df, x="Year", y="Age",
        order=year_order, ax=ax,
        color=VIRUS_COLOR[virus], inner="quartile",
        cut=0, bw_adjust=0.6, linewidth=1.0, saturation=0.85,
    )

    # Median markers (white dots) and number-of-reported-cases above each violin
    for i, yr in enumerate(year_order):
        ax.scatter(i, medians[yr], s=22, color="white",
                   edgecolor="black", linewidth=0.9, zorder=5)
        n_cases = int(df.loc[df.index[df.index.astype(str) == yr][0],
                             available].sum())
        ax.text(i, 102, f"n = {n_cases:,}".replace(",", " "),
                ha="center", va="bottom", fontsize=7.5,
                color="#555555", rotation=0)

    ax.set_ylim(0, 108)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Year")
    ax.set_ylabel("Patient age (years)")
    ax.set_title(f"{virus} — {table_title}", color=VIRUS_COLOR[virus])
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Footer legend explaining the reconstruction
    fig.text(0.01, -0.02,
             "Note: ages reconstructed from binned counts (jittered uniformly "
             "inside each bin).\nInner lines = quartiles.  White dot = median.",
             fontsize=7.5, color="#666666", ha="left")

    _save(fig, f"{virus}_04_age_group_violin.png")


def plot_federative_unit(df: pd.DataFrame, virus: str,
                         table_title: str) -> None:
    """Horizontal bars of cumulative totals per state, colored by macro-region."""
    # Drop the 'Ign/exterior' column if it exists (uninformative)
    cols = [c for c in df.columns if not c.startswith("FU - ")]
    df = df[cols]
    totals = df.sum(axis=0).sort_values(ascending=True)

    regions = [UF_REGION.get(s, "Other") for s in totals.index]
    colors = [REGION_COLOR.get(r, "#888888") for r in regions]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.barh(totals.index, totals.values, color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Total reported cases (period)")
    ax.set_title(f"{virus} — {table_title}", color=VIRUS_COLOR[virus])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}".replace(",", " ")))
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    # Annotate counts at the end of each bar
    xmax = totals.values.max()
    for s, v in totals.items():
        ax.text(v + 0.01 * xmax, s, f"{int(v):,}".replace(",", " "),
                va="center", fontsize=8)
    ax.set_xlim(0, xmax * 1.15)
    # Region legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=REGION_COLOR[r], label=r)
                    for r in REGION_COLOR]
    ax.legend(handles=legend_items, title="Region",
              loc="lower right", fontsize=8)
    _save(fig, f"{virus}_05_federative_unit_totals.png")


# ==============================================================================
# 3. DISPATCH PER TABLE
# ==============================================================================
PLOT_SLUGS = {
    "ELISA"          : "02_ELISA",
    "IgM"            : "02_IgM",
    "Hospitalization": "02_hospitalization",
    "Sex"            : "03_sex",
    "Serotype"       : "03_serotype",
    "Evolution"      : "03_evolution",
    "Criteria"       : "03_criteria",
    "Classification" : "03_classification",
    "Classfication"  : "03_classification",  # Prism file typo
    "Pregnancy"      : "03_pregnancy",
}


def dispatch_table(tbl: Dict, virus: str) -> None:
    title = tbl["title"]
    print(f"  • {title}  ({len(tbl['row_titles'])} rows × "
          f"{len(tbl['columns'])} cols)")

    if title == "General Data":
        # Export the raw master table to CSV; no single graph would be readable
        df = table_to_df(tbl)
        out = os.path.join(OUTPUT_DIR, f"{virus}_00_general_data.csv")
        df.to_csv(out)
        print(f"    Exported -> {os.path.basename(out)}  "
              f"(74-column master sheet)")
        return

    df = table_to_df(tbl)

    if title.startswith("Monthly"):
        plot_monthly(df, virus, title)

    elif title.startswith("Age group"):
        plot_age_group(df, virus, title)

    elif title.startswith("Federative unit"):
        plot_federative_unit(df, virus, title)

    else:
        # Generic categorical table (ELISA, IgM, Sex, Serotype, Evolution, ...)
        key = title.split()[0]
        slug = PLOT_SLUGS.get(key, "03_" + key.lower())
        plot_stacked_and_grouped(df, virus, title, slug)


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    set_prism_style()
    sep = "=" * 70
    print(f"\n{sep}")
    print("  GRAPHPAD PRISM (.pzfx) -> PYTHON PLOTTING PIPELINE")
    print(f"{sep}")

    for virus, fname in FILES.items():
        path = os.path.join(INPUT_DIR, fname)
        print(f"\n[{virus}] reading {fname}")
        tables = parse_pzfx(path)
        print(f"  Found {len(tables)} tables")
        for tbl in tables:
            dispatch_table(tbl, virus)

    print(f"\n{sep}")
    print(f"  Done. All outputs written to -> {OUTPUT_DIR}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
