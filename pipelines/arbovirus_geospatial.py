"""
================================================================================
 GEOSPATIAL ANALYSIS — NORMALIZED DB version
================================================================================
 Identical methodology to ../arbovirus_geospatial.py but reads the normalized
 CSV schema and writes outputs to ./outputs/.
================================================================================
"""

import os, sys, warnings, io
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
import requests
import pymannkendall as mk
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
import mapclassify

from data_loader import load_pipeline_datasets, UF_REGION

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
GEO_CACHE  = os.path.join(HERE, "brazil_states.geojson")
# Fall back to parent project cache if not present locally
if not os.path.exists(GEO_CACHE):
    parent_cache = os.path.join(os.path.dirname(HERE), "brazil_states.geojson")
    if os.path.exists(parent_cache):
        GEO_CACHE = parent_cache
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05

COLORS = {"Dengue": "#C0392B", "Chikungunya": "#E67E22", "Zika": "#2980B9"}
CMAPS  = {"Dengue": "Reds",    "Chikungunya": "Oranges", "Zika":  "Blues"}

IBGE_CODE_TO_NAME = {
    "11": "Rondonia",     "12": "Acre",               "13": "Amazonas",
    "14": "Roraima",      "15": "Para",                "16": "Amapa",
    "17": "Tocantins",    "21": "Maranhao",            "22": "Piaui",
    "23": "Ceara",        "24": "Rio Grande do Norte", "25": "Paraiba",
    "26": "Pernambuco",   "27": "Alagoas",             "28": "Sergipe",
    "29": "Bahia",        "31": "Minas Gerais",        "32": "Espirito Santo",
    "33": "Rio de Janeiro", "35": "Sao Paulo",         "41": "Parana",
    "42": "Santa Catarina", "43": "Rio Grande do Sul", "50": "Mato Grosso do Sul",
    "51": "Mato Grosso",  "52": "Goias",              "53": "Distrito Federal",
}

NAME_TO_ABBR = {
    "Rondonia": "RO",     "Acre": "AC",          "Amazonas": "AM",
    "Roraima": "RR",      "Para": "PA",           "Amapa": "AP",
    "Tocantins": "TO",    "Maranhao": "MA",       "Piaui": "PI",
    "Ceara": "CE",        "Rio Grande do Norte": "RN", "Paraiba": "PB",
    "Pernambuco": "PE",   "Alagoas": "AL",        "Sergipe": "SE",
    "Bahia": "BA",        "Minas Gerais": "MG",   "Espirito Santo": "ES",
    "Rio de Janeiro": "RJ", "Sao Paulo": "SP",    "Parana": "PR",
    "Santa Catarina": "SC", "Rio Grande do Sul": "RS",
    "Mato Grosso do Sul": "MS", "Mato Grosso": "MT",
    "Goias": "GO",        "Distrito Federal": "DF",
}

LISA_COLORS = {"HH": "#D7191C", "LL": "#2C7BB6",
               "HL": "#FDAE61", "LH": "#ABD9E9", "NS": "#EEEEEE"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"figure.facecolor":"white"})
XLIM = (-74, -33); YLIM = (-34, 6)


def _fetch_state(code):
    url = (f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{code}"
           f"?formato=application/vnd.geo+json&qualidade=intermediaria")
    resp = requests.get(url, timeout=30); resp.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(resp.content)); gdf["codarea"] = code
    return gdf


def _attach_metadata(gdf):
    gdf["Location_Name"] = gdf["codarea"].astype(str).map(IBGE_CODE_TO_NAME)
    gdf = gdf[gdf["Location_Name"].notna()].copy()
    gdf = gdf.dissolve(by="Location_Name", as_index=False)
    gdf["Region"]     = gdf["Location_Name"].map(UF_REGION)
    gdf["Abbr"]       = gdf["Location_Name"].map(NAME_TO_ABBR)
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y
    return gdf.reset_index(drop=True)


def get_brazil_states():
    if os.path.exists(GEO_CACHE):
        gdf = gpd.read_file(GEO_CACHE)
        if len(gdf) >= 26:
            print(f"  Loaded {len(gdf)} states from cache ({os.path.basename(GEO_CACHE)})")
            return _attach_metadata(gdf)
        print(f"  Cache invalid ({len(gdf)} rows), re-downloading...")
        os.remove(GEO_CACHE)
    codes = list(IBGE_CODE_TO_NAME.keys())
    frames = [None] * len(codes)
    print(f"  Downloading {len(codes)} state boundaries from IBGE...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_state, c): i for i, c in enumerate(codes)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                frames[idx] = fut.result(); done += 1
                if done % 5 == 0 or done == len(codes):
                    print(f"    {done}/{len(codes)} done")
            except Exception as exc:
                print(f"    FAILED {codes[idx]}: {exc}")
    frames = [f for f in frames if f is not None]
    if len(frames) < 26:
        raise RuntimeError(f"Only {len(frames)}/27 states downloaded")
    gdf = pd.concat(frames, ignore_index=True)
    gdf.to_file(GEO_CACHE, driver="GeoJSON")
    print(f"  Cached -> {os.path.basename(GEO_CACHE)}")
    return _attach_metadata(gdf)


def compute_annual(datasets):
    annual = {}
    for virus, df in datasets.items():
        ann = (df.groupby(["Location_Name","UF_Code","Region","Year"])
                 .agg(Total_Cases=("Cases","sum"), Population=("Population","first"))
                 .reset_index())
        ann["Annual_Inc_100k"] = (ann["Total_Cases"] / ann["Population"]) * 1e5
        annual[virus] = ann
    return annual


def compute_mean_inc(annual):
    mean_inc = {}
    for virus, ann in annual.items():
        mi = (ann.groupby("Location_Name")
                 .apply(lambda g: np.average(g["Annual_Inc_100k"], weights=g["Population"]))
                 .reset_index(name="Mean_Inc"))
        mean_inc[virus] = mi
    return mean_inc


def build_weights(gdf, k_fallback=5):
    try:
        w = Queen.from_dataframe(gdf, silence_warnings=True)
        islands = [i for i, nb in w.neighbors.items() if len(nb)==0]
        if islands: raise ValueError(f"{len(islands)} islands → KNN")
        w.transform = "r"
        print(f"  Queen weights: {w.n} units, avg {w.mean_neighbors:.1f} neighbours")
        return w
    except Exception as exc:
        print(f"  Queen failed ({exc}), using KNN(k={k_fallback})")
        coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        w = KNN.from_array(coords, k=k_fallback); w.transform = "r"
        return w


def run_global_moran(gdf_states, mean_inc):
    results = {}
    for virus, mi in mean_inc.items():
        gdf = gdf_states.merge(mi, on="Location_Name", how="left").dropna(subset=["Mean_Inc"])
        coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        w = KNN.from_array(coords, k=5); w.transform = "r"
        m = Moran(gdf["Mean_Inc"].values, w, permutations=999)
        results[virus] = {"I":m.I,"EI":m.EI,"z_sim":m.z_sim,
                          "p_sim":m.p_sim,"Significant":m.p_sim<ALPHA}
        print(f"  [{virus:12s}] I={m.I:.4f} p={m.p_sim:.4f} "
              f"{'Significant' if m.p_sim<ALPHA else 'NS'}")
    return results


def run_lisa(gdf_states, mean_inc):
    lisa = {}
    for virus, mi in mean_inc.items():
        gdf = gdf_states.merge(mi, on="Location_Name", how="left").copy()
        gdf["Mean_Inc"] = gdf["Mean_Inc"].fillna(0)
        coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        w = KNN.from_array(coords, k=5); w.transform = "r"
        lm = Moran_Local(gdf["Mean_Inc"].values, w, permutations=999)
        q_map = {1:"HH",2:"LH",3:"LL",4:"HL"}
        cats = [q_map.get(lm.q[i],"NS") if lm.p_sim[i]<ALPHA else "NS"
                for i in range(len(gdf))]
        gdf["LISA_Cat"] = cats; gdf["Local_I"] = lm.Is
        gdf["LISA_p"] = lm.p_sim; gdf["LISA_q"] = lm.q
        lisa[virus] = gdf
        counts = pd.Series(cats).value_counts().to_dict()
        print(f"  [{virus:12s}] " + " | ".join(f"{k}={v}" for k,v in sorted(counts.items())))
    return lisa


def run_mann_kendall(annual):
    results = {}
    for virus, ann in annual.items():
        rows = []
        for uf, grp in ann.groupby("Location_Name"):
            series = grp.sort_values("Year")["Annual_Inc_100k"].values
            if len(series) < 4: continue
            res = mk.original_test(series, alpha=ALPHA)
            rows.append({"Location_Name":uf,"Trend":res.trend,"Tau":res.Tau,
                         "p_value":res.p,"Sen_Slope":res.slope,
                         "Significant":res.p < ALPHA})
        results[virus] = pd.DataFrame(rows)
    return results


def _set_extent(ax):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_axis_off()


def _label_states(ax, gdf, fontsize=4.5):
    for _, row in gdf.iterrows():
        if pd.notna(row.get("Abbr")):
            ax.annotate(row["Abbr"], xy=(row["centroid_x"], row["centroid_y"]),
                        fontsize=fontsize, ha="center", va="center",
                        fontweight="bold", color="black", zorder=10)


def _axes_flat(axes):
    return np.array(axes).flatten().tolist()


def plot_choropleth_mean(gdf_states, mean_inc, moran_global):
    fig, axes = plt.subplots(1, 3, figsize=(19, 9.0), gridspec_kw={"wspace":0.04})
    fig.suptitle("Mean Annual Incidence per 100,000 Inhabitants\n"
                 "(population-weighted mean across all study years)",
                 fontsize=13, fontweight="bold")
    for ax, (virus, mi) in zip(axes, mean_inc.items()):
        gdf = gdf_states.merge(mi, on="Location_Name", how="left")
        gdf.plot(column="Mean_Inc", ax=ax, cmap=CMAPS[virus],
                 scheme="NaturalBreaks", k=5,
                 missing_kwds={"color":"#DDDDDD","label":"No data"},
                 edgecolor="#555555", linewidth=0.35, legend=True,
                 legend_kwds={"fontsize":7,"title":"Inc./100k","title_fontsize":7.5,
                              "loc":"upper center","bbox_to_anchor":(0.5,-0.02)})
        _label_states(ax, gdf); _set_extent(ax)
        mr = moran_global.get(virus, {})
        note = (f"Global Moran's I = {mr['I']:.3f}  (p = {mr['p_sim']:.3f})" if mr else "")
        ax.set_title(f"{virus}\n{note}", fontsize=11, fontweight="bold", color=COLORS[virus])
    plt.tight_layout(rect=[0,0.18,1,1])
    fig.savefig(os.path.join(OUTPUT_DIR, "geo_01_choropleth_mean.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> geo_01_choropleth_mean.png")


def plot_temporal_evolution(gdf_states, annual):
    slugs = {"Dengue":("dengue","02"),"Chikungunya":("chikv","03"),"Zika":("zikv","04")}
    for virus, ann in annual.items():
        slug, num = slugs[virus]
        years = sorted(ann["Year"].unique()); n = len(years)
        ncols = 4; nrows = (n+ncols-1)//ncols
        vmax = ann["Annual_Inc_100k"].quantile(0.97)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.0, nrows*3.8))
        flat = _axes_flat(axes)
        fig.suptitle(f"{virus} — Annual Incidence per 100,000 by State",
                     fontsize=13, fontweight="bold")
        for i, yr in enumerate(years):
            ax = flat[i]
            sub = ann[ann["Year"]==yr][["Location_Name","Annual_Inc_100k"]]
            gdf = gdf_states.merge(sub.rename(columns={"Annual_Inc_100k":"Inc"}),
                                   on="Location_Name", how="left")
            gdf.plot(column="Inc", ax=ax, cmap=CMAPS[virus], vmin=0, vmax=vmax,
                     missing_kwds={"color":"#DDDDDD"}, edgecolor="#666666",
                     linewidth=0.25, legend=False)
            _label_states(ax, gdf); _set_extent(ax)
            ax.set_title(str(yr), fontsize=9, fontweight="bold", pad=2)
        for j in range(n, len(flat)):
            flat[j].set_visible(False)
        sm = ScalarMappable(norm=Normalize(0, vmax), cmap=CMAPS[virus])
        sm.set_array([])
        plt.tight_layout(rect=[0,0.07,1,1])
        cax = fig.add_axes([0.52, 0.02, 0.42, 0.022])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Incidence / 100k", fontsize=9)
        cb.ax.tick_params(labelsize=7)
        fname = f"geo_{num}_temporal_{slug}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"    Saved -> {fname}")


def plot_lisa_clusters(lisa_results):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), gridspec_kw={"wspace":0.04})
    fig.suptitle("LISA Cluster Maps — Local Moran's I  (999 permutations, alpha = 0.05)\n"
                 "HH = high-high cluster  |  LL = low-low cluster  |  "
                 "HL/LH = spatial outliers  |  NS = not significant",
                 fontsize=11, fontweight="bold")
    legend_patches = [
        mpatches.Patch(color=LISA_COLORS["HH"], label="HH — High cluster"),
        mpatches.Patch(color=LISA_COLORS["LL"], label="LL — Low cluster"),
        mpatches.Patch(color=LISA_COLORS["HL"], label="HL — High outlier"),
        mpatches.Patch(color=LISA_COLORS["LH"], label="LH — Low outlier"),
        mpatches.Patch(color=LISA_COLORS["NS"], label="NS — Not significant"),
    ]
    for ax, (virus, gdf) in zip(axes, lisa_results.items()):
        gdf = gdf.copy(); gdf["color"] = gdf["LISA_Cat"].map(LISA_COLORS)
        gdf.plot(ax=ax, color=gdf["color"].tolist(), edgecolor="#555555", linewidth=0.35)
        _label_states(ax, gdf); _set_extent(ax)
        ax.set_title(virus, fontsize=11, fontweight="bold", color=COLORS[virus])
    plt.tight_layout(rect=[0,0.1,1,1])
    fig.legend(handles=legend_patches, loc="lower right", bbox_to_anchor=(0.99,0.01),
               fontsize=8, frameon=True, title="LISA Category", title_fontsize=8.5)
    fig.savefig(os.path.join(OUTPUT_DIR, "geo_05_lisa_clusters.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> geo_05_lisa_clusters.png")


def plot_trend_map(gdf_states, mk_results):
    TREND_PALETTE = {
        ("increasing", True):  ("#B2182B","Increasing (sig.)"),
        ("decreasing", True):  ("#2166AC","Decreasing (sig.)"),
        ("increasing", False): ("#FDDBC7","Increasing (NS)"),
        ("decreasing", False): ("#D1E5F0","Decreasing (NS)"),
        ("no trend",   False): ("#CCCCCC","No trend"),
        ("no trend",   True):  ("#CCCCCC","No trend"),
    }
    def _color(trend, sig):
        return TREND_PALETTE.get((str(trend).lower(), bool(sig)), ("#CCCCCC","No trend"))[0]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), gridspec_kw={"wspace":0.04})
    fig.suptitle("Mann-Kendall Trend Classification by State\n"
                 "(solid = p < 0.05  |  pale = not significant)",
                 fontsize=12, fontweight="bold")
    for ax, (virus, mk_df) in zip(axes, mk_results.items()):
        gdf = gdf_states.merge(mk_df[["Location_Name","Trend","Significant"]],
                               on="Location_Name", how="left")
        gdf["Trend"] = gdf["Trend"].fillna("no trend")
        gdf["Significant"] = gdf["Significant"].fillna(False)
        gdf["color"] = [_color(t,s) for t,s in zip(gdf["Trend"], gdf["Significant"])]
        gdf.plot(ax=ax, color=gdf["color"].tolist(), edgecolor="#555555", linewidth=0.35)
        _label_states(ax, gdf); _set_extent(ax)
        ax.set_title(virus, fontsize=11, fontweight="bold", color=COLORS[virus])
    seen, handles = set(), []
    for (_,_), (col, label) in TREND_PALETTE.items():
        if label not in seen:
            handles.append(mpatches.Patch(color=col, label=label))
            seen.add(label)
    plt.tight_layout(rect=[0,0.1,1,1])
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99,0.01),
               fontsize=8, frameon=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "geo_06_trend_map.png"),
                dpi=150, bbox_inches="tight"); plt.close()
    print("    Saved -> geo_06_trend_map.png")


def export_stats(mean_inc, mk_results, lisa_results, moran_global):
    path = os.path.join(OUTPUT_DIR, "geo_spatial_stats.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.concat([df.assign(Virus=v) for v,df in mean_inc.items()], ignore_index=True
        ).to_excel(writer, sheet_name="Mean_Incidence", index=False)
        pd.concat([df.assign(Virus=v) for v,df in mk_results.items()], ignore_index=True
        ).to_excel(writer, sheet_name="Mann_Kendall", index=False)
        lisa_rows = []
        for virus, gdf in lisa_results.items():
            cols = ["Location_Name","Region","Abbr","Mean_Inc","LISA_Cat","Local_I","LISA_p","LISA_q"]
            sub = gdf[[c for c in cols if c in gdf.columns]].copy()
            sub["Virus"] = virus; lisa_rows.append(sub)
        pd.concat(lisa_rows, ignore_index=True).to_excel(
            writer, sheet_name="LISA_Clusters", index=False)
        pd.DataFrame([{"Virus":v, **res} for v,res in moran_global.items()]
        ).to_excel(writer, sheet_name="Global_Moran", index=False)
    print(f"    Saved -> geo_spatial_stats.xlsx")


def main():
    sep = "=" * 62
    print(f"\n{sep}\n  GEOSPATIAL ANALYSIS — NORMALIZED DB version\n{sep}")

    print("\n[1/6] State boundaries")
    gdf_states = get_brazil_states()

    print("\n[2/6] Data loading & annual aggregation")
    datasets = load_pipeline_datasets()
    annual = compute_annual(datasets); mean_inc = compute_mean_inc(annual)

    print("\n[3/6] Global Moran's I")
    moran_global = run_global_moran(gdf_states, mean_inc)

    print("\n[4/6] Local Moran's I (LISA)")
    lisa_results = run_lisa(gdf_states, mean_inc)

    print("\n[5/6] Mann-Kendall per state")
    mk_results = run_mann_kendall(annual)

    print("\n[6/6] Generating maps")
    plot_choropleth_mean(gdf_states, mean_inc, moran_global)
    plot_temporal_evolution(gdf_states, annual)
    plot_lisa_clusters(lisa_results)
    plot_trend_map(gdf_states, mk_results)

    print("\n  Exporting...")
    export_stats(mean_inc, mk_results, lisa_results, moran_global)
    print(f"\n{sep}\n  Done! -> {OUTPUT_DIR}\n{sep}\n")


if __name__ == "__main__":
    main()
