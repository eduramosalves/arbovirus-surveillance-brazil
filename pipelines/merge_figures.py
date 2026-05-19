"""
merge_figures.py
Combines existing pipeline PNGs into composite article figures.

Figure 1 — Temporal incidence & regional distribution
Figure 2 — Trend analysis (Mann-Kendall + Sen's slope)
Figure 3 — Climate-disease correlations (Spearman)
Figure 4 — Spatial autocorrelation (Moran scatterplot)
Figure 5 — Patient demographics (age + sex, all three diseases)
Figure 6 — Maps (choropleth mean, Jenks, temporal, LISA, trend)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

BASE    = os.path.dirname(os.path.abspath(__file__))
GP_DIR  = os.path.join(BASE, "outputs")
OUT_DIR = os.path.join(BASE, "outputs")
SAVE_TO = os.path.join(BASE, "figures")  # normalized_results/figures
os.makedirs(SAVE_TO, exist_ok=True)

DPI = 200

plt.rcParams.update({"figure.facecolor": "white"})


def load(path: str):
    return mpimg.imread(path)


def place(ax, img, label: str = None):
    """Display image on axis with optional bold letter label."""
    ax.imshow(img)
    ax.axis("off")
    if label:
        ax.text(-0.01, 1.03, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="left",
                color="black")


def save(fig, name: str):
    path = os.path.join(SAVE_TO, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {name}")


# ==============================================================================
# Figure 1: Temporal Incidence & Regional Distribution
# ==============================================================================
img_inc    = load(os.path.join(OUT_DIR, "py_01_annual_incidence.png"))
img_region = load(os.path.join(OUT_DIR, "py_03_boxplot_regional.png"))

h1, w1 = img_inc.shape[:2]
h2, w2 = img_region.shape[:2]
fw = 18
fh = fw * (h1 / w1 + h2 / w2) * 0.5 + 0.3   # approximate composite height

fig = plt.figure(figsize=(fw, fh))
gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.04,
                        height_ratios=[h1, h2])

place(fig.add_subplot(gs[0]), img_inc,    label="A")
place(fig.add_subplot(gs[1]), img_region, label="B")

save(fig, "Figure1_incidence_regional.png")

# ==============================================================================
# Figure 2: Temporal Trend Analysis
# ==============================================================================
img_tau   = load(os.path.join(OUT_DIR, "py_04_mannkendall_tau.png"))
img_slope = load(os.path.join(OUT_DIR, "py_05_sens_slope.png"))

h1, w1 = img_tau.shape[:2]
h2, w2 = img_slope.shape[:2]
fw = 18
fh = fw * (h1 / w1 + h2 / w2) * 0.5 + 0.3

fig = plt.figure(figsize=(fw, fh))
gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.04,
                        height_ratios=[h1, h2])

place(fig.add_subplot(gs[0]), img_tau,   label="A")
place(fig.add_subplot(gs[1]), img_slope, label="B")

save(fig, "Figure2_trend_analysis.png")

# ==============================================================================
# Figure 3: Climate-Disease Correlations
# ==============================================================================
img_spear = load(os.path.join(OUT_DIR, "py_06_spearman_summary.png"))
img_lag   = load(os.path.join(OUT_DIR, "py_08_lag_profile_region.png"))

h1, w1 = img_spear.shape[:2]
h2, w2 = img_lag.shape[:2]
fw = 18
fh = fw * (h1 / w1 + h2 / w2) * 0.5 + 0.3

fig = plt.figure(figsize=(fw, fh))
gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.04,
                        height_ratios=[h1, h2])

place(fig.add_subplot(gs[0]), img_spear, label="A")
place(fig.add_subplot(gs[1]), img_lag,   label="B")

save(fig, "Figure3_climate_correlations.png")

# ==============================================================================
# Figure 4: Spatial Autocorrelation (Moran scatterplot — already 3 panels)
# ==============================================================================
img_moran = load(os.path.join(OUT_DIR, "py_10_moran_scatterplot.png"))
h, w = img_moran.shape[:2]
fw = 16
fh = fw * h / w

fig = plt.figure(figsize=(fw, fh))
place(fig.add_subplot(111), img_moran)

save(fig, "Figure4_spatial_autocorrelation.png")

# ==============================================================================
# Figure 5: Patient Demographics (age distribution + sex, 3 diseases)
# ==============================================================================
diseases = ["Dengue", "Chikungunya", "Zika"]

imgs_age = [load(os.path.join(GP_DIR, f"{d}_04_age_group_violin.png"))
            for d in diseases]
imgs_sex = [load(os.path.join(GP_DIR, f"{d}_03_sex.png"))
            for d in diseases]

# Row heights proportional to the first image of each row
ha = imgs_age[0].shape[0]
hs = imgs_sex[0].shape[0]
fw = 18
fh = fw * (ha / (imgs_age[0].shape[1] * 3) + hs / (imgs_sex[0].shape[1] * 3)) + 0.5

fig = plt.figure(figsize=(fw, fh))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.03,
                        height_ratios=[ha, hs])

for col, (d, img) in enumerate(zip(diseases, imgs_age)):
    place(fig.add_subplot(gs[0, col]), img,
          label="A" if col == 0 else None)

for col, (d, img) in enumerate(zip(diseases, imgs_sex)):
    place(fig.add_subplot(gs[1, col]), img,
          label="B" if col == 0 else None)

save(fig, "Figure5_demographics.png")

# ==============================================================================
# Figure 6: Maps
#   Row A (full width) : geo_01_choropleth_mean      (3-panel, ratio 1.86)
#   Row B (full width) : py_09_map_jenks              (3-panel, ratio 2.45)
#   Row C (3 cols)     : geo_02 / geo_03 / geo_04    (temporal per disease)
#   Row D (2 cols)     : geo_05_lisa / geo_06_trend
# ==============================================================================
img_choro   = load(os.path.join(OUT_DIR, "geo_01_choropleth_mean.png"))
img_jenks   = load(os.path.join(OUT_DIR, "py_09_map_jenks.png"))
img_td      = load(os.path.join(OUT_DIR, "geo_02_temporal_dengue.png"))
img_tc      = load(os.path.join(OUT_DIR, "geo_03_temporal_chikv.png"))
img_tz      = load(os.path.join(OUT_DIR, "geo_04_temporal_zikv.png"))
img_lisa    = load(os.path.join(OUT_DIR, "geo_05_lisa_clusters.png"))
img_trend   = load(os.path.join(OUT_DIR, "geo_06_trend_map.png"))

# pixel heights of each row (used for height_ratios)
h_choro  = img_choro.shape[0]   # 1205
h_jenks  = img_jenks.shape[0]   # 1217
h_temp   = img_td.shape[0]      # 1496  (all three temporal maps same height)
h_bottom = img_lisa.shape[0]    # 1112  (LISA and trend same height)

fw = 18
fig = plt.figure(figsize=(fw, fw * (h_choro + h_jenks + h_temp + h_bottom) / img_jenks.shape[1]))

# 6-column grid lets rows C and D split cleanly (thirds vs halves)
gs = gridspec.GridSpec(4, 6, figure=fig,
                       hspace=0.04, wspace=0.02,
                       height_ratios=[h_choro, h_jenks, h_temp, h_bottom])

# Row A — choropleth mean (full width)
place(fig.add_subplot(gs[0, :]), img_choro, label="A")

# Row B — Jenks classification (full width)
place(fig.add_subplot(gs[1, :]), img_jenks, label="B")

# Row C — temporal maps (3 equal columns, each spans 2 of 6 grid cols)
place(fig.add_subplot(gs[2, 0:2]), img_td, label="C")
place(fig.add_subplot(gs[2, 2:4]), img_tc)
place(fig.add_subplot(gs[2, 4:6]), img_tz)

# Row D — LISA clusters + trend map (each spans 3 of 6 grid cols)
place(fig.add_subplot(gs[3, 0:3]), img_lisa,  label="D")
place(fig.add_subplot(gs[3, 3:6]), img_trend, label="E")

save(fig, "Figure6_maps.png")

# ==============================================================================
# Figure 7: Dengue — Diagnostic & Clinical Characteristics (2 x 2)
#   A: ELISA     B: IgM
#   C: Hosp.     D: Serotype
# ==============================================================================
imgs_d = {
    "A": load(os.path.join(GP_DIR, "Dengue_02_ELISA.png")),
    "B": load(os.path.join(GP_DIR, "Dengue_02_IgM.png")),
    "C": load(os.path.join(GP_DIR, "Dengue_02_hospitalization.png")),
    "D": load(os.path.join(GP_DIR, "Dengue_03_serotype.png")),
}

h_px, w_px = next(iter(imgs_d.values())).shape[:2]
fw = 14
fh = fw * (2 * h_px) / (2 * w_px)

fig = plt.figure(figsize=(fw, fh))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.04, wspace=0.03)
positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]

for (label, img), pos in zip(imgs_d.items(), positions):
    place(fig.add_subplot(pos), img, label=label)

save(fig, "Figure7_dengue_clinical.png")

# ==============================================================================
# Figure 8: Chikungunya & Zika — Clinical Characteristics
#   Row A (Chikungunya, 3 panels): Classification, Criteria, Evolution
#   Row B (Zika,        4 panels): Classification, Criteria, Evolution, Pregnancy
#   Uses a 12-column grid so rows can have 3 or 4 equal-width cells.
# ==============================================================================
chik_imgs = [
    ("A", load(os.path.join(GP_DIR, "Chikungunya_03_classification.png"))),
    ("",  load(os.path.join(GP_DIR, "Chikungunya_03_criteria.png"))),
    ("",  load(os.path.join(GP_DIR, "Chikungunya_03_evolution.png"))),
]
zika_imgs = [
    ("B", load(os.path.join(GP_DIR, "Zika_03_classification.png"))),
    ("",  load(os.path.join(GP_DIR, "Zika_03_criteria.png"))),
    ("",  load(os.path.join(GP_DIR, "Zika_03_evolution.png"))),
    ("",  load(os.path.join(GP_DIR, "Zika_03_pregnancy.png"))),
]

h_px, w_px = chik_imgs[0][1].shape[:2]
fw = 18
fh = fw * (2 * h_px) / (4 * w_px) + 0.4

fig = plt.figure(figsize=(fw, fh))
# 12-col grid: Chikungunya row uses 3 × 4-col cells; Zika row uses 4 × 3-col cells
gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.06, wspace=0.03)

# Row 0 — Chikungunya (3 panels × 4 cols each)
for i, (lbl, img) in enumerate(chik_imgs):
    ax = fig.add_subplot(gs[0, i*4:(i+1)*4])
    place(ax, img, label=lbl if lbl else None)

# Row label: Chikungunya
fig.text(0.01, 0.74, "Chikungunya", va="center", ha="left",
         fontsize=11, fontweight="bold", color="#E67E22", rotation=90)

# Row 1 — Zika (4 panels × 3 cols each)
for i, (lbl, img) in enumerate(zika_imgs):
    ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
    place(ax, img, label=lbl if lbl else None)

fig.text(0.01, 0.27, "Zika", va="center", ha="left",
         fontsize=11, fontweight="bold", color="#2980B9", rotation=90)

save(fig, "Figure8_chikv_zika_clinical.png")

# ==============================================================================
print(f"\nAll figures saved to: {SAVE_TO}")
