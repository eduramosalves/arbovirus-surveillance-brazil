"""
================================================================================
 BOOTSTRAP 95% CONFIDENCE INTERVALS
================================================================================
 Adds non-parametric bootstrap CIs (1000 resamples, percentile method) alongside
 the existing normal-approximation CIs.  More robust to the right-skew typical
 of count and incidence data.

 Output files (in ./outputs/):
   bootstrap_ci_95_cases.csv
   bootstrap_ci_95_incidence.csv
   ci_comparison.xlsx     (joins normal-approx and bootstrap CIs in one table)
================================================================================
"""

import os
import numpy as np
import pandas as pd

from data_loader import load_pipeline_datasets, DISEASE_KEY

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "db")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT = 1000
RNG_SEED = 42


def bootstrap_ci(values, n_boot=N_BOOT, ci=95, seed=RNG_SEED):
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan, n
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = values[idx].mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi, n


def main():
    print("=" * 62)
    print("  BOOTSTRAP 95% CIs (1000 resamples, percentile method)")
    print("=" * 62)

    datasets = load_pipeline_datasets()
    code_to_virus = DISEASE_KEY            # {"DENV":"Dengue", ...}
    virus_to_code = {v: k for k, v in code_to_virus.items()}

    rows_cases, rows_inc = [], []
    for virus, df in datasets.items():
        cases = df["Cases"].dropna()
        inc   = df["Monthly_Inc_per100k"].dropna()
        # Cases
        m_c, sd_c, n_c = cases.mean(), cases.std(ddof=1), len(cases)
        se_c = sd_c / np.sqrt(n_c)
        lo_c, hi_c, _ = bootstrap_ci(cases.values)
        rows_cases.append({
            "Disease": virus_to_code[virus],
            "Disease_Name": virus,
            "Mean": m_c,
            "Normal_CI_Lower": m_c - 1.96 * se_c,
            "Normal_CI_Upper": m_c + 1.96 * se_c,
            "Bootstrap_CI_Lower": lo_c,
            "Bootstrap_CI_Upper": hi_c,
            "n": n_c,
            "method": "bootstrap-percentile-1000",
        })
        # Incidence
        m_i, sd_i, n_i = inc.mean(), inc.std(ddof=1), len(inc)
        se_i = sd_i / np.sqrt(n_i)
        lo_i, hi_i, _ = bootstrap_ci(inc.values)
        rows_inc.append({
            "Disease": virus_to_code[virus],
            "Disease_Name": virus,
            "Mean": m_i,
            "Normal_CI_Lower": m_i - 1.96 * se_i,
            "Normal_CI_Upper": m_i + 1.96 * se_i,
            "Bootstrap_CI_Lower": lo_i,
            "Bootstrap_CI_Upper": hi_i,
            "n": n_i,
            "method": "bootstrap-percentile-1000",
        })
        print(f"  [{virus:12s}] cases mean={m_c:.2f} "
              f"normal=({m_c-1.96*se_c:.2f},{m_c+1.96*se_c:.2f}) "
              f"bootstrap=({lo_c:.2f},{hi_c:.2f})")

    df_cases = pd.DataFrame(rows_cases)
    df_inc   = pd.DataFrame(rows_inc)

    # Write CSV outputs (mirror the schema of the original ci_95_*.csv files
    # but include both methods side-by-side for comparison)
    out_cases_path = os.path.join(OUT_DIR, "bootstrap_ci_95_cases.csv")
    out_inc_path   = os.path.join(OUT_DIR, "bootstrap_ci_95_incidence.csv")
    df_cases.to_csv(out_cases_path, index=False)
    df_inc.to_csv(out_inc_path, index=False)
    print(f"\n  Saved -> {os.path.basename(out_cases_path)}")
    print(f"  Saved -> {os.path.basename(out_inc_path)}")

    # Combined Excel for review
    xlsx_path = os.path.join(OUT_DIR, "ci_comparison.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df_cases.to_excel(w, sheet_name="Cases", index=False)
        df_inc.to_excel(w, sheet_name="Incidence_per100k", index=False)
    print(f"  Saved -> {os.path.basename(xlsx_path)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
