"""
================================================================================
 QUASI-POISSON REGRESSION
================================================================================
 Same point estimates as Poisson, but with SEs inflated by sqrt(dispersion)
 where dispersion = Pearson chi-squared / residual df.  This handles
 over-dispersion without specifying a different distribution.

 Equivalent to R's family = quasipoisson().

 Output:
   outputs/glm_quasi_poisson_per_state.csv
   outputs/glm_quasi_poisson_lagged.csv
   outputs/glm_quasi_poisson_pooled_fe.csv
   outputs/glm_quasi_poisson_results.xlsx
================================================================================
"""
from glm_common import run_models, export_results, summary_climate_only


def main():
    print("=" * 62)
    print("  QUASI-POISSON GLM — Poisson with Pearson dispersion correction")
    print("=" * 62)
    results = run_models("quasi-poisson")
    export_results("quasi-poisson", results)
    summary_climate_only(results, "quasi-poisson")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
