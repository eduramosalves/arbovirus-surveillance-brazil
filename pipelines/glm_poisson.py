"""
================================================================================
 POISSON REGRESSION
================================================================================
 Cases ~ Climate + offset(log Population), family = Poisson, link = log.
 Three model specs (per-state, lagged, pooled with state fixed effects).

 Output:
   outputs/glm_poisson_per_state.csv
   outputs/glm_poisson_lagged.csv
   outputs/glm_poisson_pooled_fe.csv
   outputs/glm_poisson_results.xlsx
================================================================================
"""
from glm_common import run_models, export_results, summary_climate_only


def main():
    print("=" * 62)
    print("  POISSON GLM — Cases ~ Climate + offset(log Pop)")
    print("=" * 62)
    results = run_models("poisson")
    export_results("poisson", results)
    summary_climate_only(results, "poisson")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
