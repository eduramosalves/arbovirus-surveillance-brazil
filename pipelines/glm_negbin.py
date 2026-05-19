"""
================================================================================
 NEGATIVE BINOMIAL REGRESSION (NB2 parameterization)
================================================================================
 Cases ~ Climate + offset(log Population), family = NegativeBinomial.
 Handles over-dispersion via a quadratic mean-variance relationship:
   Var(Y) = mu + alpha * mu^2
 where alpha > 0 is the dispersion parameter estimated by MLE.

 Output:
   outputs/glm_negbin_per_state.csv
   outputs/glm_negbin_lagged.csv
   outputs/glm_negbin_pooled_fe.csv
   outputs/glm_negbin_results.xlsx
================================================================================
"""
from glm_common import run_models, export_results, summary_climate_only


def main():
    print("=" * 62)
    print("  NEGATIVE BINOMIAL GLM — Cases ~ Climate + offset(log Pop)")
    print("=" * 62)
    results = run_models("negbin")
    export_results("negbin", results)
    summary_climate_only(results, "negbin")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
