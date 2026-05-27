"""
================================================================================
 GLM COMMON HELPERS — Poisson / quasi-Poisson / Negative Binomial
================================================================================
 Shared regression machinery for the three count-data GLM scripts.

 Model: Cases ~ climate predictors  with  offset(log Population)
 Three model specifications per model family:

   spec="per_state"   : one GLM per (Disease, State) on its monthly time series
                        Predictors: Temp_Avg, Rain_mm
   spec="lagged"      : per (Disease, State) but predictors include lag-0/1/2
                        for both Temp and Rain
   spec="pooled_fe"   : one GLM per Disease, pooled across all states with
                        State as a categorical fixed effect.
                        Predictors: Temp_Avg, Rain_mm, C(Location_Name)

 Output rows include: IRR (exp(coef)), 95% CI on IRR, p-value, dispersion,
 AIC / BIC / log-likelihood, n.

 References:
   - Poisson GLM            : statsmodels GLM family=Poisson, link=log
   - Quasi-Poisson          : statsmodels GLM family=Poisson with cov_type=
                              not supported natively; we report Pearson chi2/df
                              as dispersion and inflate SEs by sqrt(dispersion).
                              Equivalent to R's quasipoisson.
   - Negative Binomial      : statsmodels NegativeBinomial (NB2 parameterization)
                              with alpha estimated by MLE.
================================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from data_loader import load_pipeline_datasets, DISEASE_KEY

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05

BASE_PREDICTORS   = ["Temp_Avg", "Rain_mm"]
LAGGED_PREDICTORS = ["Temp_Avg", "Temp_Lag1", "Temp_Lag2",
                     "Rain_mm",  "Rain_Lag1", "Rain_Lag2"]


def prepare_glm_frame(df, predictors):
    """Drop NA on predictors and dependent variables; require Population > 0."""
    cols = ["Cases", "Population", "Location_Name"] + predictors
    sub = df[cols].dropna()
    sub = sub[sub["Population"] > 0]
    sub = sub.copy()
    sub["log_pop"] = np.log(sub["Population"].astype(float))
    return sub


def _summarize(model, model_label, scale=None):
    """Build a per-coefficient summary row dict list. `scale` inflates SEs (for
    quasi-Poisson). For Poisson/NB, scale=None."""
    p = model.params
    if scale is None:
        se = model.bse
        pval = model.pvalues
    else:
        se = model.bse * np.sqrt(scale)
        # Wald z-score with inflated SE, two-sided p
        z = p / se
        pval = 2 * (1 - sm.stats.stattools.stats.norm.cdf(np.abs(z))) \
               if False else pd.Series(
                   [2 * (1 - sm.distributions.ECDF([0])(abs(z_i))) for z_i in z],
                   index=p.index
               )
        # Simpler: use scipy
        from scipy import stats as _st
        pval = pd.Series(
            [2 * (1 - _st.norm.cdf(abs(zi))) for zi in z], index=p.index
        )

    rows = []
    for var in p.index:
        if var == "alpha":
            # NegBin dispersion parameter — reported separately, not a coefficient
            continue
        coef = p[var]
        rows.append({
            "Term":        var,
            "Coef":        coef,
            "SE":          se[var],
            "IRR":         float(np.exp(coef)),
            "IRR_CI_L":    float(np.exp(coef - 1.96 * se[var])),
            "IRR_CI_U":    float(np.exp(coef + 1.96 * se[var])),
            "z":           coef / se[var],
            "p_value":     pval[var],
            "Significant": pval[var] < ALPHA,
        })
    return rows


def fit_poisson(formula, data, *, quasi=False):
    """Fit a Poisson GLM. If quasi=True returns inflated SE based on Pearson chi2/df."""
    m = smf.glm(formula=formula, data=data, family=sm.families.Poisson(),
                offset=data["log_pop"]).fit()
    if quasi:
        df_resid = m.df_resid
        pearson_chi2 = float(m.pearson_chi2)
        dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan
        return m, dispersion
    return m, None


def fit_negbin(formula, data):
    """Fit Negative Binomial (NB2) regression with MLE of alpha.
    Returns (result, alpha). Uses statsmodels.NegativeBinomial with the
    patsy-built design matrix to recover alpha as a fitted parameter."""
    from patsy import dmatrices
    y, X = dmatrices(formula, data, return_type="dataframe")
    try:
        m = sm.NegativeBinomial(
            endog=y.iloc[:, 0].astype(float),
            exog=X,
            offset=data["log_pop"].values,
        ).fit(disp=0, method="nm", maxiter=2000)
        alpha = float(m.params.get("alpha", np.nan))
        return m, alpha
    except Exception:
        # Fallback to GLM with fixed alpha
        m = smf.glm(formula=formula, data=data,
                    family=sm.families.NegativeBinomial(alpha=1.0),
                    offset=data["log_pop"]).fit()
        return m, 1.0


def negbin_dispersion(m, alpha):
    """Pearson chi2 / df_resid for an NB2 fit (analogue to GLM dispersion).

    Works for both the discrete NegativeBinomial result and the GLM fallback.
    Under correct NB2 specification this ratio should be ~1; large departures
    flag remaining mis-fit beyond what alpha absorbs.
    """
    try:
        if hasattr(m, "pearson_chi2") and hasattr(m, "df_resid") and m.df_resid:
            return float(m.pearson_chi2) / float(m.df_resid)
        mu = np.asarray(m.predict())
        y  = np.asarray(m.model.endog, dtype=float)
        var = mu + alpha * mu**2
        var = np.where(var > 0, var, np.nan)
        pearson = float(np.nansum((y - mu)**2 / var))
        # Discrete NB params include 'alpha'; subtract it so df_resid matches
        # the GLM convention (residual df relative to mean-model parameters).
        n_mean_params = sum(1 for k in m.params.index if k != "alpha")
        df_resid = len(y) - n_mean_params
        return pearson / df_resid if df_resid > 0 else np.nan
    except Exception:
        return np.nan


def run_models(model_family):
    """Run all three model specifications for the given family.
    model_family ∈ {"poisson", "quasi-poisson", "negbin"}
    Returns dict of {spec_name: DataFrame}.
    """
    datasets = load_pipeline_datasets()
    results = {}

    # -- Spec 1: per-state --
    print(f"\n[{model_family} | per_state] fitting one GLM per (Disease, State)")
    rows = []
    diag_rows = []
    for virus, df in datasets.items():
        for uf, grp in df.groupby("Location_Name"):
            sub = prepare_glm_frame(grp, BASE_PREDICTORS)
            if len(sub) < 12 or sub["Cases"].sum() == 0:
                continue
            try:
                if model_family == "poisson":
                    m, _ = fit_poisson("Cases ~ Temp_Avg + Rain_mm", sub)
                    scale = None
                    disp = float(m.pearson_chi2/m.df_resid) if m.df_resid>0 else np.nan
                    aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
                elif model_family == "quasi-poisson":
                    m, disp = fit_poisson("Cases ~ Temp_Avg + Rain_mm", sub, quasi=True)
                    scale = disp
                    aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
                elif model_family == "negbin":
                    m, alpha = fit_negbin("Cases ~ Temp_Avg + Rain_mm", sub)
                    scale = None
                    disp = negbin_dispersion(m, alpha)
                    aux = {"dispersion": disp, "alpha": alpha, "overdispersion": alpha}
                else:
                    raise ValueError(model_family)
                coefs = _summarize(m, model_family, scale=scale)
                for r in coefs:
                    r.update({"Virus": virus, "Location_Name": uf,
                              "Model": model_family, "Spec": "per_state",
                              "AIC": getattr(m, "aic", np.nan),
                              "n": int(len(sub)),
                              **aux})
                    rows.append(r)
                diag_rows.append({"Virus": virus, "Location_Name": uf,
                                  "AIC": getattr(m, "aic", np.nan),
                                  "BIC": getattr(m, "bic", np.nan),
                                  "logLik": getattr(m, "llf", np.nan),
                                  "n": int(len(sub)),
                                  "Model": model_family, "Spec": "per_state",
                                  **aux})
            except Exception as exc:
                print(f"    fit failed for {virus} {uf}: {exc}")
    results["per_state"] = pd.DataFrame(rows)
    results["per_state_diag"] = pd.DataFrame(diag_rows)

    # -- Spec 2: lagged climate per-state --
    print(f"\n[{model_family} | lagged] per (Disease, State) with lag-0/1/2 predictors")
    rows = []
    diag_rows = []
    formula_lagged = ("Cases ~ Temp_Avg + Temp_Lag1 + Temp_Lag2 + "
                      "Rain_mm + Rain_Lag1 + Rain_Lag2")
    for virus, df in datasets.items():
        for uf, grp in df.groupby("Location_Name"):
            sub = prepare_glm_frame(grp, LAGGED_PREDICTORS)
            if len(sub) < 18 or sub["Cases"].sum() == 0:
                continue
            try:
                if model_family == "poisson":
                    m, _ = fit_poisson(formula_lagged, sub)
                    scale = None
                    disp = float(m.pearson_chi2/m.df_resid) if m.df_resid>0 else np.nan
                    aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
                elif model_family == "quasi-poisson":
                    m, disp = fit_poisson(formula_lagged, sub, quasi=True)
                    scale = disp
                    aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
                elif model_family == "negbin":
                    m, alpha = fit_negbin(formula_lagged, sub)
                    scale = None
                    disp = negbin_dispersion(m, alpha)
                    aux = {"dispersion": disp, "alpha": alpha, "overdispersion": alpha}
                coefs = _summarize(m, model_family, scale=scale)
                for r in coefs:
                    r.update({"Virus": virus, "Location_Name": uf,
                              "Model": model_family, "Spec": "lagged",
                              "AIC": getattr(m, "aic", np.nan),
                              "n": int(len(sub)),
                              **aux})
                    rows.append(r)
                diag_rows.append({"Virus": virus, "Location_Name": uf,
                                  "AIC": getattr(m, "aic", np.nan),
                                  "BIC": getattr(m, "bic", np.nan),
                                  "logLik": getattr(m, "llf", np.nan),
                                  "n": int(len(sub)),
                                  "Model": model_family, "Spec": "lagged",
                                  **aux})
            except Exception as exc:
                print(f"    fit failed for {virus} {uf}: {exc}")
    results["lagged"] = pd.DataFrame(rows)
    results["lagged_diag"] = pd.DataFrame(diag_rows)

    # -- Spec 3: pooled with state fixed effects --
    print(f"\n[{model_family} | pooled_fe] one GLM per disease with C(state)")
    rows = []
    diag_rows = []
    for virus, df in datasets.items():
        sub = prepare_glm_frame(df, BASE_PREDICTORS)
        if len(sub) < 50:
            continue
        try:
            formula_fe = "Cases ~ Temp_Avg + Rain_mm + C(Location_Name)"
            if model_family == "poisson":
                m, _ = fit_poisson(formula_fe, sub)
                scale = None
                disp = float(m.pearson_chi2/m.df_resid) if m.df_resid>0 else np.nan
                aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
            elif model_family == "quasi-poisson":
                m, disp = fit_poisson(formula_fe, sub, quasi=True)
                scale = disp
                aux = {"dispersion": disp, "alpha": np.nan, "overdispersion": disp}
            elif model_family == "negbin":
                m, alpha = fit_negbin(formula_fe, sub)
                scale = None
                disp = negbin_dispersion(m, alpha)
                aux = {"dispersion": disp, "alpha": alpha, "overdispersion": alpha}
            coefs = _summarize(m, model_family, scale=scale)
            for r in coefs:
                r.update({"Virus": virus, "Location_Name": "ALL",
                          "Model": model_family, "Spec": "pooled_fe",
                          "AIC": getattr(m, "aic", np.nan),
                          "n": int(len(sub)),
                          **aux})
                rows.append(r)
            diag_rows.append({"Virus": virus, "Location_Name": "ALL",
                              "AIC": getattr(m, "aic", np.nan),
                              "BIC": getattr(m, "bic", np.nan),
                              "logLik": getattr(m, "llf", np.nan),
                              "n": int(len(sub)),
                              "Model": model_family, "Spec": "pooled_fe",
                              **aux})
        except Exception as exc:
            print(f"    fit failed for {virus}: {exc}")
    results["pooled_fe"] = pd.DataFrame(rows)
    results["pooled_fe_diag"] = pd.DataFrame(diag_rows)

    return results


def export_results(model_family, results):
    """Write three CSVs + one Excel workbook with all spec results."""
    base = os.path.join(OUT_DIR, f"glm_{model_family.replace('-','_')}")
    # Per-spec CSVs
    for spec in ("per_state", "lagged", "pooled_fe"):
        df = results[spec]
        if df.empty: continue
        df.to_csv(f"{base}_{spec}.csv", index=False)
        print(f"    Saved -> {os.path.basename(base)}_{spec}.csv ({len(df)} rows)")

    # Combined Excel
    xlsx = f"{base}_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for spec in ("per_state", "lagged", "pooled_fe"):
            df = results[spec]
            if not df.empty:
                df.to_excel(w, sheet_name=spec, index=False)
            diag = results.get(f"{spec}_diag")
            if diag is not None and not diag.empty:
                diag.to_excel(w, sheet_name=f"{spec}_diagnostics", index=False)
    print(f"    Saved -> {os.path.basename(xlsx)}")


def summary_climate_only(results, model_family):
    """Print a short console summary: # significant Temp / Rain coefficients
    by spec and virus."""
    print(f"\n  [{model_family}] summary of climate IRRs (Temp_Avg, Rain_mm):")
    for spec in ("per_state", "lagged", "pooled_fe"):
        df = results[spec]
        if df.empty: continue
        clim = df[df["Term"].isin(["Temp_Avg","Rain_mm"])]
        if clim.empty: continue
        for virus, g in clim.groupby("Virus"):
            n_total = g["Location_Name"].nunique()
            n_sig_t = g[(g["Term"]=="Temp_Avg") & (g["Significant"])].shape[0]
            n_sig_r = g[(g["Term"]=="Rain_mm") & (g["Significant"])].shape[0]
            print(f"    [{spec:10s}] {virus:12s} "
                  f"Temp sig={n_sig_t}/{n_total}  "
                  f"Rain sig={n_sig_r}/{n_total}")
