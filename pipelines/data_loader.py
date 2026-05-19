"""
================================================================================
 DATA LOADER — Normalized Schema Adapter
================================================================================
 Loads the normalized DB (CSV-based 3NF schema) and produces per-virus
 long-format DataFrames in the same shape the original pipelines expect.

 Normalized schema:
   dim_location      : Location_Name, State_Code, Location_ID
   fact_cases        : Location_ID, Disease, Year, Month, Cases, Monthly_Inc_per100k
   fact_climate      : Location_ID, Year, Month, Temp_Avg, Rain_mm,
                       Temp_Lag1, Temp_Lag2, Rain_Lag1, Rain_Lag2
   fact_population   : Location_ID, Year, Population

 Disease key mapping:
   DENV  -> Dengue
   CHIKV -> Chikungunya
   ZIKV  -> Zika

 Filter: Year >= 2016 for all diseases (CHIKV naturally starts 2017).
================================================================================
"""
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(_HERE, "db")

DISEASE_KEY = {"DENV": "Dengue", "CHIKV": "Chikungunya", "ZIKV": "Zika"}

UF_REGION = {
    "Rondonia": "Norte",    "Acre": "Norte",        "Amazonas": "Norte",
    "Roraima": "Norte",     "Para": "Norte",         "Amapa": "Norte",
    "Tocantins": "Norte",   "Maranhao": "Nordeste",  "Piaui": "Nordeste",
    "Ceara": "Nordeste",    "Rio Grande do Norte": "Nordeste",
    "Paraiba": "Nordeste",  "Pernambuco": "Nordeste",
    "Alagoas": "Nordeste",  "Sergipe": "Nordeste",   "Bahia": "Nordeste",
    "Minas Gerais": "Sudeste", "Espirito Santo": "Sudeste",
    "Rio de Janeiro": "Sudeste", "Sao Paulo": "Sudeste",
    "Parana": "Sul",        "Santa Catarina": "Sul", "Rio Grande do Sul": "Sul",
    "Mato Grosso do Sul": "Centro-Oeste", "Mato Grosso": "Centro-Oeste",
    "Goias": "Centro-Oeste", "Distrito Federal": "Centro-Oeste",
}

YEAR_START = 2016


def load_normalized_tables():
    """Load the four normalized CSV tables. Returns dict of DataFrames."""
    return {
        "loc":     pd.read_csv(os.path.join(DB_DIR, "normalized_dim_location.csv")),
        "cases":   pd.read_csv(os.path.join(DB_DIR, "normalized_fact_cases.csv")),
        "climate": pd.read_csv(os.path.join(DB_DIR, "normalized_fact_climate.csv")),
        "pop":     pd.read_csv(os.path.join(DB_DIR, "normalized_fact_population.csv")),
    }


def load_pipeline_datasets(year_start=YEAR_START):
    """Return {virus_name: long-format DataFrame} matching original Excel schema.

    Output columns per row:
      Year, Month, Location_Name, UF_Code, Cases, Population, Monthly_Inc_per100k,
      Temp_Avg, Rain_mm, Temp_Lag1, Temp_Lag2, Rain_Lag1, Rain_Lag2,
      Date, Region, Monthly_Inc_100k (legacy alias for downstream compatibility)
    """
    tbl = load_normalized_tables()
    loc, cases, clim, pop = tbl["loc"], tbl["cases"], tbl["climate"], tbl["pop"]

    # Filter to >= year_start
    cases = cases[cases["Year"] >= year_start].copy()
    clim  = clim[clim["Year"]  >= year_start].copy()
    pop   = pop[pop["Year"]    >= year_start].copy()

    datasets = {}
    for code, virus in DISEASE_KEY.items():
        d = cases[cases["Disease"] == code].copy()
        d = d.merge(clim, on=["Location_ID", "Year", "Month"], how="left")
        d = d.merge(pop, on=["Location_ID", "Year"], how="left")
        d = d.merge(loc, on="Location_ID", how="left")
        d = d.rename(columns={"State_Code": "UF_Code"})

        d["Date"] = pd.to_datetime(
            d["Year"].astype(str) + "-" +
            d["Month"].astype(str).str.zfill(2) + "-01"
        )
        d["Region"] = d["Location_Name"].map(UF_REGION)
        # Legacy alias used by the original pipeline code:
        d["Monthly_Inc_100k"] = d["Monthly_Inc_per100k"]

        d = d.sort_values(["Location_Name", "Year", "Month"]).reset_index(drop=True)
        datasets[virus] = d
        print(f"  [{virus:12s}] {len(d):5d} obs | "
              f"{d['Location_Name'].nunique()} states | "
              f"{d['Year'].min()}-{d['Year'].max()}")
    return datasets
