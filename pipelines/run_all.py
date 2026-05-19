"""
Run the full normalized pipeline end-to-end:
  1. arbovirus_pipeline_en.py         — descriptive + trend + Spearman + Moran
  2. arbovirus_geospatial.py          — choropleth, LISA, trend maps
  3. graphpad_to_python_pipeline.py   — clinical / demographic plots
  4. bootstrap_ci.py                  — bootstrap 95% CIs alongside normal-approx
  5. glm_poisson.py                   — Poisson regression
  6. glm_quasi_poisson.py             — quasi-Poisson regression
  7. glm_negbin.py                    — Negative Binomial regression
  8. merge_figures.py                 — composite publication figures
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "arbovirus_pipeline_en.py",
    "arbovirus_geospatial.py",
    "graphpad_to_python_pipeline.py",
    "bootstrap_ci.py",
    "glm_poisson.py",
    "glm_quasi_poisson.py",
    "glm_negbin.py",
    "merge_figures.py",
]


def main():
    failed = []
    for s in SCRIPTS:
        path = os.path.join(HERE, s)
        print(f"\n{'='*70}\n>>> Running {s}\n{'='*70}")
        try:
            subprocess.run([sys.executable, path], cwd=HERE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! {s} failed (exit {e.returncode})")
            failed.append(s)
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {len(SCRIPTS)-len(failed)}/{len(SCRIPTS)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
