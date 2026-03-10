"""
Fama-French Three-Factor Model — Full Replication from Scratch
==============================================================
Data source : Ken French Data Library (manual CSV download)
Factors built: MKT, SMB, HML
Regression  : OLS via numpy (no statsmodels dependency)

HOW TO GET THE DATA
-------------------
1. Go to https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
2. Download "Fama/French 3 Factors" -> "CSV"  → save as  ff3_factors.csv
3. Download "Portfolios Formed on Size" (6 portfolios, Value Weighted) → ff_6portfolios.csv
   (This gives us the S/L, S/N, S/H, B/L, B/N, B/H monthly returns
    that we use to verify our replication against French's own numbers.)

Alternatively you can replicate entirely from stock-level CRSP data —
see the CUSTOM DATA section at the bottom of this file for that workflow.
"""

import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD KEN FRENCH DATA
# ─────────────────────────────────────────────────────────────────────────────

def _download_french_zip(url: str, cache_path: str) -> list:
    """
    Download a zip from Ken French's website, extract the first CSV inside,
    return its lines. Caches the raw CSV to cache_path so subsequent runs
    don't re-download.
    """
    if os.path.exists(cache_path):
        print(f"    Using cached file: {cache_path}")
        with open(cache_path, "r", encoding="latin-1") as f:
            return f.readlines()

    print(f"    Downloading from French's library...")
    headers = {"User-Agent": "Mozilla/5.0 (academic research)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        content  = zf.read(csv_name).decode("latin-1")

    with open(cache_path, "w", encoding="latin-1") as f:
        f.write(content)
    print(f"    Saved to: {cache_path}")
    return content.splitlines(keepends=True)


def _parse_french_csv(lines: list, n_cols: int, col_names: list) -> pd.DataFrame:
    """
    Parse ONE monthly data block from a Ken French CSV.
    Stops at the first blank line after data begins — this prevents
    accidentally reading a second section (e.g. Equal Weighted returns
    that immediately follows Value Weighted in the 6-portfolio file).
    """
    # Find first 6-digit YYYYMM row
    start = 0
    for i, line in enumerate(lines):
        tok = line.strip().split(",")[0].strip()
        if tok.isdigit() and len(tok) == 6:
            start = i
            break

    # Collect rows — stop at the FIRST blank or non-YYYYMM line after start
    rows = []
    for line in lines[start:]:
        parts = [p.strip() for p in line.strip().split(",")]
        tok   = parts[0] if parts else ""
        if tok.isdigit() and len(tok) == 6 and len(parts) >= n_cols:
            rows.append(parts[:n_cols])
        elif rows:
            # We had data and now hit something else — stop immediately
            break

    df = pd.DataFrame(rows, columns=col_names)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df / 100  # percent → decimal
    return df


def load_french_factors(
    filepath: str = "ff3_factors.csv",
    url: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
) -> pd.DataFrame:
    """
    Load Fama-French 3 Factors (MKT-RF, SMB, HML, RF).
    Auto-downloads from Ken French's website if file not found.
    Caches to filepath for subsequent runs.
    """
    lines = _download_french_zip(url, filepath) if not os.path.exists(filepath) \
            else open(filepath, "r", encoding="latin-1").readlines()

    return _parse_french_csv(
        lines,
        n_cols    = 5,
        col_names = ["date", "Mkt_RF", "SMB", "HML", "RF"]
    )


def load_six_portfolios(
    filepath: str = "ff_6portfolios.csv",
    url: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_BE-ME_CSV.zip"
) -> pd.DataFrame:
    """
    Load Ken French's 6 Size-BM portfolios (Value Weighted returns).
    Auto-downloads from Ken French's website if file not found.
    Caches to filepath for subsequent runs.
    Columns: SL, SN, SH, BL, BN, BH (decimal returns).
    """
    lines = _download_french_zip(url, filepath) if not os.path.exists(filepath) \
            else open(filepath, "r", encoding="latin-1").readlines()

    # The 6-portfolio file has multiple sections; we want "Value Weighted Returns"
    # Find that section header, then parse from there
    vw_start = 0
    for i, line in enumerate(lines):
        if "Value Weighted Returns" in line or "AVERAGE VALUE WEIGHTED" in line.upper():
            vw_start = i + 1
            break

    return _parse_french_csv(
        lines[vw_start:],
        n_cols    = 7,
        col_names = ["date", "SL", "SN", "SH", "BL", "BN", "BH"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPLICATE SMB AND HML FROM THE SIX PORTFOLIOS
# ─────────────────────────────────────────────────────────────────────────────

def replicate_factors(six_port: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """
    Replicate SMB and HML from the six size-BM portfolios exactly as in
    Fama & French (1993), equations (2.6) and (2.7).

    SMB = (1/3)(S/L + S/N + S/H) - (1/3)(B/L + B/N + B/H)
    HML = (1/2)(S/H + B/H)       - (1/2)(S/L + B/L)

    Parameters
    ----------
    six_port : DataFrame with columns SL, SN, SH, BL, BN, BH (decimal returns)
    rf       : Series of risk-free rate (decimal)

    Returns
    -------
    DataFrame with replicated SMB and HML columns
    """
    df = six_port.copy()

    # ── SMB: average small minus average big ─────────────────────────────────
    df["SMB_rep"] = (
        (df["SL"] + df["SN"] + df["SH"]) / 3
      - (df["BL"] + df["BN"] + df["BH"]) / 3
    )

    # ── HML: average high B/M minus average low B/M ───────────────────────────
    df["HML_rep"] = (
        (df["SH"] + df["BH"]) / 2
      - (df["SL"] + df["BL"]) / 2
    )

    return df[["SMB_rep", "HML_rep"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. OLS REGRESSION (numpy, no statsmodels)
# ─────────────────────────────────────────────────────────────────────────────

def ols_regression(y: np.ndarray, X: np.ndarray):
    """
    OLS: β = (X'X)^{-1} X'y
    Returns: coefficients, t-stats, R², residual std, residuals

    Newey-West HAC standard errors with lag = floor(4*(T/100)^(2/9))
    """
    T, k = X.shape

    # ── OLS estimates ─────────────────────────────────────────────────────────
    XtX_inv = np.linalg.inv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta
    sigma2  = resid @ resid / (T - k)         # unbiased variance estimate

    # ── Newey-West HAC covariance ─────────────────────────────────────────────
    lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

    # Meat of sandwich: S = Σ_t X_t X_t' ε_t² + lag-weighted cross terms
    scores = X * resid[:, None]               # T × k matrix of score vectors
    S = scores.T @ scores                     # lag-0 term

    for l in range(1, lag + 1):
        weight    = 1 - l / (lag + 1)         # Bartlett kernel
        cov_l     = scores[l:].T @ scores[:-l]
        S        += weight * (cov_l + cov_l.T)

    # Sandwich: V = (X'X)^{-1} S (X'X)^{-1}
    V_hac    = XtX_inv @ S @ XtX_inv
    se       = np.sqrt(np.diag(V_hac))
    t_stats  = beta / se

    # ── Goodness of fit ────────────────────────────────────────────────────────
    ss_res   = resid @ resid
    ss_tot   = np.sum((y - y.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot
    adj_r2   = 1 - (1 - r2) * (T - 1) / (T - k)

    return {
        "beta"   : beta,
        "se"     : se,
        "t_stat" : t_stats,
        "r2"     : r2,
        "adj_r2" : adj_r2,
        "resid"  : resid,
        "sigma"  : np.sqrt(sigma2),
        "T"      : T,
        "k"      : k,
    }


def print_regression_table(result: dict, labels: list, dep_var: str = "R_i - R_f"):
    """Pretty-print regression output."""
    print(f"\n{'═'*60}")
    print(f"  OLS Regression: {dep_var}")
    print(f"  T = {result['T']}   Newey-West SE (HAC)")
    print(f"{'═'*60}")
    print(f"  {'Variable':<14} {'Coef':>10} {'SE':>10} {'t-stat':>10}")
    print(f"  {'-'*46}")
    for i, lbl in enumerate(labels):
        sig = ""
        if abs(result["t_stat"][i]) > 3.291: sig = "***"
        elif abs(result["t_stat"][i]) > 2.576: sig = "** "
        elif abs(result["t_stat"][i]) > 1.960: sig = "*  "
        else: sig = "   "
        print(f"  {lbl:<14} {result['beta'][i]:>10.4f} "
              f"{result['se'][i]:>10.4f} {result['t_stat'][i]:>10.3f} {sig}")
    print(f"  {'-'*46}")
    print(f"  {'R²':<14} {result['r2']:>10.4f}")
    print(f"  {'Adj. R²':<14} {result['adj_r2']:>10.4f}")
    print(f"{'═'*60}")
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFICATION: compare replicated vs. French's published factors
# ─────────────────────────────────────────────────────────────────────────────

def verify_replication(ff3: pd.DataFrame, replicated: pd.DataFrame):
    """
    Regress replicated SMB/HML on French's published SMB/HML.
    A perfect replication gives: α ≈ 0, β ≈ 1, R² ≈ 1.
    """
    common = ff3.index.intersection(replicated.index)
    if len(common) == 0:
        print("[WARNING] No overlapping dates for verification.")
        return

    print("\n" + "─"*60)
    print("  REPLICATION VERIFICATION")
    print("  (Regress replicated factor on French's published factor)")
    print("─"*60)

    for factor, rep_col in [("SMB", "SMB_rep"), ("HML", "HML_rep")]:
        y = replicated.loc[common, rep_col].values
        X = np.column_stack([
            np.ones(len(common)),
            ff3.loc[common, factor].values
        ])
        res = ols_regression(y, X)
        print(f"\n  {factor}: α={res['beta'][0]:.6f}  β={res['beta'][1]:.6f}  R²={res['r2']:.6f}")
        if res["r2"] > 0.999:
            print(f"  ✓ Near-perfect replication (R² > 0.999)")
        else:
            print(f"  ⚠ Replication not exact — check portfolio definitions/date alignment")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def factor_summary(ff3: pd.DataFrame):
    """
    Print annualised summary statistics for MKT-RF, SMB, HML.
    Annualised mean  = monthly mean × 12
    Annualised vol   = monthly std  × √12
    Sharpe ratio     = annualised mean / annualised vol
    """
    factors = ["Mkt_RF", "SMB", "HML"]
    labels  = ["MKT-RF", "SMB", "HML"]

    print("\n" + "─"*60)
    print("  FACTOR SUMMARY STATISTICS (Annualised)")
    print("─"*60)
    print(f"  {'Factor':<10} {'Ann.Mean':>10} {'Ann.Vol':>10} {'Sharpe':>10} {'Skew':>8} {'Kurt':>8}")
    print(f"  {'-'*56}")

    for f, lbl in zip(factors, labels):
        s        = ff3[f]
        ann_mu   = s.mean() * 12
        ann_vol  = s.std() * np.sqrt(12)
        sharpe   = ann_mu / ann_vol
        skew     = float(((s - s.mean())**3).mean() / s.std()**3)
        kurt     = float(((s - s.mean())**4).mean() / s.std()**4) - 3  # excess
        print(f"  {lbl:<10} {ann_mu:>10.4f} {ann_vol:>10.4f} {sharpe:>10.3f} {skew:>8.3f} {kurt:>8.3f}")

    print(f"\n  Sample: {ff3.index[0].strftime('%b %Y')} – {ff3.index[-1].strftime('%b %Y')}  "
          f"(T = {len(ff3)} months)")

    print(f"\n  Factor Correlation Matrix:")
    corr = ff3[factors].corr()
    print(f"  {'':>10} {'MKT-RF':>10} {'SMB':>10} {'HML':>10}")
    for f, lbl in zip(factors, labels):
        row = "  " + f"{lbl:<10}"
        for f2 in factors:
            row += f"{corr.loc[f, f2]:>10.3f}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXAMPLE: RUN FF3 REGRESSION ON A TEST PORTFOLIO
#    Replace 'test_portfolio_returns' with your actual excess returns series
# ─────────────────────────────────────────────────────────────────────────────

def run_ff3_regression(
    portfolio_excess_returns: pd.Series,
    ff3: pd.DataFrame,
    name: str = "Test Portfolio"
):
    """
    Run a Fama-French 3-factor time-series regression on a portfolio.

    Parameters
    ----------
    portfolio_excess_returns : pd.Series
        Monthly EXCESS returns (return minus risk-free rate), decimal.
        Index must be DatetimeIndex aligned to ff3.
    ff3 : pd.DataFrame
        Output of load_french_factors() — must contain Mkt_RF, SMB, HML.
    name : str
        Label for the output table.
    """
    common = ff3.index.intersection(portfolio_excess_returns.index)
    y = portfolio_excess_returns.loc[common].values
    X = np.column_stack([
        np.ones(len(common)),             # intercept (alpha)
        ff3.loc[common, "Mkt_RF"].values, # MKT
        ff3.loc[common, "SMB"].values,    # SMB
        ff3.loc[common, "HML"].values,    # HML
    ])

    result = ols_regression(y, X)
    print_regression_table(
        result,
        labels   = ["Alpha", "β_MKT", "β_SMB", "β_HML"],
        dep_var  = f"{name} excess return"
    )

    # Annualised alpha
    ann_alpha = result["beta"][0] * 12
    print(f"\n  Annualised alpha: {ann_alpha:.4f} ({ann_alpha*100:.2f}% p.a.)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FAMA-FRENCH THREE-FACTOR MODEL — REPLICATION")
    print("=" * 60)

    # ── Load French's published factors ───────────────────────────────────────
    print("\n[1] Loading Fama-French factors...")
    ff3 = load_french_factors("ff3_factors.csv")
    print(f"    Loaded {len(ff3)} monthly observations "
          f"({ff3.index[0].strftime('%b %Y')} – {ff3.index[-1].strftime('%b %Y')})")

    # ── Summary statistics ─────────────────────────────────────────────────────
    print("\n[2] Factor Summary Statistics")
    factor_summary(ff3)

    # ── Replicate from 6 portfolios (if file available) ───────────────────────
    print("\n[3] Replicating SMB & HML from 6 portfolios...")
    six_port = load_six_portfolios("ff_6portfolios.csv")

    if six_port is not None:
        replicated = replicate_factors(six_port, ff3["RF"])
        verify_replication(ff3, replicated)
    else:
        print("    Skipping — ff_6portfolios.csv not found.")
        print("    Download 'Portfolios Formed on Size and Book-to-Market (6 Portfolios)'")
        print("    from Ken French's library to enable this step.")

    # ── Example regression ────────────────────────────────────────────────────
    print("\n[4] Example FF3 Regression")
    print("    (Using a synthetic portfolio: 0.6*MKT + 0.3*SMB - 0.1*HML + 0.001 + noise)")
    print("    Replace with your actual portfolio excess returns.\n")

    np.random.seed(42)
    T = len(ff3)
    synthetic_excess = (
          0.001                             # true alpha = 0.001/month
        + 0.60 * ff3["Mkt_RF"].values
        + 0.30 * ff3["SMB"].values
        - 0.10 * ff3["HML"].values
        + np.random.normal(0, 0.02, T)     # idiosyncratic noise
    )
    synthetic_series = pd.Series(synthetic_excess, index=ff3.index, name="Synthetic")
    result = run_ff3_regression(synthetic_series, ff3, name="Synthetic Portfolio")

    # ── Save factor data to CSV ───────────────────────────────────────────────
    output_path = "ff3_factors_processed.csv"
    ff3.to_csv(output_path)
    print(f"\n[5] Factor data saved to '{output_path}'")
    print(f"    Columns: {list(ff3.columns)}")
    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM DATA WORKFLOW (stock-level CRSP data)
# ─────────────────────────────────────────────────────────────────────────────
# If you have stock-level data (e.g. from CRSP/WRDS), the factor construction
# steps are:
#
#  1. MARKET FACTOR (MKT)
#     mkt = value_weighted_return(all_stocks, weights=market_cap) - rf
#
#  2. SIZE SORT (June of each year t)
#     breakpoint = NYSE_median(market_cap, month=June_t)
#     small  = stocks where market_cap < breakpoint
#     big    = stocks where market_cap >= breakpoint
#
#  3. BOOK-TO-MARKET SORT (June of each year t)
#     B/M  = book_equity(fiscal_year_t-1) / market_equity(Dec_t-1)
#     Use NYSE 30th and 70th percentiles as breakpoints.
#     low    = B/M < p30
#     neutral= p30 <= B/M <= p70
#     high   = B/M > p70
#
#  4. SIX PORTFOLIOS (value-weighted monthly returns, July_t to June_{t+1})
#     S/L, S/N, S/H, B/L, B/N, B/H
#
#  5. SMB = (1/3)(S/L + S/N + S/H) - (1/3)(B/L + B/N + B/H)
#     HML = (1/2)(S/H + B/H)       - (1/2)(S/L + B/L)
#
#  Key data requirements:
#   - Monthly stock returns (CRSP)
#   - Market capitalisation (CRSP)
#   - Book equity (Compustat: ceq or seq - ps - txditc)
#   - Risk-free rate (1-month T-bill, Ibbotson or CRSP)
#   - NYSE exchange flag (to compute NYSE-only breakpoints)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()