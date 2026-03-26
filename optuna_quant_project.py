"""
============================================================
  Optuna Hyperparameter Tuning — Quant Finance Alpha Model
  XGBoost signal predictor | Walk-forward CV | Sharpe obj.
============================================================

PROJECT STRUCTURE
-----------------
1. Data generation   — synthetic OHLCV + factor data
2. Feature engineering — momentum, mean-reversion, vol signals
3. Objective function  — walk-forward CV with Sharpe ratio
4. Optuna study        — Bayesian search + pruning
5. Analysis            — param importance, optimization history
6. Out-of-sample test  — final holdout evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. SYNTHETIC MARKET DATA GENERATION
# ─────────────────────────────────────────────

def generate_market_data(n_assets: int = 50, n_days: int = 1500) -> pd.DataFrame:
    """
    Generate realistic synthetic equity return data.
    Includes a latent factor structure + idiosyncratic noise.
    """
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # Market factor (common shock)
    market_factor = np.random.normal(0.0003, 0.01, n_days)

    # Sector factors (3 sectors)
    sector_factors = np.random.normal(0, 0.008, (3, n_days))

    # Asset assignments
    sectors = np.random.choice(3, n_assets)
    betas   = np.random.uniform(0.5, 1.5, n_assets)

    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        idio = np.random.normal(0, 0.015, n_days)
        returns[:, i] = (
            betas[i] * market_factor
            + 0.4 * sector_factors[sectors[i]]
            + idio
        )

    df = pd.DataFrame(returns, index=dates,
                      columns=[f"ASSET_{i:03d}" for i in range(n_assets)])
    return df


def engineer_features(returns: pd.DataFrame) -> tuple:
    """
    Build predictive features for next-day return.
    Returns feature matrix X and target y (forward 1-day return).
    """
    feats = pd.DataFrame(index=returns.index)

    # Pick one asset as target (ASSET_000), use cross-section for features
    target_col = "ASSET_000"
    r = returns[target_col]

    # ── Momentum signals ──────────────────────────────
    for w in [5, 10, 21, 63]:
        feats[f"mom_{w}d"]     = r.shift(1).rolling(w).sum()
        feats[f"vol_{w}d"]     = r.shift(1).rolling(w).std()
        feats[f"skew_{w}d"]    = r.shift(1).rolling(w).skew()

    # ── Mean-reversion signals ────────────────────────
    for w in [5, 10, 21]:
        ma = r.shift(1).rolling(w).mean()
        feats[f"rev_{w}d"]     = r.shift(1) - ma
        feats[f"zscore_{w}d"]  = feats[f"rev_{w}d"] / (r.shift(1).rolling(w).std() + 1e-8)

    # ── Cross-sectional rank features ─────────────────
    cs_mom_5 = returns.shift(1).rolling(5).sum()
    if target_col in cs_mom_5.columns:
        feats["cs_rank_mom5"] = cs_mom_5[target_col].rank(pct=True)
    else:
        feats["cs_rank_mom5"] = 0.5

    # ── Volatility regime ─────────────────────────────
    feats["vol_ratio"]     = feats["vol_5d"] / (feats["vol_63d"] + 1e-8)

    # ── Lagged returns ────────────────────────────────
    for lag in [1, 2, 3, 5]:
        feats[f"lag_{lag}d"] = r.shift(lag)

    # ── Target: next day return ───────────────────────
    y = r.shift(-1)

    # Clean up
    valid = feats.dropna().index.intersection(y.dropna().index)
    X = feats.loc[valid]
    y = y.loc[valid]

    return X, y


# ─────────────────────────────────────────────
# 2. WALK-FORWARD CROSS-VALIDATION
# ─────────────────────────────────────────────

class WalkForwardCV:
    """
    Expanding-window walk-forward CV with embargo.
    Prevents look-ahead bias in time-series evaluation.
    """
    def __init__(self,
                 n_folds: int = 5,
                 test_size: int = 63,    # ~3 months
                 min_train: int = 252,   # ~1 year min train
                 embargo: int = 5):      # 5-day gap
        self.n_folds  = n_folds
        self.test_size = test_size
        self.min_train = min_train
        self.embargo   = embargo

    def split(self, X: pd.DataFrame):
        n = len(X)
        splits = []
        for fold in range(self.n_folds):
            test_end   = n - fold * self.test_size
            test_start = test_end - self.test_size
            train_end  = test_start - self.embargo
            train_start = max(0, train_end - (n - self.min_train - self.n_folds * self.test_size))

            if train_end - train_start < self.min_train:
                continue

            train_idx = list(range(train_start, train_end))
            test_idx  = list(range(test_start, test_end))
            splits.append((train_idx, test_idx))

        return splits[::-1]  # chronological order


# ─────────────────────────────────────────────
# 3. PERFORMANCE METRICS
# ─────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, annualize: float = 252.0) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 5 or returns.std() < 1e-10:
        return -999.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualize))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank IC: Spearman correlation of predictions vs realized returns."""
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd   = (equity_curve - peak) / (peak + 1e-10)
    return float(dd.min())


# ─────────────────────────────────────────────
# 4. OPTUNA OBJECTIVE FUNCTION
# ─────────────────────────────────────────────

def build_objective(X: pd.DataFrame, y: pd.Series, cv: WalkForwardCV):
    """
    Closure that returns the Optuna objective.
    Objective = mean Sharpe ratio across walk-forward folds.
    Uses Optuna's pruning to kill bad trials early.
    """
    X_arr = X.values
    y_arr = y.values
    splits = cv.split(X)

    def objective(trial: optuna.Trial) -> float:

        # ── Hyperparameter search space ───────────────
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 30),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "random_state":      SEED,
            "tree_method":       "hist",
            "objective":         "reg:squarederror",
            "verbosity":         0,
        }

        fold_sharpes = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_te, y_te = X_arr[test_idx],  y_arr[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_s, y_tr)

            # Predict & score
            preds = model.predict(X_te_s)

            # Long top-quintile, short bottom-quintile
            threshold_long  = np.percentile(preds, 80)
            threshold_short = np.percentile(preds, 20)
            position = np.where(preds >= threshold_long,  1.0,
                       np.where(preds <= threshold_short, -1.0, 0.0))

            pnl = position * y_te
            sr  = sharpe_ratio(pnl)

            fold_sharpes.append(sr)

            # ── Pruning: report intermediate value ────
            trial.report(float(np.mean(fold_sharpes)), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_sharpes))

    return objective


# ─────────────────────────────────────────────
# 5. RUN OPTUNA STUDY
# ─────────────────────────────────────────────

def run_study(X: pd.DataFrame, y: pd.Series, n_trials: int = 60) -> optuna.Study:

    cv = WalkForwardCV(n_folds=5, test_size=63, min_train=252, embargo=5)
    objective = build_objective(X, y, cv)

    # TPE sampler: Bayesian optimization via Tree-structured Parzen Estimators
    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=15,
        multivariate=True,
    )

    # MedianPruner: kill trials below median of completed trials at same step
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="quant_alpha_xgb",
    )

    print(f"\n{'='*60}")
    print(f"  Running Optuna study: {n_trials} trials")
    print(f"  Sampler : TPE (multivariate=True)")
    print(f"  Pruner  : MedianPruner")
    print(f"  CV      : 5-fold walk-forward, 5-day embargo")
    print(f"  Obj.    : Mean Sharpe ratio across folds")
    print(f"{'='*60}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    return study


# ─────────────────────────────────────────────
# 6. FINAL OUT-OF-SAMPLE EVALUATION
# ─────────────────────────────────────────────

def final_evaluation(X: pd.DataFrame, y: pd.Series,
                     best_params: dict) -> dict:
    """
    True holdout test — last 126 trading days (~6 months).
    Called ONCE after tuning. Do not loop over this.
    """
    cutoff = int(len(X) * 0.85)
    X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
    X_test,  y_test  = X.iloc[cutoff:], y.iloc[cutoff:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    params = {**best_params, "random_state": SEED,
              "tree_method": "hist", "objective": "reg:squarederror",
              "verbosity": 0}

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr_s, y_train)
    preds = model.predict(X_te_s)

    threshold_long  = np.percentile(preds, 80)
    threshold_short = np.percentile(preds, 20)
    position = np.where(preds >= threshold_long,  1.0,
               np.where(preds <= threshold_short, -1.0, 0.0))

    pnl    = position * y_test.values
    equity = np.cumprod(1 + pnl)
    mdd    = max_drawdown(equity)
    sr     = sharpe_ratio(pnl)
    ic     = information_coefficient(y_test.values, preds)
    hit_r  = float(np.mean(np.sign(preds) == np.sign(y_test.values)))

    results = {
        "sharpe":       sr,
        "ic":           ic,
        "max_drawdown": mdd,
        "hit_rate":     hit_r,
        "total_return": float(equity[-1] - 1),
        "n_days":       len(pnl),
        "equity_curve": equity,
        "pnl":          pnl,
        "feature_importances": model.feature_importances_,
        "feature_names":       list(X.columns),
    }
    return results


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(study: optuna.Study, oos_results: dict,
                 output_path: str) -> None:

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    C_BG   = "#0d1117"
    C_CARD = "#161b22"
    C_BLUE = "#58a6ff"
    C_TEAL = "#3fb950"
    C_AMB  = "#d29922"
    C_RED  = "#f85149"
    C_PURP = "#bc8cff"
    C_TXT  = "#c9d1d9"
    C_MUT  = "#8b949e"
    C_LINE = "#21262d"

    def card_ax(subplot_spec):
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values():
            sp.set_color(C_LINE)
        ax.tick_params(colors=C_MUT, labelsize=8)
        ax.xaxis.label.set_color(C_MUT)
        ax.yaxis.label.set_color(C_MUT)
        ax.title.set_color(C_TXT)
        return ax

    # ── (A) Optimization history ──────────────────────────────
    ax_a = card_ax(gs[0, :2])
    trials   = [t for t in study.trials if t.value is not None]
    trial_ns = [t.number for t in trials]
    values   = [t.value for t in trials]
    best_so_far = pd.Series(values).cummax().values
    pruned_n = [t.number for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED]

    ax_a.scatter(trial_ns, values, s=18, alpha=0.5, color=C_BLUE,
                 zorder=2, label="Trial Sharpe")
    ax_a.plot(trial_ns, best_so_far, color=C_TEAL, lw=2,
              label="Best so far", zorder=3)
    for pn in pruned_n:
        ax_a.axvline(pn, color=C_RED, alpha=0.15, lw=0.8)
    ax_a.axhline(0, color=C_LINE, lw=1, ls="--")
    ax_a.set_title("Optimization history  |  objective = mean Sharpe (walk-forward)", fontsize=10)
    ax_a.set_xlabel("Trial number")
    ax_a.set_ylabel("Sharpe ratio")
    ax_a.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TXT,
                framealpha=0.8)
    ax_a.grid(axis="y", color=C_LINE, lw=0.5)

    # ── (B) Hyperparameter importance ────────────────────────
    ax_b = card_ax(gs[0, 2])
    try:
        importance = optuna.importance.get_param_importances(study)
        params_imp = list(importance.keys())[:8]
        vals_imp   = [importance[p] for p in params_imp]
        colors_imp = [C_PURP if v == max(vals_imp) else C_BLUE for v in vals_imp]
        ax_b.barh(params_imp[::-1], vals_imp[::-1],
                  color=colors_imp[::-1], height=0.6)
        ax_b.set_title("Param importance (fANOVA)", fontsize=10)
        ax_b.set_xlabel("Importance score")
        ax_b.set_xlim(0, max(vals_imp) * 1.2)
        ax_b.grid(axis="x", color=C_LINE, lw=0.5)
    except Exception:
        ax_b.text(0.5, 0.5, "Need ≥10 completed\ntrials for fANOVA",
                  ha="center", va="center", color=C_MUT, transform=ax_b.transAxes)
        ax_b.set_title("Param importance (fANOVA)", fontsize=10)

    # ── (C) Out-of-sample equity curve ───────────────────────
    ax_c = card_ax(gs[1, :2])
    eq  = oos_results["equity_curve"]
    pnl = oos_results["pnl"]
    x   = np.arange(len(eq))

    # Color positive/negative PnL bars
    bar_colors = [C_TEAL if p >= 0 else C_RED for p in pnl]
    ax_c2 = ax_c.twinx()
    ax_c2.set_facecolor(C_CARD)
    ax_c2.bar(x, pnl, color=bar_colors, alpha=0.25, width=1.0, zorder=1)
    ax_c2.tick_params(colors=C_MUT, labelsize=8)
    ax_c2.yaxis.label.set_color(C_MUT)
    for sp in ax_c2.spines.values():
        sp.set_color(C_LINE)

    ax_c.plot(x, (eq - 1) * 100, color=C_BLUE, lw=2, zorder=3)
    ax_c.fill_between(x, (eq - 1) * 100, alpha=0.12, color=C_BLUE)
    ax_c.axhline(0, color=C_LINE, lw=1, ls="--")
    ax_c.set_title("Out-of-sample equity curve  |  L/S quintile portfolio", fontsize=10)
    ax_c.set_xlabel("Trading days")
    ax_c.set_ylabel("Cumulative return (%)")
    ax_c2.set_ylabel("Daily PnL", alpha=0.7)
    ax_c.set_zorder(ax_c2.get_zorder() + 1)
    ax_c.patch.set_visible(False)
    ax_c.grid(axis="y", color=C_LINE, lw=0.5)

    # ── (D) Performance metrics scorecard ────────────────────
    ax_d = card_ax(gs[1, 2])
    ax_d.axis("off")
    ax_d.set_title("OOS performance", fontsize=10)

    metrics = [
        ("Sharpe Ratio",    f"{oos_results['sharpe']:.3f}"),
        ("Max Drawdown",    f"{oos_results['max_drawdown']*100:.2f}%"),
        ("Total Return",    f"{oos_results['total_return']*100:.2f}%"),
        ("Hit Rate",        f"{oos_results['hit_rate']*100:.1f}%"),
        ("Rank IC",         f"{oos_results['ic']:.4f}"),
        ("Trading Days",    f"{oos_results['n_days']}"),
    ]
    
    # Draw metrics
    for i, (label, val) in enumerate(metrics):
        y_pos = 0.88 - i * 0.15
        ax_d.text(0.05, y_pos, label, transform=ax_d.transAxes,
                  color=C_MUT, fontsize=9, va='center')
        ax_d.text(0.95, y_pos, val, transform=ax_d.transAxes,
                  color=C_TEAL, fontsize=11, fontweight="bold", 
                  ha='right', va='center')
        
        # Draw separator line
        if i < len(metrics) - 1:
            ax_d.axhline(y=y_pos - 0.07, xmin=0.02, xmax=0.98, 
                        color=C_LINE, lw=0.5)

    # ── (E) Feature importances ───────────────────────────────
    ax_e = card_ax(gs[2, :2])
    fi = oos_results["feature_importances"]
    fn = oos_results["feature_names"]
    top_k = min(15, len(fi))
    idx = np.argsort(fi)[-top_k:]
    bar_cols = [C_PURP if fi[i] == fi.max() else C_TEAL for i in idx]
    ax_e.barh([fn[i] for i in idx], fi[idx], color=bar_cols, height=0.6)
    ax_e.set_title(f"Top {top_k} XGBoost feature importances (gain)", fontsize=10)
    ax_e.set_xlabel("Importance (gain)")
    ax_e.grid(axis="x", color=C_LINE, lw=0.5)

    # ── (F) Hyperparameter parallel plot (top trials) ─────────
    ax_f = card_ax(gs[2, 2])
    completed = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:20]
    
    if len(completed) > 0:
        lr_vals = [t.params.get("learning_rate", 0) for t in completed]
        depth_vals = [t.params.get("max_depth", 0) for t in completed]
        sr_vals  = [t.value for t in completed]
        scatter = ax_f.scatter(lr_vals, depth_vals, c=sr_vals,
                               cmap="RdYlGn", s=60, alpha=0.85,
                               edgecolors="none")
        cbar = plt.colorbar(scatter, ax=ax_f)
        cbar.ax.tick_params(colors=C_MUT, labelsize=7)
        cbar.set_label("Sharpe", color=C_MUT, fontsize=8)
        ax_f.set_xscale("log")
        ax_f.set_title("Top-20 trials: lr vs depth", fontsize=10)
        ax_f.set_xlabel("learning_rate (log)")
        ax_f.set_ylabel("max_depth")
        ax_f.grid(color=C_LINE, lw=0.5)
    else:
        ax_f.text(0.5, 0.5, "No completed trials", 
                  ha="center", va="center", color=C_MUT, transform=ax_f.transAxes)
        ax_f.set_title("Top-20 trials: lr vs depth", fontsize=10)

    # Header
    fig.text(0.5, 0.98,
             "Optuna Hyperparameter Tuning  |  Quant Alpha Model (XGBoost)",
             ha="center", va="top", fontsize=14, color=C_TXT, fontweight="bold")
    fig.text(0.5, 0.955,
             f"Best trial Sharpe: {study.best_value:.4f}  |  "
             f"Completed trials: {len([t for t in study.trials if t.value is not None])}  |  "
             f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}",
             ha="center", va="top", fontsize=9, color=C_MUT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"\n  Chart saved → {output_path}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  OPTUNA QUANT ALPHA MODEL  —  Full Pipeline")
    print("="*60)

    # Generate data
    print("\n[1/5] Generating synthetic market data...")
    returns = generate_market_data(n_assets=50, n_days=1500)
    print(f"      {returns.shape[0]} trading days | {returns.shape[1]} assets")

    # Feature engineering
    print("[2/5] Engineering features...")
    X, y = engineer_features(returns)
    print(f"      Feature matrix: {X.shape}  |  Target: {y.shape}")
    print(f"      Features: {list(X.columns[:5])} ...")

    # Run Optuna
    print("[3/5] Running Optuna optimization...")
    study = run_study(X, y, n_trials=60)

    # Print best results
    print(f"\n[4/5] Optimization complete.")
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  BEST TRIAL RESULTS                  ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Trial #    : {study.best_trial.number:<24}║")
    print(f"  ║  Sharpe     : {study.best_value:<24.4f}║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  BEST HYPERPARAMETERS                ║")
    print(f"  ╠══════════════════════════════════════╣")
    for k, v in study.best_params.items():
        val_str = f"{v:.5g}" if isinstance(v, float) else str(v)
        print(f"  ║  {k:<18}: {val_str:<18}║")
    print(f"  ╚══════════════════════════════════════╝")

    # Final holdout test
    print("\n[5/5] Final out-of-sample evaluation (true holdout)...")
    oos = final_evaluation(X, y, study.best_params)
    print(f"\n  OOS Sharpe     : {oos['sharpe']:.4f}")
    print(f"  OOS Max DD     : {oos['max_drawdown']*100:.2f}%")
    print(f"  OOS Total Ret  : {oos['total_return']*100:.2f}%")
    print(f"  OOS Hit Rate   : {oos['hit_rate']*100:.1f}%")
    print(f"  OOS Rank IC    : {oos['ic']:.4f}")

    # Plot - Use current directory
    output_dir = os.path.join(os.getcwd(), "optuna_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"optuna_quant_results_{timestamp}.png")
    print(f"\n  Saving visualisation to: {out_path}")
    plot_results(study, oos, out_path)

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("="*60 + "\n")

    return study, oos


if __name__ == "__main__":
    study, oos_results = main()