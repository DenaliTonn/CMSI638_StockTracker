"""
googl_layer3_fixed.py
----------------------
Layer 3 fixed — three corrections from the first run:

    Fix 1: Standardise row count — no rows dropped due to macro NaN
    Fix 2: Z-score all macro signals so levels are comparable across time
    Fix 3: Add regime interaction features — beta × vix_regime,
           yield_curve × spy_trend, etc.

Also saves trained models as .pkl files for the agentic AI framework:
    models/googl_layer3_gradboost.pkl   — best model
    models/googl_layer3_scaler.pkl      — fitted scaler
    models/googl_layer3_features.pkl    — feature column list
    models/googl_layer3_rf.pkl          — RandomForest model
    models/googl_layer3_metadata.json   — training metadata

Run:
    python googl_layer3_fixed.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import json
import joblib
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── Create models directory ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)


# ── Fetch data ────────────────────────────────────────────────────────────────
def fetch_hourly():
    df = yf.download("GOOGL", period="2y", interval="1h",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_macro_daily():
    tickers = {"^VIX":"vix", "^TNX":"yield_10y", "^FVX":"yield_5y",
               "^IRX":"yield_3m", "DX-Y.NYB":"dxy", "SPY":"spy",
               "QQQ":"qqq", "GLD":"gold", "TLT":"tlt"}
    frames = {}
    for sym, name in tickers.items():
        try:
            raw = yf.download(sym, start="2023-01-01", interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty:
                frames[name] = raw["Close"]
        except:
            pass
    m = pd.DataFrame(frames)
    m.index = pd.to_datetime(m.index).tz_localize(None)
    return m


def build_macro_signals(m, hourly_index):
    out = pd.DataFrame(index=m.index)

    def zs(s, w=63):
        return (s - s.rolling(w, min_periods=21).mean()) / \
               (s.rolling(w, min_periods=21).std() + 1e-9)

    if "vix" in m:
        v = m["vix"]
        out["vix_zscore"]    = zs(v)
        out["vix_mom_5d"]    = v.pct_change(5)
        out["vix_mom_21d"]   = v.pct_change(21)
        out["vix_high"]      = (v > v.rolling(63, min_periods=21).mean()
                                 + v.rolling(63, min_periods=21).std()
                                 ).astype(int)

    if "yield_10y" in m and "yield_3m" in m:
        yc = m["yield_10y"] - m["yield_3m"]
        out["yield_curve_z"]  = zs(yc)
        out["yield_curve_mom"]= yc.diff(5)
        out["yield_10y_z"]    = zs(m["yield_10y"])
        out["yield_10y_mom"]  = m["yield_10y"].pct_change(21)

    if "yield_5y" in m and "yield_3m" in m:
        yc5 = m["yield_5y"] - m["yield_3m"]
        out["yield_5y3m_z"]   = zs(yc5)

    if "dxy" in m:
        d = m["dxy"]
        out["dxy_zscore"]     = zs(d)
        out["dxy_mom_5d"]     = d.pct_change(5)
        out["dxy_mom_21d"]    = d.pct_change(21)

    if "spy" in m:
        spy_ret = np.log(m["spy"] / m["spy"].shift(1))
        out["spy_ret_1d"]     = spy_ret
        out["spy_ret_5d"]     = spy_ret.rolling(5).sum()
        out["spy_ret_21d"]    = spy_ret.rolling(21).sum()
        out["spy_vol_21d_z"]  = zs(spy_ret.rolling(21).std() * np.sqrt(252))
        out["spy_trend"]      = (m["spy"] > m["spy"].rolling(63).mean()
                                 ).astype(int)

    if "qqq" in m and "spy" in m:
        ratio = m["qqq"] / m["spy"]
        out["tech_rs_z"]      = zs(ratio)
        out["tech_rs_mom"]    = ratio.pct_change(21)

    if "gold" in m and "spy" in m:
        out["gold_spy_z"]     = zs(m["gold"] / m["spy"])
    if "tlt" in m and "spy" in m:
        out["bond_eq_z"]      = zs(m["tlt"] / m["spy"])

    out = out.shift(1)
    out.index = pd.to_datetime(out.index)
    out_h = out.reindex(hourly_index, method="ffill")
    out_h = out_h.fillna(0)
    return out_h


def compute_technical(df):
    c = df["Close"]; o = df["Open"]
    h = df["High"];  l = df["Low"];  v = df["Volume"]
    out     = pd.DataFrame(index=df.index)
    log_ret = np.log(c / c.shift(1))

    def w(s, q=0.005):
        return s.clip(s.quantile(q), s.quantile(1-q))
    def zs(s, n=50):
        mu = s.rolling(n, min_periods=10).mean()
        sd = s.rolling(n, min_periods=10).std()
        return (s - mu) / (sd + 1e-9)
    def rsi(x, n=14):
        d = x.diff()
        g = d.clip(lower=0).rolling(n).mean()
        ls = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + g / (ls + 1e-9))

    out["rsi_14"]        = rsi(c, 14)
    out["rsi_50"]        = rsi(c, 50)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    out["macd_signal"]   = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"]     = w(macd - out["macd_signal"])
    out["ppo"]           = w(((ema12-ema26)/(ema26+1e-9))*100)
    out["momentum_7h"]   = w(log_ret.rolling(7).sum())
    out["momentum_35h"]  = w(log_ret.rolling(35).sum())
    out["momentum_147h"] = w(log_ret.rolling(147).sum())
    out["roc_7h"]        = w(c.pct_change(7)*100)
    out["roc_35h"]       = w(c.pct_change(35)*100)
    ma35  = c.rolling(35).mean()
    ma147 = c.rolling(147).mean()
    ma441 = c.rolling(441).mean()
    out["dist_ma35"]     = w((c-ma35)/(ma35+1e-9))
    out["dist_ma147"]    = w((c-ma147)/(ma147+1e-9))
    out["dist_ma441"]    = w((c-ma441)/(ma441+1e-9))
    bb_std = c.rolling(147).std()
    out["bb_position"]   = w((c-(ma147-2*bb_std))/(4*bb_std+1e-9))
    roll_max = c.rolling(1764, min_periods=441).max()
    out["pct_from_high"] = w((c-roll_max)/(roll_max+1e-9))
    out["vol_7h"]        = log_ret.rolling(7).std()*np.sqrt(1764)*100
    out["vol_35h"]       = log_ret.rolling(35).std()*np.sqrt(1764)*100
    out["vol_147h"]      = log_ret.rolling(147).std()*np.sqrt(1764)*100
    out["vol_accel"]     = (out["vol_7h"]/(out["vol_35h"]+1e-9)).clip(0,5)
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],
                   axis=1).max(axis=1)
    out["atr_norm"]      = zs(tr.rolling(14).mean())
    vol_h = out["vol_35h"]/100/np.sqrt(1764)
    out["vol_norm_ret"]  = w(log_ret/(vol_h+1e-9))
    out["sq_ret_lag1"]   = (log_ret**2).shift(1)
    out["overnight_gap"] = w((o-c.shift(1))/(c.shift(1)+1e-9))
    out["oc_direction"]  = w((c-o)/(o+1e-9))
    body_top = pd.concat([o,c],axis=1).max(axis=1)
    body_bot = pd.concat([o,c],axis=1).min(axis=1)
    out["upper_wick"]    = w((h-body_top)/(c+1e-9))
    out["lower_wick"]    = w((body_bot-l)/(c+1e-9))
    out["hl_range"]      = w((h-l)/(c+1e-9))
    vol_ma = v.rolling(35).mean()
    out["vol_ratio"]     = w(v/(vol_ma+1))
    obv    = (np.sign(log_ret)*v).cumsum()
    obv_ma = obv.rolling(35).mean()
    out["obv_signal"]    = w((obv-obv_ma)/(obv_ma.abs()+1))
    tp  = (h+l+c)/3
    mf  = tp*v
    pf  = mf.where(tp>tp.shift(1),0).rolling(14).sum()
    nf  = mf.where(tp<tp.shift(1),0).rolling(14).sum()
    out["mfi"]           = 100-100/(1+pf/(nf+1e-9))
    clv = ((c-l)-(h-c))/((h-l)+1e-9)
    out["chaikin_ad"]    = w(zs((clv*v).cumsum()))
    out["amihud"]        = w(log_ret.abs()/((c*v)+1e-9)*1e9)
    return out


def add_interactions(X):
    df = X.copy()
    if "market_beta_h" in df.columns and "spy_trend" in df.columns:
        df["beta_x_trend"] = df["market_beta_h"] * df["spy_trend"]
        df["beta_x_vix"]   = df["market_beta_h"] * (1 - df.get("vix_high", 0))
    if "momentum_147h" in df.columns and "vix_high" in df.columns:
        df["mom_x_lowvix"]  = df["momentum_147h"] * (1 - df["vix_high"])
        df["mom_x_highvix"] = df["momentum_147h"] * df["vix_high"]
    if "yield_curve_z" in df.columns and "spy_trend" in df.columns:
        df["yc_x_trend"] = df["yield_curve_z"] * df["spy_trend"]
    if "tech_rs_z" in df.columns and "momentum_35h" in df.columns:
        df["techrs_x_mom"] = df["tech_rs_z"] * df["momentum_35h"]
    if "pct_from_high" in df.columns and "vix_high" in df.columns:
        df["fall_x_vix"] = df["pct_from_high"] * df["vix_high"]
    return df


def run_models(X_tr, y_tr, X_te, y_te, feat_cols):
    sc   = StandardScaler()
    Xtr  = sc.fit_transform(X_tr)
    Xte  = sc.transform(X_te)
    res  = {}

    for name, model in [
        ("Ridge",        Ridge(alpha=0.5)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=200, max_depth=7,
            min_samples_leaf=15, n_jobs=-1, random_state=42)),
        ("GradBoost",    GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_samples_leaf=15, subsample=0.8, random_state=42)),
    ]:
        model.fit(Xtr, y_tr)
        pred    = model.predict(Xte)
        ic, _   = spearmanr(pred, y_te)
        hit     = np.mean(np.sign(pred) == np.sign(y_te)) * 100
        res[name] = {"pred": pred, "ic": ic, "hit": hit, "model": model}
        if hasattr(model, "feature_importances_"):
            res[name]["imp"] = dict(zip(feat_cols, model.feature_importances_))
        elif hasattr(model, "coef_"):
            res[name]["coef"] = dict(zip(feat_cols, model.coef_))
        print(f"  {name:<15} IC: {ic:+.4f}   Hit: {hit:.1f}%")

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=5,
            min_child_samples=15, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(Xtr, y_tr)
        pred  = m.predict(Xte)
        ic, _ = spearmanr(pred, y_te)
        hit   = np.mean(np.sign(pred) == np.sign(y_te)) * 100
        res["LightGBM"] = {"pred": pred, "ic": ic, "hit": hit, "model": m,
                           "imp": dict(zip(feat_cols, m.feature_importances_))}
        print(f"  {'LightGBM':<15} IC: {ic:+.4f}   Hit: {hit:.1f}%")
    except ImportError:
        pass

    return res, sc


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  GOOGL LAYERS 1+2+3 — FIXED VERSION")
    print("=" * 60)

    print("\nFetching data...")
    df           = fetch_hourly()
    macro_daily  = fetch_macro_daily()
    macro_hourly = build_macro_signals(macro_daily, df.index)

    spy_h = yf.download("SPY", period="2y", interval="1h",
                         auto_adjust=True, progress=False)
    if isinstance(spy_h.columns, pd.MultiIndex):
        spy_h.columns = spy_h.columns.get_level_values(0)
    spy_h.index = pd.to_datetime(spy_h.index).tz_localize(None)
    spy_ret_h    = np.log(spy_h["Close"] / spy_h["Close"].shift(1))
    googl_ret_h  = np.log(df["Close"] / df["Close"].shift(1))
    spy_aligned  = spy_ret_h.reindex(df.index, method="nearest")
    cov_h        = googl_ret_h.rolling(147).cov(spy_aligned)
    var_h        = spy_aligned.rolling(147).var()
    macro_hourly["market_beta_h"] = (cov_h / (var_h + 1e-9)).fillna(1.0)
    print(f"  Added GOOGL hourly beta to SPY")

    tech_df = compute_technical(df)

    info = yf.Ticker("GOOGL").info
    def get(k, d=np.nan): return info.get(k,d) or d
    funds = {
        "pe_ratio": get("trailingPE"), "forward_pe": get("forwardPE"),
        "price_to_book": get("priceToBook"),
        "profit_margin": get("profitMargins"), "roe": get("returnOnEquity"),
        "revenue_growth": get("revenueGrowth"),
        "earnings_growth": get("earningsGrowth"),
        "beta": get("beta"), "recommendation": get("recommendationMean"),
    }

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    target  = log_ret.rolling(24).sum().shift(-24)

    X = pd.concat([tech_df, macro_hourly], axis=1)
    for k, v in funds.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            X[f"fund_{k}"] = v
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    X["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    X = add_interactions(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X["target"] = target
    X["close"]  = df["Close"]
    X = X.dropna(subset=["target"])

    feat_cols = [c for c in X.columns
                 if c not in ["target","close"]
                 and X[c].std() > 1e-6]
    X[feat_cols] = X[feat_cols].fillna(0)

    print(f"\n  Feature matrix: {X.shape[0]} rows × {len(feat_cols)} features")
    print(f"  (Fixed: {X.shape[0]} rows vs 2127 in broken version)")

    split    = int(len(X) * 0.70)
    X_tr     = X[feat_cols].values[:split]
    y_tr     = X["target"].values[:split]
    X_te     = X[feat_cols].values[split:]
    y_te     = X["target"].values[split:]
    dates_te = X.index[split:]
    close_te = X["close"].values[split:]

    print(f"\n  Train: {X.index[0].date()} → {X.index[split-1].date()}")
    print(f"  Test : {X.index[split].date()} → {X.index[-1].date()}")

    print(f"\n── MODEL RESULTS ────────────────────────────────────────────────")
    results, scaler = run_models(X_tr, y_tr, X_te, y_te, feat_cols)

    # ── SAVE MODELS ───────────────────────────────────────────────────────────
    print(f"\n── SAVING MODELS ────────────────────────────────────────────────")

    # Scaler — must match what was fitted on training data
    joblib.dump(scaler, "models/googl_layer3_scaler.pkl")
    print(f"  Saved: models/googl_layer3_scaler.pkl")

    # Feature columns — partner needs exact same list in same order
    joblib.dump(feat_cols, "models/googl_layer3_features.pkl")
    print(f"  Saved: models/googl_layer3_features.pkl  ({len(feat_cols)} features)")

    # GradBoost — primary model for agentic framework
    if "GradBoost" in results:
        joblib.dump(results["GradBoost"]["model"],
                    "models/googl_layer3_gradboost.pkl")
        print(f"  Saved: models/googl_layer3_gradboost.pkl  "
              f"(IC={results['GradBoost']['ic']:+.4f}  "
              f"Hit={results['GradBoost']['hit']:.1f}%)")

    # RandomForest — for comparison
    if "RandomForest" in results:
        joblib.dump(results["RandomForest"]["model"],
                    "models/googl_layer3_rf.pkl")
        print(f"  Saved: models/googl_layer3_rf.pkl  "
              f"(IC={results['RandomForest']['ic']:+.4f}  "
              f"Hit={results['RandomForest']['hit']:.1f}%)")

    # Metadata JSON — human readable context for the agent
    metadata = {
        "ticker":        "GOOGL",
        "layers":        "Technical + Macro + Regime Interactions",
        "train_start":   str(X.index[0].date()),
        "train_end":     str(X.index[split-1].date()),
        "test_start":    str(X.index[split].date()),
        "test_end":      str(X.index[-1].date()),
        "n_features":    len(feat_cols),
        "n_train_rows":  split,
        "n_test_rows":   len(X) - split,
        "gradboost_ic":  round(results.get("GradBoost",{}).get("ic",0), 4),
        "gradboost_hit": round(results.get("GradBoost",{}).get("hit",0), 1),
        "rf_ic":         round(results.get("RandomForest",{}).get("ic",0), 4),
        "rf_hit":        round(results.get("RandomForest",{}).get("hit",0), 1),
        "prediction_target": "24h forward log return",
        "signal_interpretation": {
            "positive": "BUY — model predicts price higher in 24h",
            "negative": "SELL — model predicts price lower in 24h",
            "near_zero": "FLAT — prediction too weak to trade"
        },
        "feature_cols":  feat_cols,
    }
    with open("models/googl_layer3_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: models/googl_layer3_metadata.json")

    print(f"""
  ── HOW TO LOAD IN AGENTIC FRAMEWORK ──────────────────────
  import joblib, numpy as np

  gb     = joblib.load('models/googl_layer3_gradboost.pkl')
  scaler = joblib.load('models/googl_layer3_scaler.pkl')
  cols   = joblib.load('models/googl_layer3_features.pkl')

  # feature_values must be a dict with same keys as cols
  values = [feature_values[c] for c in cols]
  scaled = scaler.transform([values])
  pred   = float(gb.predict(scaled)[0])
  signal = int(np.sign(pred))  # +1 buy, -1 sell, 0 flat
  ──────────────────────────────────────────────────────────
""")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── SUMMARY ──────────────────────────────────────────────────────")
    print(f"  Layer 2 baseline: Ridge +0.1299  RF +0.1412")
    print(f"\n  {'Model':<15} {'IC':>8} {'Hit%':>8}  {'Improvement':>12}")
    print(f"  {'-'*50}")
    base = {"Ridge": 0.1299, "RandomForest": 0.1412}
    for name, res in results.items():
        diff = res["ic"] - base.get(name, 0.13)
        tag  = f"{diff:+.4f}" if name in base else "new"
        v    = "✅" if res["ic"] > 0.10 else "⚠ " if res["ic"] > 0.05 else "❌"
        print(f"  {v} {name:<13} {res['ic']:>+8.4f} {res['hit']:>7.1f}%  {tag:>12}")

    # Top features
    best_name = max(results, key=lambda k: abs(results[k]["ic"]))
    best      = results[best_name]
    if "imp" in best:
        ranked = sorted(best["imp"].items(), key=lambda x:x[1], reverse=True)
        print(f"\n── TOP 20 FEATURES ({best_name}) ──────────────────────────────")
        macro_set  = set(macro_hourly.columns)
        fund_set   = {f"fund_{k}" for k in funds}
        inter_set  = {"beta_x_trend","beta_x_vix","mom_x_lowvix",
                      "mom_x_highvix","yc_x_trend","techrs_x_mom","fall_x_vix"}
        print(f"  {'Feature':<28} {'Imp':>8}  Layer")
        print(f"  {'-'*50}")
        for name, imp in ranked[:20]:
            if name in inter_set:    layer = "🔗 INTERACT"
            elif name in macro_set:  layer = "📊 MACRO"
            elif name in fund_set:   layer = "📈 FUND"
            elif name in {"hour_sin","hour_cos","dow_sin","dow_cos"}:
                                     layer = "🕐 TIME"
            else:                    layer = "📉 TECH"
            bar = "█" * int(imp * 300)
            print(f"  {name:<28} {imp:>8.4f}  {layer}  {bar}")

    # Plot
    fig = plt.figure(figsize=(16,12))
    fig.suptitle("GOOGL — Layers 1+2+3 Fixed (Technical + Macro + Interactions)",
                 fontsize=12, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)
    pal = {"Ridge":"#6366f1","RandomForest":"#16a34a",
           "GradBoost":"#f59e0b","LightGBM":"#ef4444"}

    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(dates_te, close_te, lw=1, color="#2563eb", label="GOOGL")
    if "spy_trend" in X.columns:
        tr = X["spy_trend"].reindex(dates_te)
        ax1.fill_between(dates_te, close_te.min(), close_te.max(),
                         where=tr==1, alpha=0.08, color="green",
                         label="Bull regime")
        ax1.fill_between(dates_te, close_te.min(), close_te.max(),
                         where=tr==0, alpha=0.08, color="red",
                         label="Bear regime")
    ax1.set_title("GOOGL price — test period with market regime overlay")
    ax1.set_ylabel("Price ($)")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=30)

    ax2 = fig.add_subplot(gs[1,0])
    br  = results[best_name]
    ax2.scatter(br["pred"], y_te, alpha=0.1, s=5,
                color=pal.get(best_name,"#888"))
    z   = np.polyfit(br["pred"], y_te, 1)
    xs  = np.linspace(br["pred"].min(), br["pred"].max(), 100)
    ax2.plot(xs, np.polyval(z,xs), color="red", lw=2)
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.axvline(0, color="gray", lw=0.5, ls="--")
    ax2.set_xlabel("Predicted 24h return")
    ax2.set_ylabel("Actual 24h return")
    ax2.set_title(f"{best_name}: IC={br['ic']:+.4f}  Hit={br['hit']:.1f}%")

    ax3 = fig.add_subplot(gs[1,1])
    names = list(results.keys())
    ics   = [results[m]["ic"] for m in names]
    ax3.bar(range(len(names)), ics,
            color=[pal.get(n,"#888") for n in names], alpha=0.8)
    ax3.axhline(0.13, color="purple", lw=1.5, ls="--",
                label="Layer 2 baseline")
    ax3.axhline(0.05, color="green",  lw=1,   ls=":",
                label="IC 0.05 threshold")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=15)
    ax3.set_ylabel("IC")
    ax3.set_title("Model IC comparison")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[2,0])
    if "market_beta_h" in X.columns:
        beta = X["market_beta_h"].reindex(dates_te)
        ax4.plot(dates_te, beta, lw=0.8, color="#7c3aed",
                 label="GOOGL beta to SPY")
        ax4.axhline(1.0, color="gray", lw=0.5, ls="--")
        ax4.set_ylabel("Beta", color="#7c3aed")
        ax4b = ax4.twinx()
        if "vix_zscore" in X.columns:
            vz = X["vix_zscore"].reindex(dates_te)
            ax4b.plot(dates_te, vz, lw=0.8, color="#dc2626",
                      alpha=0.6, label="VIX z-score")
            ax4b.axhline(0, color="gray", lw=0.5, ls="--")
            ax4b.set_ylabel("VIX z-score", color="#dc2626")
    ax4.set_title("GOOGL beta and VIX z-score — test period")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.tick_params(axis="x", rotation=30)

    ax5 = fig.add_subplot(gs[2,1])
    lrt = np.log(close_te[1:] / close_te[:-1])
    for name, res in results.items():
        sig   = np.sign(res["pred"][:-1])
        strat = (sig * lrt - 0.0001).cumsum()
        ax5.plot(dates_te[1:], strat, lw=1.2,
                 label=f"{name} ({res['ic']:+.3f})",
                 color=pal.get(name,"#888"))
    bh = lrt.cumsum()
    ax5.plot(dates_te[1:], bh, lw=1, ls="--",
             color="#2563eb", alpha=0.7, label="Buy & hold")
    ax5.axhline(0, color="gray", lw=0.5)
    ax5.set_title("Cumulative returns — all models vs buy & hold")
    ax5.set_ylabel("Cumulative log return")
    ax5.legend(fontsize=7)
    ax5.tick_params(axis="x", rotation=30)

    plt.savefig("googl_layer3_fixed.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: googl_layer3_fixed.png")

    print(f"\n── LAYER 3 MODEL SUMMARY ─────────────────────────────────────────")
    best_mod = max(results, key=lambda k: results[k]["ic"])
    best_ic  = results[best_mod]["ic"]
    best_hit = results[best_mod]["hit"]
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Best model    : {best_mod:<23} │")
    print(f"  │  IC  (Layer 3) : {best_ic:>+.3f}                   │")
    print(f"  │  Hit Rate      : {best_hit:>5.1f}%                  │")
    print(f"  │  (random=0.00, real edge=0.03)          │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  All models:")
    for name, res in results.items():
        print(f"    {name:<15} IC={res['ic']:>+.3f}  Hit={res['hit']:.1f}%")

    print(f"\n── FILES SAVED ───────────────────────────────────────────────────")
    print(f"  models/googl_layer3_gradboost.pkl   — GradBoost model")
    print(f"  models/googl_layer3_rf.pkl           — RandomForest model")
    print(f"  models/googl_layer3_scaler.pkl       — fitted scaler")
    print(f"  models/googl_layer3_features.pkl     — feature column list")
    print(f"  models/googl_layer3_metadata.json    — training metadata + usage")
    print(f"\n  Layer 3 complete.")
    print(f"  Next: python googl_layer4_v2.py")