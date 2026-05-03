from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
import heapq
import json
import os
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ── ML imports with graceful fallbacks ──────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

app = Flask(__name__, static_folder='static')

portfolio    = []
history_log  = []
HISTORY_FILE = 'history.json'

if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r') as f:
            history_log = json.load(f)
    except Exception:
        history_log = []

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_log, f)


# ════════════════════════════════════════════════════════════════════════════
#  INDICATOR LIBRARY
# ════════════════════════════════════════════════════════════════════════════
class Indicators:

    @staticmethod
    def ema(data, n):
        k = 2 / (n + 1)
        result = [float('nan')] * len(data)
        start  = n - 1
        if start >= len(data):
            return result
        result[start] = sum(data[:n]) / n
        for i in range(start + 1, len(data)):
            prev = result[i-1]
            result[i] = data[i] * k + prev * (1 - k) if prev == prev else data[i]
        return result

    @staticmethod
    def sma(data, n):
        result = [float('nan')] * len(data)
        for i in range(n - 1, len(data)):
            result[i] = sum(data[i-n+1:i+1]) / n
        return result

    @staticmethod
    def rsi(closes, period=14):
        result = [float('nan')] * len(closes)
        if len(closes) < period + 1:
            return result
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses[:period]) / period
        for i in range(period, len(closes)):
            if avg_l == 0:
                result[i] = 100.0
            else:
                rs = avg_g / avg_l
                result[i] = round(100 - (100 / (1 + rs)), 2)
            if i < len(closes) - 1:
                d     = closes[i] - closes[i-1]
                avg_g = (avg_g * (period-1) + max(d,  0)) / period
                avg_l = (avg_l * (period-1) + max(-d, 0)) / period
        return result

    @staticmethod
    def macd(closes):
        ema12   = Indicators.ema(closes, 12)
        ema26   = Indicators.ema(closes, 26)
        macd_l  = [
            (ema12[i] - ema26[i]) if (ema12[i] == ema12[i] and ema26[i] == ema26[i])
            else float('nan')
            for i in range(len(closes))
        ]
        valid   = [(i, v) for i, v in enumerate(macd_l) if v == v]
        sig_l   = [float('nan')] * len(closes)
        hist_l  = [float('nan')] * len(closes)
        if len(valid) >= 9:
            idxs = [x[0] for x in valid]
            vals = [x[1] for x in valid]
            sig  = Indicators.ema(vals, 9)
            for j, idx in enumerate(idxs):
                sig_l[idx]  = sig[j] if j < len(sig) and sig[j] == sig[j] else float('nan')
                hist_l[idx] = (macd_l[idx] - sig_l[idx]) if sig_l[idx] == sig_l[idx] else float('nan')
        return macd_l, sig_l, hist_l

    @staticmethod
    def bollinger(closes, n=20, k=2):
        upper = [float('nan')] * len(closes)
        lower = [float('nan')] * len(closes)
        mid   = Indicators.sma(closes, n)
        for i in range(n-1, len(closes)):
            window = closes[i-n+1:i+1]
            std    = (sum((x - mid[i])**2 for x in window) / n) ** 0.5
            upper[i] = mid[i] + k * std
            lower[i] = mid[i] - k * std
        return upper, mid, lower

    @staticmethod
    def atr(closes, highs, lows, period=14):
        result = [float('nan')] * len(closes)
        if len(closes) < period + 1:
            return result
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        result[period] = sum(trs[:period]) / period
        for i in range(period+1, len(closes)):
            result[i] = (result[i-1] * (period-1) + trs[i-1]) / period
        return result

    @staticmethod
    def stochastic(closes, highs, lows, k=14, d=3):
        stoch_k = [float('nan')] * len(closes)
        stoch_d = [float('nan')] * len(closes)
        for i in range(k-1, len(closes)):
            lo  = min(lows[i-k+1:i+1])
            hi  = max(highs[i-k+1:i+1])
            if hi != lo:
                stoch_k[i] = ((closes[i] - lo) / (hi - lo)) * 100
        valid_k = [(i, v) for i, v in enumerate(stoch_k) if v == v]
        if len(valid_k) >= d:
            for j in range(d-1, len(valid_k)):
                stoch_d[valid_k[j][0]] = sum(valid_k[j-d+1+x][1] for x in range(d)) / d
        return stoch_k, stoch_d

    @staticmethod
    def williams_r(closes, highs, lows, period=14):
        result = [float('nan')] * len(closes)
        for i in range(period-1, len(closes)):
            hi = max(highs[i-period+1:i+1])
            lo = min(lows[i-period+1:i+1])
            if hi != lo:
                result[i] = ((hi - closes[i]) / (hi - lo)) * -100
        return result

    @staticmethod
    def obv(closes, volumes):
        result = [0] * len(closes)
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                result[i] = result[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                result[i] = result[i-1] - volumes[i]
            else:
                result[i] = result[i-1]
        return result

    @staticmethod
    def vwap(closes, highs, lows, volumes):
        result = [float('nan')] * len(closes)
        cum_pv = cum_v = 0
        for i in range(len(closes)):
            tp      = (highs[i] + lows[i] + closes[i]) / 3
            cum_pv += tp * volumes[i]
            cum_v  += volumes[i]
            result[i] = cum_pv / cum_v if cum_v else float('nan')
        return result

    @staticmethod
    def support_resistance(closes, n=20):
        """Simple pivot-based S/R"""
        pivots_hi, pivots_lo = [], []
        for i in range(n, len(closes)-n):
            window = closes[i-n:i+n+1]
            if closes[i] == max(window):
                pivots_hi.append(closes[i])
            if closes[i] == min(window):
                pivots_lo.append(closes[i])
        resistance = round(sorted(pivots_hi)[-1], 2) if pivots_hi else None
        support    = round(sorted(pivots_lo)[0],  2) if pivots_lo else None
        return support, resistance

    @staticmethod
    def adx(closes, highs, lows, period=14):
        """Average Directional Index"""
        if len(closes) < period * 2:
            return [float('nan')] * len(closes)
        tr_list, dm_plus, dm_minus = [], [], []
        for i in range(1, len(closes)):
            tr  = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            up  = highs[i]  - highs[i-1]
            dn  = lows[i-1] - lows[i]
            tr_list.append(tr)
            dm_plus.append(up  if up > dn  and up  > 0 else 0)
            dm_minus.append(dn if dn > up  and dn  > 0 else 0)

        def smooth(data, p):
            s = [0.0] * len(data)
            s[p-1] = sum(data[:p])
            for i in range(p, len(data)):
                s[i] = s[i-1] - (s[i-1]/p) + data[i]
            return s

        atr_s  = smooth(tr_list,   period)
        dmp_s  = smooth(dm_plus,   period)
        dmm_s  = smooth(dm_minus,  period)
        adx_r  = [float('nan')] * len(closes)
        dx_list = []
        for i in range(period, len(tr_list)+1):
            if atr_s[i-1] == 0:
                continue
            pdi = 100 * dmp_s[i-1] / atr_s[i-1]
            mdi = 100 * dmm_s[i-1] / atr_s[i-1]
            dx  = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) else 0
            dx_list.append(dx)
        if len(dx_list) >= period:
            adx_val = sum(dx_list[:period]) / period
            adx_r[period*2] = adx_val
            for i in range(period*2+1, len(closes)):
                j = i - period - 1
                if j < len(dx_list):
                    adx_val = (adx_val*(period-1) + dx_list[j]) / period
                    adx_r[i] = adx_val
        return adx_r


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
    'vol_3d', 'vol_5d', 'vol_10d',
    'rsi_14', 'rsi_7',
    'macd_val', 'macd_hist', 'macd_cross',
    'bb_pct', 'bb_width',
    'stoch_k', 'stoch_d',
    'williams_r',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma50_ratio',
    'ema12_ratio', 'ema26_ratio',
    'volume_ratio_5d', 'volume_ratio_20d',
    'obv_slope',
    'atr_pct',
    'price_vs_52h', 'price_vs_52l',
    'adx',
    'momentum_5', 'momentum_10',
    'candle_body', 'upper_shadow', 'lower_shadow',
]


def build_feature_matrix(hist):
    closes  = hist['Close'].tolist()
    highs   = hist['High'].tolist()
    lows    = hist['Low'].tolist()
    volumes = hist['Volume'].tolist()
    opens   = hist['Open'].tolist()

    # Indicators
    rsi14   = Indicators.rsi(closes, 14)
    rsi7    = Indicators.rsi(closes, 7)
    macd_l, sig_l, hist_l = Indicators.macd(closes)
    bb_u, bb_m, bb_l = Indicators.bollinger(closes, 20)
    ema12   = Indicators.ema(closes, 12)
    ema26   = Indicators.ema(closes, 26)
    ema50   = Indicators.ema(closes, 50)
    sma5    = Indicators.sma(closes, 5)
    sma10   = Indicators.sma(closes, 10)
    sma20   = Indicators.sma(closes, 20)
    sma50   = Indicators.sma(closes, 50)
    stoch_k, stoch_d = Indicators.stochastic(closes, highs, lows)
    will_r  = Indicators.williams_r(closes, highs, lows)
    obv_v   = Indicators.obv(closes, volumes)
    atr_v   = Indicators.atr(closes, highs, lows)
    adx_v   = Indicators.adx(closes, highs, lows)

    vol_sma5  = Indicators.sma(volumes, 5)
    vol_sma20 = Indicators.sma(volumes, 20)

    hi52 = max(highs)
    lo52 = min(lows)

    X, y, dates_out = [], [], []

    WARMUP = 55
    for i in range(WARMUP, len(closes) - 1):
        c   = closes[i]
        # Skip if any core indicator is NaN
        core = [rsi14[i], macd_l[i], sig_l[i], ema12[i], ema26[i], ema50[i], sma20[i], atr_v[i]]
        if any(v != v for v in core):
            continue

        # Returns
        r1  = (closes[i] - closes[i-1]) / closes[i-1]
        r3  = (closes[i] - closes[i-3]) / closes[i-3] if i >= 3 else 0
        r5  = (closes[i] - closes[i-5]) / closes[i-5] if i >= 5 else 0
        r10 = (closes[i] - closes[i-10])/ closes[i-10] if i >= 10 else 0

        # Volatility (rolling std of returns)
        def rolling_vol(n):
            if i < n: return 0
            rets = [(closes[j]-closes[j-1])/closes[j-1] for j in range(i-n+1, i+1)]
            m    = sum(rets)/n
            return (sum((r-m)**2 for r in rets)/n)**0.5

        vol3  = rolling_vol(3)
        vol5  = rolling_vol(5)
        vol10 = rolling_vol(10)

        # Bollinger %B and width
        if bb_u[i] == bb_u[i] and bb_l[i] == bb_l[i] and (bb_u[i]-bb_l[i]) != 0:
            bb_pct   = (c - bb_l[i]) / (bb_u[i] - bb_l[i])
            bb_width = (bb_u[i] - bb_l[i]) / bb_m[i] if bb_m[i] else 0
        else:
            bb_pct = bb_width = 0

        # MACD cross (1 if just crossed up, -1 down, else 0)
        prev_ml = macd_l[i-1] if i > 0 and macd_l[i-1]==macd_l[i-1] else macd_l[i]
        prev_sl = sig_l[i-1]  if i > 0 and sig_l[i-1]==sig_l[i-1]   else sig_l[i]
        if macd_l[i] > sig_l[i] and prev_ml <= prev_sl: macd_cross = 1
        elif macd_l[i] < sig_l[i] and prev_ml >= prev_sl: macd_cross = -1
        else: macd_cross = 0

        # Stochastic
        sk = stoch_k[i] if stoch_k[i]==stoch_k[i] else 50
        sd = stoch_d[i] if stoch_d[i]==stoch_d[i] else 50
        wr = will_r[i]  if will_r[i]==will_r[i]   else -50

        # MA ratios
        ma5r  = (c / sma5[i]  - 1) if sma5[i]==sma5[i]   and sma5[i]   else 0
        ma10r = (c / sma10[i] - 1) if sma10[i]==sma10[i]  and sma10[i]  else 0
        ma20r = (c / sma20[i] - 1) if sma20[i]==sma20[i]  and sma20[i]  else 0
        ma50r = (c / sma50[i] - 1) if sma50[i]==sma50[i]  and sma50[i]  else 0
        e12r  = (c / ema12[i] - 1) if ema12[i]==ema12[i]  and ema12[i]  else 0
        e26r  = (c / ema26[i] - 1) if ema26[i]==ema26[i]  and ema26[i]  else 0

        # Volume ratios
        vs5  = (volumes[i] / vol_sma5[i]  - 1) if vol_sma5[i]==vol_sma5[i]   and vol_sma5[i]  else 0
        vs20 = (volumes[i] / vol_sma20[i] - 1) if vol_sma20[i]==vol_sma20[i]  and vol_sma20[i] else 0

        # OBV slope (5-period)
        obv_slope = (obv_v[i] - obv_v[i-5]) / (abs(obv_v[i-5])+1) if i >= 5 else 0

        # ATR %
        atr_pct = atr_v[i] / c if c else 0

        # 52-week
        p52h = (c - hi52) / hi52 if hi52 else 0
        p52l = (c - lo52) / lo52 if lo52 else 0

        # ADX
        adx_val = adx_v[i] if adx_v[i]==adx_v[i] else 20

        # Momentum
        mom5  = (closes[i] - closes[i-5])  / closes[i-5]  if i >= 5  and closes[i-5]  else 0
        mom10 = (closes[i] - closes[i-10]) / closes[i-10] if i >= 10 and closes[i-10] else 0

        # Candlestick body / shadows
        body  = (c - opens[i]) / opens[i] if opens[i] else 0
        upper = (highs[i] - max(c, opens[i])) / opens[i] if opens[i] else 0
        lower = (min(c, opens[i]) - lows[i]) / opens[i]  if opens[i] else 0

        features = [
            r1, r3, r5, r10,
            vol3, vol5, vol10,
            rsi14[i], rsi7[i] if rsi7[i]==rsi7[i] else 50,
            macd_l[i], hist_l[i] if hist_l[i]==hist_l[i] else 0, macd_cross,
            bb_pct, bb_width,
            sk, sd, wr,
            ma5r, ma10r, ma20r, ma50r,
            e12r, e26r,
            vs5, vs20,
            obv_slope,
            atr_pct,
            p52h, p52l,
            adx_val,
            mom5, mom10,
            body, upper, lower,
        ]

        label = 1 if closes[i+1] > closes[i] else 0
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ════════════════════════════════════════════════════════════════════════════
#  LSTM SEQUENCE MODEL
# ════════════════════════════════════════════════════════════════════════════
def build_lstm_sequences(X, y, seq_len=10):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train_seq, y_train_seq, n_features):
    model = Sequential([
        LSTM(64, input_shape=(X_train_seq.shape[1], n_features), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=16,
              validation_split=0.1, callbacks=[es], verbose=0)
    return model


# ════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE ENGINE
# ════════════════════════════════════════════════════════════════════════════
class EnsembleEngine:

    def __init__(self, symbol):
        self.symbol = symbol.upper()

    # ── Data fetch ──────────────────────────────────────────────────────────
    def fetch(self, period='5y'):
        stock = yf.Ticker(self.symbol)
        hist  = stock.history(period=period)
        if hist.empty:
            hist = stock.history(period='1y')
        if hist.empty:
            raise Exception("Invalid symbol or no data.")
        self._hist = hist
        closes     = hist['Close'].dropna().tolist()
        prices6    = [round(p,2) for p in closes[-6:]]
        try:    current = round(stock.fast_info['last_price'], 2)
        except: current = prices6[-1]
        info = {}
        try:
            raw  = stock.info
            info = {
                'pe_ratio':       raw.get('trailingPE'),
                'pb_ratio':       raw.get('priceToBook'),
                'market_cap':     raw.get('marketCap'),
                'roe':            raw.get('returnOnEquity'),
                'debt_to_equity': raw.get('debtToEquity'),
                'eps':            raw.get('trailingEps'),
                'dividend_yield': raw.get('dividendYield'),
                'revenue_growth': raw.get('revenueGrowth'),
                'profit_margin':  raw.get('profitMargins'),
                'current_ratio':  raw.get('currentRatio'),
                'beta':           raw.get('beta'),
                'name':           raw.get('longName', self.symbol),
                'sector':         raw.get('sector', 'N/A'),
                'industry':       raw.get('industry', 'N/A'),
                '52w_high':       raw.get('fiftyTwoWeekHigh'),
                '52w_low':        raw.get('fiftyTwoWeekLow'),
                'analyst_target': raw.get('targetMeanPrice'),
                'recommendation': raw.get('recommendationKey',''),
            }
        except Exception:
            info = {'name': self.symbol}
        return prices6, current, info

    # ── Train ensemble ───────────────────────────────────────────────────────
    def train_ensemble(self):
        hist = self._hist
        if len(hist) < 80:
            raise Exception("Need at least 80 trading days of data.")

        X, y = build_feature_matrix(hist)
        if len(X) < 40:
            raise Exception("Not enough processed data for training.")

        split      = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        models_info = {}

        # ── Random Forest ──
        rf  = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                     max_features='sqrt', random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        rf_acc = round(accuracy_score(y_te, rf.predict(X_te_s))*100, 1)
        models_info['rf'] = {'model': rf, 'acc': rf_acc, 'weight': 1.0}

        # ── GradientBoosting ──
        gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=4, subsample=0.8, random_state=42)
        gb.fit(X_tr_s, y_tr)
        gb_acc = round(accuracy_score(y_te, gb.predict(X_te_s))*100, 1)
        models_info['gb'] = {'model': gb, 'acc': gb_acc, 'weight': 1.2}

        # ── XGBoost (if available) ──
        if HAS_XGB:
            try:
                xgb_m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.1, reg_lambda=1.0,
                                           use_label_encoder=False, eval_metric='logloss',
                                           random_state=42, n_jobs=-1)
                xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
                xgb_acc = round(accuracy_score(y_te, xgb_m.predict(X_te_s))*100, 1)
                models_info['xgb'] = {'model': xgb_m, 'acc': xgb_acc, 'weight': 1.5}
            except Exception:
                pass

        # ── LightGBM (if available) ──
        if HAS_LGB:
            try:
                lgb_m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                            max_depth=5, num_leaves=31,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=42, n_jobs=-1, verbose=-1)
                lgb_m.fit(X_tr_s, y_tr,
                          eval_set=[(X_te_s, y_te)],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
                lgb_acc = round(accuracy_score(y_te, lgb_m.predict(X_te_s))*100, 1)
                models_info['lgb'] = {'model': lgb_m, 'acc': lgb_acc, 'weight': 1.5}
            except Exception:
                pass

        # ── LSTM (if available) ──
        lstm_model = None
        lstm_acc   = None
        if HAS_LSTM and len(X_tr_s) >= 50:
            try:
                SEQ = 10
                X_tr_seq, y_tr_seq = build_lstm_sequences(X_tr_s, y_tr, SEQ)
                X_te_seq, y_te_seq = build_lstm_sequences(X_te_s, y_te, SEQ)
                if len(X_tr_seq) >= 20:
                    lstm_model = train_lstm(X_tr_seq, y_tr_seq, X_tr_s.shape[1])
                    lstm_pred  = (lstm_model.predict(X_te_seq, verbose=0).flatten() > 0.5).astype(int)
                    lstm_acc   = round(accuracy_score(y_te_seq, lstm_pred)*100, 1)
                    models_info['lstm'] = {'model': lstm_model, 'acc': lstm_acc, 'weight': 1.3}
            except Exception:
                lstm_model = None

        # ── Ensemble weighted probability ──
        latest_scaled = scaler.transform(X[-1:])
        total_weight  = 0.0
        prob_up_sum   = 0.0

        model_details = {}
        for name, info in models_info.items():
            if name == 'lstm':
                continue
            m   = info['model']
            w   = info['weight']
            prob = m.predict_proba(latest_scaled)[0][1]
            prob_up_sum  += prob * w
            total_weight += w
            model_details[name] = {'prob': round(prob*100, 1), 'acc': info['acc']}

        # LSTM contribution
        if lstm_model is not None and HAS_LSTM:
            SEQ = 10
            if len(X) >= SEQ:
                seq_input = scaler.transform(X[-SEQ:]).reshape(1, SEQ, X.shape[1])
                lstm_prob = float(lstm_model.predict(seq_input, verbose=0)[0][0])
                w = models_info['lstm']['weight']
                prob_up_sum  += lstm_prob * w
                total_weight += w
                model_details['lstm'] = {'prob': round(lstm_prob*100, 1), 'acc': lstm_acc}

        ensemble_prob = (prob_up_sum / total_weight) if total_weight else 0.5

        # ── Cross-val accuracy (RF as representative) ──
        cv_scores = cross_val_score(rf, X_tr_s, y_tr, cv=3, scoring='accuracy')
        cv_acc    = round(cv_scores.mean()*100, 1)

        return {
            'ensemble_prob': round(ensemble_prob, 4),
            'direction':     'UP' if ensemble_prob >= 0.50 else 'DOWN',
            'confidence':    round(abs(ensemble_prob - 0.5) * 200, 1),  # 0–100
            'model_details': model_details,
            'cv_accuracy':   cv_acc,
            'scaler':        scaler,
            'latest_X':      X[-1],
            'models':        models_info,
        }

    # ── Compute all indicators for display ──────────────────────────────────
    def compute_indicators(self):
        hist    = self._hist
        closes  = hist['Close'].tolist()
        highs   = hist['High'].tolist()
        lows    = hist['Low'].tolist()
        volumes = hist['Volume'].tolist()

        rsi14    = Indicators.rsi(closes, 14)
        rsi7     = Indicators.rsi(closes, 7)
        macd_l, sig_l, hist_l = Indicators.macd(closes)
        bb_u, bb_m, bb_l = Indicators.bollinger(closes, 20)
        ema20    = Indicators.ema(closes, 20)
        ema50    = Indicators.ema(closes, 50)
        sma5     = Indicators.sma(closes, 5)
        sma20    = Indicators.sma(closes, 20)
        stoch_k, stoch_d = Indicators.stochastic(closes, highs, lows)
        will_r   = Indicators.williams_r(closes, highs, lows)
        atr_v    = Indicators.atr(closes, highs, lows)
        adx_v    = Indicators.adx(closes, highs, lows)
        support, resistance = Indicators.support_resistance(closes)

        def last(arr):
            for v in reversed(arr):
                if v == v:
                    return v
            return float('nan')

        return {
            'rsi14':      round(last(rsi14), 2),
            'rsi7':       round(last(rsi7), 2)  if last(rsi7)==last(rsi7) else None,
            'macd':       round(last(macd_l), 4),
            'macd_sig':   round(last(sig_l), 4),
            'macd_hist':  round(last(hist_l), 4),
            'bb_upper':   round(last(bb_u), 2),
            'bb_mid':     round(last(bb_m), 2),
            'bb_lower':   round(last(bb_l), 2),
            'bb_pct':     round((closes[-1]-last(bb_l))/(last(bb_u)-last(bb_l)), 3)
                          if last(bb_u)!=last(bb_l) and last(bb_u)==last(bb_u) else None,
            'ema20':      round(last(ema20), 2),
            'ema50':      round(last(ema50), 2),
            'sma5':       round(last(sma5), 2),
            'sma20':      round(last(sma20), 2),
            'stoch_k':    round(last(stoch_k), 2) if last(stoch_k)==last(stoch_k) else None,
            'stoch_d':    round(last(stoch_d), 2) if last(stoch_d)==last(stoch_d) else None,
            'williams_r': round(last(will_r), 2)  if last(will_r)==last(will_r)  else None,
            'atr':        round(last(atr_v), 2),
            'atr_pct':    round(last(atr_v)/closes[-1]*100, 2) if closes[-1] else None,
            'adx':        round(last(adx_v), 2)   if last(adx_v)==last(adx_v)   else None,
            'support':    support,
            'resistance': resistance,
        }

    # ── Multi-factor scoring (0-100) ─────────────────────────────────────────
    def multi_factor_score(self, ensemble_prob, indicators, info):
        score     = 0
        max_score = 0
        breakdown = {}

        # 1. ML Ensemble (30 pts)
        ml_score = round(ensemble_prob * 30)
        score   += ml_score; max_score += 30
        breakdown['ML Ensemble'] = {'score': ml_score, 'max': 30}

        # 2. Technical momentum (25 pts)
        tech = 0
        rsi  = indicators.get('rsi14', 50) or 50
        if 40 <= rsi <= 60:   tech += 5  # healthy zone
        elif rsi < 35:        tech += 4  # oversold (opportunity)
        elif rsi > 70:        tech -= 2  # overbought
        macd_h = indicators.get('macd_hist', 0) or 0
        if macd_h > 0:        tech += 5
        stk = indicators.get('stoch_k', 50) or 50
        if stk < 30:          tech += 4
        elif stk > 80:        tech -= 2
        bb_p = indicators.get('bb_pct', 0.5) or 0.5
        if bb_p < 0.25:       tech += 4  # near lower band
        elif bb_p > 0.85:     tech -= 2
        adx_v2 = indicators.get('adx', 20) or 20
        if adx_v2 > 25:       tech += 4  # strong trend
        wr = indicators.get('williams_r', -50) or -50
        if wr < -80:          tech += 3  # oversold
        tech = max(0, min(tech, 25))
        score += tech; max_score += 25
        breakdown['Technical Indicators'] = {'score': tech, 'max': 25}

        # 3. Trend alignment (20 pts)
        trend = 0
        ema20 = indicators.get('ema20') or 0
        ema50 = indicators.get('ema50') or 0
        sma20 = indicators.get('sma20') or 0
        close = self._hist['Close'].iloc[-1]
        if ema20 and ema50 and ema20 > ema50:  trend += 8
        if sma20 and close > sma20:            trend += 6
        if ema20 and close > ema20:            trend += 6
        trend = min(trend, 20)
        score += trend; max_score += 20
        breakdown['Trend Alignment'] = {'score': trend, 'max': 20}

        # 4. Fundamentals (15 pts)
        fund = 0
        pe   = info.get('pe_ratio')
        pb   = info.get('pb_ratio')
        roe  = info.get('roe')
        de   = info.get('debt_to_equity')
        if pe  and 0 < pe < 25:   fund += 4
        elif pe and pe < 40:      fund += 2
        if pb  and 0 < pb < 3:    fund += 3
        if roe and roe > 0.15:    fund += 4
        elif roe and roe > 0.08:  fund += 2
        if de is not None and de < 1: fund += 4
        elif de is not None and de < 2: fund += 2
        fund = min(fund, 15)
        score += fund; max_score += 15
        breakdown['Fundamentals'] = {'score': fund, 'max': 15}

        # 5. Volume & volatility (10 pts)
        volvol = 0
        atr_p  = indicators.get('atr_pct', 2) or 2
        if atr_p < 1.5:    volvol += 5  # low volatility
        elif atr_p < 3:    volvol += 3
        elif atr_p > 5:    volvol += 0  # high vol = risky
        beta   = info.get('beta')
        if beta and 0.5 < beta < 1.5: volvol += 5
        volvol = min(volvol, 10)
        score += volvol; max_score += 10
        breakdown['Volatility & Risk'] = {'score': volvol, 'max': 10}

        pct = round((score / max_score) * 100) if max_score else 50
        return pct, score, max_score, breakdown

    # ── Decision with entry/target/stop ─────────────────────────────────────
    def generate_signal(self, ensemble_prob, indicators, composite_score, current_price, info):
        rsi   = indicators.get('rsi14', 50) or 50
        ema20 = indicators.get('ema20') or current_price
        ema50 = indicators.get('ema50') or current_price
        atr   = indicators.get('atr', current_price * 0.02) or current_price * 0.02
        macd_h = indicators.get('macd_hist', 0) or 0
        adx_v  = indicators.get('adx', 20) or 20

        # Determine final verdict based on composite score
        if composite_score >= 75:
            final = 'STRONG BUY'
        elif composite_score >= 60:
            final = 'BUY'
        elif composite_score <= 25:
            final = 'STRONG SELL'
        elif composite_score <= 40:
            final = 'SELL'
        else:
            final = 'HOLD'

        # Entry, target, stop-loss
        entry  = current_price
        sl_atr = round(current_price - 2.0 * atr, 2)
        sl_pct = round(current_price * 0.94, 2)     # 6% hard stop
        stop   = max(sl_atr, sl_pct)

        r2r    = 2.0  # risk-to-reward
        risk   = current_price - stop
        target_1 = round(current_price + risk * r2r,       2)
        target_2 = round(current_price + risk * r2r * 1.5, 2)

        # Risk-reward ratio
        rr_ratio = round(risk_reward := (target_1 - entry) / (entry - stop), 2) \
                   if (entry - stop) > 0 else 0

        # Risk classification
        atr_pct = indicators.get('atr_pct', 2) or 2
        if atr_pct > 4 or rsi > 75:   risk_level = 'HIGH'
        elif atr_pct > 2 or rsi > 65: risk_level = 'MEDIUM'
        else:                          risk_level = 'LOW'

        # Reason string
        trend_str = 'bullish (EMA20>EMA50)' if ema20 > ema50 else 'bearish (EMA20<EMA50)'
        rsi_str   = f'RSI {rsi:.1f} — {"oversold" if rsi<35 else "overbought" if rsi>70 else "neutral"}'
        macd_str  = 'MACD bullish' if macd_h > 0 else 'MACD bearish'
        adx_str   = f'ADX {adx_v:.0f} — {"strong trend" if adx_v>25 else "weak trend"}'
        prob_str  = f'Ensemble probability: {ensemble_prob*100:.1f}% UP'
        reason = (f"{prob_str}. Trend {trend_str}. {rsi_str}. {macd_str}. {adx_str}. "
                  f"Composite score {composite_score}/100.")

        return {
            'final':     final,
            'entry':     round(entry, 2),
            'target_1':  target_1,
            'target_2':  target_2,
            'stop_loss': round(stop, 2),
            'rr_ratio':  rr_ratio,
            'risk':      risk_level,
            'reason':    reason,
        }

    # ── Volatility / risk ────────────────────────────────────────────────────
    @staticmethod
    def compute_risk(closes):
        if len(closes) < 2: return 'MEDIUM'
        returns  = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
        mean_r   = sum(returns)/len(returns)
        variance = sum((r-mean_r)**2 for r in returns)/len(returns)
        vol = variance**0.5
        if vol > 0.03:    return 'HIGH'
        elif vol > 0.015: return 'MEDIUM'
        else:             return 'LOW'

    # ── Dip label ────────────────────────────────────────────────────────────
    @staticmethod
    def dip_label(prices, long_ma):
        below   = sum(1 for p in prices[-3:] if p < long_ma)
        dip_pct = round(((long_ma-prices[-1])/long_ma)*100, 2)
        if   below >= 2 and dip_pct >= 2: dip = 'STRONG DIP'
        elif below >= 1 and dip_pct >= 1: dip = 'MILD DIP'
        elif prices[-1] > long_ma * 1.05: dip = 'OVERBOUGHT'
        else:                             dip = 'NEAR FAIR VALUE'
        return dip, abs(dip_pct)

    # ── Linear regression prediction ─────────────────────────────────────────
    @staticmethod
    def lr_predict(prices):
        x  = list(range(len(prices)))
        n  = len(x)
        mx = sum(x)/n; my = sum(prices)/n
        num = sum((x[i]-mx)*(prices[i]-my) for i in range(n))
        den = sum((x[i]-mx)**2 for i in range(n))
        slope  = num/den if den else 0
        next_p = slope*n + (my - slope*mx)
        pct    = round(((next_p-prices[-1])/prices[-1])*100, 2)
        return round(next_p, 2), 'UP' if next_p > prices[-1] else 'DOWN', pct


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data      = request.json or {}
    symbol    = data.get('stock', '').strip()
    user_type = data.get('user_type', 'trader').lower()
    if not symbol:
        return jsonify({'error': 'Stock symbol required'})
    try:
        eng = EnsembleEngine(symbol)
        prices6, current, info = eng.fetch()

        # Ensemble training
        result   = eng.train_ensemble()
        ensemble_prob = result['ensemble_prob']
        direction     = result['direction']
        confidence    = result['confidence']
        model_details = result['model_details']
        cv_acc        = result['cv_accuracy']

        # Indicators
        inds = eng.compute_indicators()

        # Composite score
        c_score, raw_s, raw_max, breakdown = eng.multi_factor_score(ensemble_prob, inds, info)

        # Signal
        sig = eng.generate_signal(ensemble_prob, inds, c_score, current, info)

        # Change
        closes_all = eng._hist['Close'].dropna().tolist()
        change_pct = round((closes_all[-1]-closes_all[-2])/closes_all[-2]*100, 2) \
                     if len(closes_all) >= 2 else 0.0

        dip, dip_pct = EnsembleEngine.dip_label(prices6, inds['sma20'] or prices6[-1])

        # History entry
        entry = {
            'id': len(history_log)+1, 'stock': symbol.upper(),
            'name': info.get('name', symbol), 'price': current,
            'predicted': round(closes_all[-1], 2), 'direction': direction,
            'change_pct': change_pct, 'final': sig['final'],
            'trend':      'BULLISH' if inds['ema20'] and inds['ema50'] and inds['ema20']>inds['ema50'] else 'BEARISH',
            'momentum':   'UPWARD'  if direction == 'UP' else 'DOWNWARD',
            'composite':  c_score,
            'timestamp':  datetime.now().strftime('%d %b %Y, %H:%M'),
        }
        history_log.append(entry)
        if len(history_log) > 50: history_log.pop(0)
        save_history()

        # Chart data (recent 60 days)
        chart_closes = [round(v, 2) for v in closes_all[-60:]]
        hist_dates   = [str(d.date()) for d in eng._hist.index[-60:]]

        return jsonify({
            'stock':    symbol.upper(),
            'name':     info.get('name', symbol),
            'price':    current,
            'prices':   prices6,
            'chart_prices': chart_closes,
            'chart_dates':  hist_dates,
            'prediction':   round(closes_all[-1], 2),
            'direction':    direction,
            'change_pct':   change_pct,
            'ensemble_prob': round(ensemble_prob*100, 1),
            'confidence':   round(confidence, 1),
            'cv_accuracy':  cv_acc,
            'model_details': model_details,
            'composite_score': c_score,
            'score_breakdown': breakdown,
            # Indicators
            'rsi14':      inds['rsi14'],
            'rsi7':       inds['rsi7'],
            'macd':       inds['macd'],
            'macd_sig':   inds['macd_sig'],
            'macd_hist':  inds['macd_hist'],
            'bb_upper':   inds['bb_upper'],
            'bb_mid':     inds['bb_mid'],
            'bb_lower':   inds['bb_lower'],
            'bb_pct':     inds['bb_pct'],
            'ema20':      inds['ema20'],
            'ema50':      inds['ema50'],
            'sma5':       inds['sma5'],
            'sma20':      inds['sma20'],
            'stoch_k':    inds['stoch_k'],
            'stoch_d':    inds['stoch_d'],
            'williams_r': inds['williams_r'],
            'atr':        inds['atr'],
            'atr_pct':    inds['atr_pct'],
            'adx':        inds['adx'],
            'support':    inds['support'],
            'resistance': inds['resistance'],
            # Signal
            'final':      sig['final'],
            'entry':      sig['entry'],
            'target_1':   sig['target_1'],
            'target_2':   sig['target_2'],
            'stop_loss':  sig['stop_loss'],
            'rr_ratio':   sig['rr_ratio'],
            'risk':       sig['risk'],
            'reason':     sig['reason'],
            # Fundamentals
            'sector':         info.get('sector', 'N/A'),
            'industry':       info.get('industry', 'N/A'),
            '52w_high':       info.get('52w_high'),
            '52w_low':        info.get('52w_low'),
            'pe_ratio':       info.get('pe_ratio'),
            'pb_ratio':       info.get('pb_ratio'),
            'market_cap':     info.get('market_cap'),
            'roe':            info.get('roe'),
            'debt_to_equity': info.get('debt_to_equity'),
            'eps':            info.get('eps'),
            'dividend_yield': info.get('dividend_yield'),
            'revenue_growth': info.get('revenue_growth'),
            'profit_margin':  info.get('profit_margin'),
            'current_ratio':  info.get('current_ratio'),
            'beta':           info.get('beta'),
            'analyst_target': info.get('analyst_target'),
            'recommendation': info.get('recommendation'),
            'dip':            dip,
            'dip_pct':        dip_pct,
            'short_ma':       inds['sma5'],
            'long_ma':        inds['sma20'],
            'trend':          'BULLISH' if inds['ema20'] and inds['ema50'] and inds['ema20']>inds['ema50'] else 'BEARISH',
            'momentum':       'UPWARD'  if direction == 'UP' else 'DOWNWARD',
            'user_type':      user_type,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/backtest', methods=['POST'])
def backtest():
    symbol = request.json.get('stock', '').strip()
    period = request.json.get('period', '1y')
    if not symbol:
        return jsonify({'error': 'Stock symbol required'})
    try:
        stock_obj = yf.Ticker(symbol.upper())
        hist      = stock_obj.history(period=period)
        if hist.empty or len(hist) < 60:
            return jsonify({'error': 'Not enough data — try 1 Year or 2 Years.'})

        closes  = hist['Close'].dropna().tolist()
        highs   = hist['High'].tolist()
        lows    = hist['Low'].tolist()
        dates   = [str(d.date()) for d in hist.index]

        rsi_vals            = Indicators.rsi(closes, 14)
        macd_line, sig_line, hist_line = Indicators.macd(closes)
        atr_vals            = Indicators.atr(closes, highs, lows)
        ema20               = Indicators.ema(closes, 20)
        ema50               = Indicators.ema(closes, 50)
        bb_u, _, bb_l       = Indicators.bollinger(closes)
        stoch_k, stoch_d    = Indicators.stochastic(closes, highs, lows)
        adx_vals            = Indicators.adx(closes, highs, lows)

        trades = []
        wins = losses = 0
        total_return = 0.0
        in_trade = False
        buy_price = buy_date = None
        trail_stop = highest_price = None
        hold_days  = 0
        MAX_HOLD   = 30
        WARMUP     = 55

        for i in range(WARMUP, len(closes)):
            price = closes[i]
            date  = dates[i]
            rsi   = rsi_vals[i]
            atr   = atr_vals[i]
            ml    = macd_line[i]
            sl    = sig_line[i]
            e20   = ema20[i]
            e50   = ema50[i]
            bbu   = bb_u[i]
            bbl   = bb_l[i]
            sk    = stoch_k[i]
            adx   = adx_vals[i]

            if any(v != v for v in [rsi, atr, ml, sl, e20, e50]):
                continue

            prev_ml = macd_line[i-1] if i > 0 and macd_line[i-1]==macd_line[i-1] else ml
            prev_sl = sig_line[i-1]  if i > 0 and sig_line[i-1]==sig_line[i-1]   else sl

            cond1 = e20 > e50
            cond2 = 35 < rsi < 65
            cond3 = ml > sl
            cond4 = price > e20
            cond5 = (sk == sk and sk < 70)
            cond6 = (bbl == bbl and price > bbl)
            score = sum([cond1, cond2, cond3, cond4, cond5, cond6])

            macd_cross_up   = ml > sl and prev_ml <= prev_sl
            macd_cross_down = ml < sl and prev_ml >= prev_sl

            if not in_trade:
                if score >= 4 and cond1:
                    in_trade      = True
                    buy_price     = price
                    buy_date      = date
                    hold_days     = 0
                    highest_price = price
                    trail_stop    = price - 2.0 * atr
            else:
                hold_days += 1
                if price > highest_price:
                    highest_price = price
                new_stop = highest_price - 2.0 * atr
                if new_stop > trail_stop:
                    trail_stop = new_stop

                pnl_pct = (price - buy_price) / buy_price

                hit_trail   = price <= trail_stop
                hit_profit  = pnl_pct >= 0.08
                hit_ob_rsi  = rsi > 75
                hit_macd_dn = macd_cross_down
                hit_ema_dn  = price < e20 * 0.97
                hit_timeout = hold_days >= MAX_HOLD

                if any([hit_trail, hit_profit, hit_ob_rsi, hit_macd_dn, hit_ema_dn, hit_timeout]):
                    in_trade  = False
                    pnl = round(pnl_pct * 100, 2)
                    won = pnl > 0
                    wins   += 1 if won else 0
                    losses += 0 if won else 1
                    total_return += pnl
                    if hit_trail:    reason = 'TRAIL-STOP'
                    elif hit_profit: reason = 'TAKE-PROFIT'
                    elif hit_ob_rsi: reason = 'RSI-EXIT'
                    elif hit_macd_dn: reason = 'MACD-EXIT'
                    elif hit_ema_dn:  reason = 'EMA-EXIT'
                    else:             reason = 'TIMEOUT'
                    trades.append({
                        'buy_date': buy_date, 'sell_date': date,
                        'buy_price': round(buy_price, 2), 'sell_price': round(price, 2),
                        'pnl': pnl, 'result': 'WIN' if won else 'LOSS',
                        'exit_reason': reason, 'hold_days': hold_days,
                    })

        if in_trade:
            price = closes[-1]
            pnl   = round(((price-buy_price)/buy_price)*100, 2)
            won   = pnl > 0
            wins  += 1 if won else 0; losses += 0 if won else 1
            total_return += pnl
            trades.append({
                'buy_date': buy_date, 'sell_date': dates[-1]+' (open)',
                'buy_price': round(buy_price, 2), 'sell_price': round(price, 2),
                'pnl': pnl, 'result': 'WIN' if won else 'LOSS',
                'exit_reason': 'OPEN', 'hold_days': hold_days,
            })

        total_trades = wins + losses
        win_rate     = round(wins / total_trades * 100, 1) if total_trades else 0
        avg_return   = round(total_return / total_trades, 2) if total_trades else 0
        max_dd       = 0.0
        peak         = closes[0]
        for c in closes:
            if c > peak: peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd: max_dd = dd

        # Sharpe (annualised, simplified)
        if len(closes) >= 2:
            rets     = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
            mean_r   = sum(rets)/len(rets)
            std_r    = (sum((r-mean_r)**2 for r in rets)/len(rets))**0.5
            sharpe   = round((mean_r / std_r) * (252**0.5), 2) if std_r else 0
        else:
            sharpe = 0

        step = max(1, len(closes)//100)
        return jsonify({
            'stock': symbol.upper(), 'period': period,
            'total_trades': total_trades, 'wins': wins, 'losses': losses,
            'win_rate': win_rate, 'total_return': round(total_return, 2),
            'avg_return': avg_return, 'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': sharpe,
            'trades': trades[-30:],
            'chart_prices': [round(closes[i], 2) for i in range(0, len(closes), step)],
            'chart_dates':  [dates[i]            for i in range(0, len(dates),  step)],
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/compare', methods=['POST'])
def compare():
    symbols = request.json.get('stocks', [])
    symbols = [s.strip().upper() for s in symbols if s.strip()][:3]
    if len(symbols) < 2:
        return jsonify({'error': 'Enter at least 2 stock symbols'})
    results = []
    for sym in symbols:
        try:
            eng = EnsembleEngine(sym)
            prices6, current, info = eng.fetch('1y')
            result   = eng.train_ensemble()
            ep       = result['ensemble_prob']
            direction = result['direction']
            inds     = eng.compute_indicators()
            c_score, *_ = eng.multi_factor_score(ep, inds, info)
            sig      = eng.generate_signal(ep, inds, c_score, current, info)
            closes_all = eng._hist['Close'].dropna().tolist()
            change_pct = round((closes_all[-1]-closes_all[-2])/closes_all[-2]*100, 2) \
                         if len(closes_all) >= 2 else 0.0
            dip, dip_pct = EnsembleEngine.dip_label(prices6, inds['sma20'] or prices6[-1])
            ema20 = inds['ema20'] or current
            ema50 = inds['ema50'] or current
            results.append({
                'stock': sym, 'name': info.get('name', sym), 'price': current,
                'prediction': round(closes_all[-1], 2), 'direction': direction,
                'change_pct': change_pct,
                'ensemble_prob': round(ep*100, 1),
                'composite_score': c_score,
                'final': sig['final'],
                'trend':    'BULLISH' if ema20 > ema50 else 'BEARISH',
                'momentum': 'UPWARD'  if direction == 'UP' else 'DOWNWARD',
                'dip': dip, 'dip_pct': dip_pct,
                'rsi': inds['rsi14'], 'adx': inds['adx'],
                'target_1': sig['target_1'], 'stop_loss': sig['stop_loss'],
                'rr_ratio': sig['rr_ratio'],
                'pe_ratio': info.get('pe_ratio'), 'pb_ratio': info.get('pb_ratio'),
                'market_cap': info.get('market_cap'), 'roe': info.get('roe'),
                'sector': info.get('sector', 'N/A'),
                '52w_high': info.get('52w_high'), '52w_low': info.get('52w_low'),
                'prices': [round(p,2) for p in closes_all[-30:]],
                'score': c_score, 'error': None,
            })
        except Exception as e:
            results.append({'stock': sym, 'error': str(e)})

    valid = sorted([r for r in results if not r.get('error')],
                   key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(valid): r['rank'] = i+1
    return jsonify(results)


@app.route('/add_portfolio', methods=['POST'])
def add_portfolio():
    stock = request.json.get('stock', '').strip().upper()
    if stock and stock not in portfolio: portfolio.append(stock)
    return jsonify({'portfolio': portfolio})

@app.route('/remove_portfolio', methods=['POST'])
def remove_portfolio():
    stock = request.json.get('stock', '').strip().upper()
    if stock in portfolio: portfolio.remove(stock)
    return jsonify({'portfolio': portfolio})

@app.route('/portfolio')
def get_portfolio(): return jsonify(portfolio)


@app.route('/top')
def top_stocks():
    sample = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
              'HINDUNILVR.NS','SBIN.NS','BAJFINANCE.NS','KOTAKBANK.NS','BHARTIARTL.NS',
              'ITC.NS','LT.NS','AXISBANK.NS','MARUTI.NS','WIPRO.NS','HCLTECH.NS',
              'SUNPHARMA.NS','NTPC.NS','POWERGRID.NS','TITAN.NS']
    heap = []
    for s in sample:
        try:
            stock = yf.Ticker(s)
            hist  = stock.history(period='1mo')
            if hist.empty: continue
            closes = hist['Close'].dropna().tolist()
            if len(closes) < 2: continue
            current = round(closes[-1], 2)
            score   = ((closes[-1]-closes[0])/closes[0])*100
            heapq.heappush(heap, (-score, s, current, round(score, 2)))
        except Exception:
            pass
    top = []
    while heap and len(top) < 5:
        _, sym, price, pct = heapq.heappop(heap)
        top.append({'symbol': sym, 'price': price, 'growth': pct})
    return jsonify(top)


@app.route('/history')
def get_history(): return jsonify(list(reversed(history_log)))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    history_log.clear(); save_history()
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    app.run(debug=True)