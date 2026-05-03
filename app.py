from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
import heapq
import json
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='static')

portfolio   = []
history_log = []
HISTORY_FILE = 'history.json'

if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r') as f:
            history_log = json.load(f)
    except:
        history_log = []

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_log, f)


class Engine:
    def __init__(self, symbol):
        self.symbol = symbol.upper()

    def fetch(self):
        stock = yf.Ticker(self.symbol)
        hist  = stock.history(period="3mo")
        if hist.empty:
            raise Exception("Invalid stock symbol or no data found.")
        closes_full = hist['Close'].dropna()
        prices = [round(p, 2) for p in closes_full.tolist()[-6:]]
        if len(prices) < 2:
            raise Exception("Not enough price data.")
        try:
            current = round(stock.fast_info['last_price'], 2)
        except:
            current = prices[-1]
        self._hist = hist
        info = {}
        try:
            raw  = stock.info
            info = {
                'pe_ratio':       raw.get('trailingPE',       None),
                'pb_ratio':       raw.get('priceToBook',      None),
                'market_cap':     raw.get('marketCap',        None),
                'roe':            raw.get('returnOnEquity',   None),
                'debt_to_equity': raw.get('debtToEquity',     None),
                'name':           raw.get('longName',         self.symbol),
                'sector':         raw.get('sector',           'N/A'),
                '52w_high':       raw.get('fiftyTwoWeekHigh', None),
                '52w_low':        raw.get('fiftyTwoWeekLow',  None),
            }
        except:
            info = {'name': self.symbol}
        return prices, current, info

    def fetch_extended(self, period="6mo"):
        stock = yf.Ticker(self.symbol)
        hist  = stock.history(period=period)
        if hist.empty:
            raise Exception("Invalid stock symbol or no data found.")
        closes = hist['Close'].dropna().tolist()
        dates  = [str(d.date()) for d in hist.index]
        return closes, dates

    def compute_rsi(self, closes, period=14):
        rsi_values = [float('nan')] * len(closes)
        if len(closes) < period + 1:
            return rsi_values
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(closes)):
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = round(100 - (100 / (1 + rs)), 2)
            if i < len(closes) - 1:
                d = closes[i] - closes[i-1]
                avg_gain = (avg_gain*(period-1) + max(d, 0))  / period
                avg_loss = (avg_loss*(period-1) + max(-d, 0)) / period
        return rsi_values

    def build_features(self, hist):
        closes  = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        rsi_all = self.compute_rsi(closes, period=14)
        records = []
        for i in range(20, len(closes) - 1):
            rsi = rsi_all[i]
            if rsi != rsi:
                continue
            ret     = (closes[i]-closes[i-1])/closes[i-1] if closes[i-1] else 0
            ma5     = sum(closes[i-4:i+1]) / 5
            ma20    = sum(closes[i-19:i+1]) / 20
            vol_chg = (volumes[i]-volumes[i-1])/volumes[i-1] if volumes[i-1] else 0
            label   = 1 if closes[i+1] > closes[i] else 0
            records.append({'Returns': ret, 'MA_short': ma5, 'MA_long': ma20,
                            'Volume_Change': vol_chg, 'RSI': rsi, 'Label': label})
        return records

    def train_model(self, hist):
        records = self.build_features(hist)
        if len(records) < 10:
            raise Exception("Not enough data to train ML model (need 30+ trading days).")
        FEATURES = ['Returns', 'MA_short', 'MA_long', 'Volume_Change', 'RSI']
        X = [[r[f] for f in FEATURES] for r in records]
        y = [r['Label'] for r in records]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        correct  = sum(p==a for p,a in zip(model.predict(X_test), y_test))
        accuracy = round(correct/len(y_test)*100, 1) if y_test else 0
        latest          = records[-1]
        latest_features = [latest[f] for f in FEATURES]
        return model, accuracy, latest_features, latest

    def ml_predict(self, model, latest_features):
        pred       = model.predict([latest_features])[0]
        proba      = model.predict_proba([latest_features])[0]
        direction  = "UP" if pred == 1 else "DOWN"
        confidence = round(max(proba)*100, 1)
        return direction, confidence

    def decision(self, ml_direction, rsi, ma_short, ma_long, user_type="trader"):
        trend    = "BULLISH" if ma_short > ma_long else "BEARISH"
        momentum = "UPWARD"  if ml_direction == "UP" else "DOWNWARD"

        if trend == "BULLISH" and momentum == "UPWARD":
            short = "BUY"
        elif trend == "BEARISH" and momentum == "DOWNWARD":
            short = "SELL"
        else:
            short = "HOLD"

        if rsi < 30:
            long_v = "BUY"
        elif rsi > 70:
            long_v = "SELL"
        else:
            long_v = "BUY" if ma_short > ma_long else "SELL"

        if user_type == "investor":
            if rsi < 30:
                final  = "BUY"
                reason = (f"RSI {rsi:.1f} — oversold. "
                          "Historically cheap, good long-term entry.")
            elif rsi > 70:
                final  = "SELL"
                reason = (f"RSI {rsi:.1f} — overbought. "
                          "Price likely to correct, book profits.")
            else:
                final  = "HOLD"
                reason = (f"RSI {rsi:.1f} — neutral. "
                          "No strong signal for long-term investor.")
        else:
            if short == "BUY" and long_v == "BUY":
                final  = "STRONG BUY"
                reason = (f"ML predicts UP, trend is {trend}, RSI={rsi:.1f}. "
                          "Both short-term and long-term signals are bullish — strong entry.")
            elif short == "BUY" and long_v == "SELL":
                final  = "BUY"
                reason = (f"ML predicts UP, trend is {trend}. Short-term bullish. "
                          f"RSI={rsi:.1f} — cautious buy, monitor long-term.")
            elif short == "SELL" and long_v == "SELL":
                final  = "STRONG SELL"
                reason = (f"ML predicts DOWN, trend is {trend}, RSI={rsi:.1f}. "
                          "Both signals bearish — exit or avoid position.")
            elif short == "SELL" and long_v == "BUY":
                final  = "HOLD"
                reason = (f"Short-term pullback (ML=DOWN) but long-term still healthy "
                          f"(RSI={rsi:.1f}, trend={trend}). Wait for better entry.")
            else:
                final  = "HOLD"
                reason = (f"Mixed signals — ML={ml_direction}, trend={trend}, "
                          f"RSI={rsi:.1f}. No clear entry or exit signal.")

        return {"trend": trend, "momentum": momentum, "short": short,
                "long": long_v, "final": final, "reason": reason}

    def predict(self, prices):
        """Legacy linear regression — used by /compare and /top."""
        x  = list(range(len(prices)))
        n  = len(x)
        mx = sum(x)/n;  my = sum(prices)/n
        num = sum((x[i]-mx)*(prices[i]-my) for i in range(n))
        den = sum((x[i]-mx)**2 for i in range(n))
        slope  = num/den if den else 0
        next_p = slope*n + (my - slope*mx)
        direction  = "UP" if next_p > prices[-1] else "DOWN"
        change_pct = round(((next_p-prices[-1])/prices[-1])*100, 2)
        return round(next_p, 2), direction, change_pct

    def dip_label(self, prices, long_ma):
        below   = sum(1 for p in prices[-3:] if p < long_ma)
        dip_pct = round(((long_ma - prices[-1]) / long_ma) * 100, 2)
        if   below >= 2 and dip_pct >= 2: dip = "STRONG DIP"
        elif below >= 1 and dip_pct >= 1: dip = "MILD DIP"
        elif prices[-1] > long_ma * 1.05: dip = "OVERBOUGHT"
        else:                             dip = "NEAR FAIR VALUE"
        return dip, abs(dip_pct)

    def compute_risk(self, closes):
        """Volatility-based risk: std dev of daily returns."""
        if len(closes) < 2:
            return "MEDIUM"
        returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
        mean_r  = sum(returns) / len(returns)
        variance = sum((r - mean_r)**2 for r in returns) / len(returns)
        vol = variance ** 0.5
        if vol > 0.03:
            return "HIGH"
        elif vol > 0.015:
            return "MEDIUM"
        else:
            return "LOW"


@app.route('/')
def home():
    return send_from_directory('static', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data      = request.json or {}
    symbol    = data.get('stock', '').strip()
    user_type = data.get('user_type', 'trader').lower()
    if not symbol:
        return jsonify({"error": "Stock symbol required"})
    try:
        eng = Engine(symbol)
        prices, current, info = eng.fetch()

        model, accuracy, latest_features, latest_row = eng.train_model(eng._hist)
        closes_all = eng._hist['Close'].dropna().tolist()
        rsi_all    = eng.compute_rsi(closes_all, period=14)
        latest_rsi = rsi_all[-1] if rsi_all[-1] == rsi_all[-1] else latest_row['RSI']

        ml_direction, confidence = eng.ml_predict(model, latest_features)
        dec = eng.decision(ml_direction, latest_rsi,
                           latest_row['MA_short'], latest_row['MA_long'], user_type)

        # ✅ FIXED: now using the same MA values as decision engine
        short_ma_display = round(latest_row['MA_short'], 2)
        long_ma_display  = round(latest_row['MA_long'], 2)

        dip, dip_pct     = eng.dip_label(prices, long_ma_display)
        change_pct = round((closes_all[-1]-closes_all[-2])/closes_all[-2]*100, 2) \
                     if len(closes_all) >= 2 else 0.0

        risk = eng.compute_risk(closes_all)

        entry = {
            "id": len(history_log)+1, "stock": symbol.upper(),
            "name": info.get('name', symbol), "price": current,
            "predicted": round(closes_all[-1],2), "direction": ml_direction,
            "change_pct": change_pct, "final": dec['final'],
            "trend": dec['trend'], "momentum": dec['momentum'],
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        history_log.append(entry)
        if len(history_log) > 50: history_log.pop(0)
        save_history()

        return jsonify({
            "stock": symbol.upper(), "name": info.get('name', symbol),
            "price": current, "prices": [round(p,2) for p in prices],
            "prediction": round(closes_all[-1],2), "direction": ml_direction,
            "change_pct": change_pct,
            "sector": info.get('sector','N/A'), "52w_high": info.get('52w_high'),
            "52w_low": info.get('52w_low'), "pe_ratio": info.get('pe_ratio'),
            "pb_ratio": info.get('pb_ratio'), "market_cap": info.get('market_cap'),
            "roe": info.get('roe'),
            "short_ma": short_ma_display, "long_ma": long_ma_display,
            "dip": dip, "dip_pct": dip_pct,
            "trend":    dec['trend'],
            "momentum": dec['momentum'],
            "short":    dec['short'],
            "long":     dec['long'],
            "final":    dec['final'],
            "confidence": f"{confidence}%",
            "accuracy":   f"{accuracy}%",
            "risk":       risk,
            "rsi":        round(latest_rsi, 2),
            "reason":     dec['reason'],
            "user_type":  user_type,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/backtest', methods=['POST'])
def backtest():
    symbol = request.json.get('stock', '').strip()
    period = request.json.get('period', '6mo')
    if not symbol:
        return jsonify({"error": "Stock symbol required"})
    try:
        eng = Engine(symbol)
        closes, dates = eng.fetch_extended(period)
        if len(closes) < 8:
            return jsonify({"error": "Not enough historical data"})
        trades = []; wins = losses = 0; total_return = 0.0
        in_trade = False; buy_price = buy_date = 0
        for i in range(6, len(closes)):
            window   = closes[i-6:i]
            short_ma = sum(window[-3:])/3
            long_ma  = sum(window)/6
            trend    = "BULLISH" if short_ma > long_ma else "BEARISH"
            momentum = "UPWARD"  if window[-1] > window[-2] else "DOWNWARD"
            long_v   = "BUY" if window[-1] > long_ma else "SELL"
            if trend=="BULLISH" and momentum=="UPWARD" and long_v=="BUY":   signal="BUY"
            elif trend=="BEARISH" and momentum=="DOWNWARD" and long_v=="SELL": signal="SELL"
            else: signal="HOLD"
            price, date = closes[i], dates[i]
            if signal=="BUY" and not in_trade:
                in_trade=True; buy_price=price; buy_date=date
            elif signal=="SELL" and in_trade:
                in_trade=False
                pnl=round(((price-buy_price)/buy_price)*100,2); won=pnl>0
                wins+=1 if won else 0; losses+=0 if won else 1; total_return+=pnl
                trades.append({"buy_date":buy_date,"sell_date":date,
                                "buy_price":round(buy_price,2),"sell_price":round(price,2),
                                "pnl":pnl,"result":"WIN" if won else "LOSS"})
        if in_trade:
            price=closes[-1]; pnl=round(((price-buy_price)/buy_price)*100,2); won=pnl>0
            wins+=1 if won else 0; losses+=0 if won else 1; total_return+=pnl
            trades.append({"buy_date":buy_date,"sell_date":dates[-1]+" (open)",
                           "buy_price":round(buy_price,2),"sell_price":round(price,2),
                           "pnl":pnl,"result":"WIN" if won else "LOSS"})
        total_trades=wins+losses
        win_rate=round(wins/total_trades*100,1) if total_trades else 0
        avg_return=round(total_return/total_trades,2) if total_trades else 0
        step=max(1,len(closes)//60)
        return jsonify({
            "stock":symbol.upper(),"period":period,"total_trades":total_trades,
            "wins":wins,"losses":losses,"win_rate":win_rate,
            "total_return":round(total_return,2),"avg_return":avg_return,
            "trades":trades[-20:],
            "chart_prices":[round(closes[i],2) for i in range(0,len(closes),step)],
            "chart_dates" :[dates[i]           for i in range(0,len(dates), step)],
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/compare', methods=['POST'])
def compare():
    symbols = request.json.get('stocks', [])
    symbols = [s.strip().upper() for s in symbols if s.strip()][:3]
    if len(symbols) < 2:
        return jsonify({"error": "Enter at least 2 stock symbols"})
    results = []
    for sym in symbols:
        try:
            eng = Engine(sym)
            prices, current, info = eng.fetch()
            prediction, direction, change_pct = eng.predict(prices)
            short_ma = round(sum(prices[-3:])/3, 2)
            long_ma  = round(sum(prices)/len(prices), 2)
            trend    = "BULLISH" if short_ma > long_ma else "BEARISH"
            momentum = "UPWARD"  if prices[-1] > prices[-2] else "DOWNWARD"
            long_v   = "BUY" if prices[-1] > long_ma else "SELL"
            dip, dip_pct = eng.dip_label(prices, long_ma)
            if trend=="BULLISH" and momentum=="UPWARD":    short="BUY"
            elif trend=="BEARISH" and momentum=="DOWNWARD": short="SELL"
            else:                                           short="HOLD"
            if short=="BUY"  and long_v=="BUY":   final="STRONG BUY"
            elif short=="BUY"  and long_v=="SELL": final="BUY"
            elif short=="SELL" and long_v=="SELL": final="STRONG SELL"
            elif short=="SELL" and long_v=="BUY":  final="HOLD"
            else:                                  final="HOLD"
            score = 0
            if trend=="BULLISH":                       score+=2
            if momentum=="UPWARD":                     score+=2
            if direction=="UP":                        score+=1
            if dip in ("STRONG DIP","MILD DIP"):       score+=2
            results.append({
                "stock":sym,"name":info.get('name',sym),"price":current,
                "prediction":prediction,"direction":direction,"change_pct":change_pct,
                "trend":trend,"momentum":momentum,"dip":dip,"dip_pct":dip_pct,
                "short":short,"long":long_v,"final":final,
                "short_ma":short_ma,"long_ma":long_ma,
                "pe_ratio":info.get('pe_ratio'),"pb_ratio":info.get('pb_ratio'),
                "market_cap":info.get('market_cap'),"roe":info.get('roe'),
                "sector":info.get('sector','N/A'),
                "52w_high":info.get('52w_high'),"52w_low":info.get('52w_low'),
                "prices":[round(p,2) for p in prices],"score":score,"error":None,
            })
        except Exception as e:
            results.append({"stock":sym,"error":str(e)})
    valid=[r for r in results if not r.get('error')]
    valid.sort(key=lambda x:x['score'],reverse=True)
    for i,r in enumerate(valid): r['rank']=i+1
    return jsonify(results)


@app.route('/add_portfolio',    methods=['POST'])
def add_portfolio():
    stock=request.json.get('stock','').strip().upper()
    if stock and stock not in portfolio: portfolio.append(stock)
    return jsonify({"portfolio":portfolio})

@app.route('/remove_portfolio', methods=['POST'])
def remove_portfolio():
    stock=request.json.get('stock','').strip().upper()
    if stock in portfolio: portfolio.remove(stock)
    return jsonify({"portfolio":portfolio})

@app.route('/portfolio')
def get_portfolio(): return jsonify(portfolio)

@app.route('/top')
def top_stocks():
    sample=["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
            "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","KOTAKBANK.NS","BHARTIARTL.NS",
            "ITC.NS","LT.NS","AXISBANK.NS","MARUTI.NS","WIPRO.NS","HCLTECH.NS",
            "SUNPHARMA.NS","NTPC.NS","POWERGRID.NS","TITAN.NS"]
    heap=[]
    for s in sample:
        try:
            eng=Engine(s); prices,current,_=eng.fetch()
            score=((prices[-1]-prices[0])/prices[0])*100
            heapq.heappush(heap,(-score,s,round(current,2),round(score,2)))
        except: pass
    top=[]
    while heap and len(top)<5:
        _,sym,price,pct=heapq.heappop(heap)
        top.append({"symbol":sym,"price":price,"growth":pct})
    return jsonify(top)

@app.route('/history')
def get_history(): return jsonify(list(reversed(history_log)))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    history_log.clear(); save_history()
    return jsonify({"status":"cleared"})


if __name__ == '__main__':
    app.run(debug=True)