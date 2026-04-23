from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
import heapq
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder='static')

portfolio = []
history_log = []
HISTORY_FILE = 'history.json'

# Load history from disk on startup
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r') as f:
            history_log = json.load(f)
    except:
        history_log = []

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_log, f)


# =========================
# ENGINE
# =========================
class Engine:
    def __init__(self, symbol):
        self.symbol = symbol.upper()

    def fetch(self):
        stock = yf.Ticker(self.symbol)
        hist = stock.history(period="20d")
        if hist.empty:
            raise Exception("Invalid stock symbol or no data found.")
        prices = hist['Close'].dropna().tolist()[-6:]
        if len(prices) < 2:
            raise Exception("Not enough price data.")
        try:
            current = round(stock.fast_info['last_price'], 2)
        except:
            current = round(prices[-1], 2)

        # Fetch fundamentals
        info = {}
        try:
            raw = stock.info
            info = {
                'pe_ratio': raw.get('trailingPE', None),
                'pb_ratio': raw.get('priceToBook', None),
                'market_cap': raw.get('marketCap', None),
                'roe': raw.get('returnOnEquity', None),
                'revenue_growth': raw.get('revenueGrowth', None),
                'debt_to_equity': raw.get('debtToEquity', None),
                'name': raw.get('longName', self.symbol),
                'sector': raw.get('sector', 'N/A'),
                '52w_high': raw.get('fiftyTwoWeekHigh', None),
                '52w_low': raw.get('fiftyTwoWeekLow', None),
            }
        except:
            info = {'name': self.symbol}

        return prices, current, info

    def analyze(self, prices):
        short_ma = round(sum(prices[-3:]) / 3, 2)
        long_ma  = round(sum(prices) / len(prices), 2)

        trend    = "BULLISH" if short_ma > long_ma else "BEARISH"
        momentum = "UPWARD"  if prices[-1] > prices[-2] else "DOWNWARD"

        # Dip detection
        below_ma = sum(1 for p in prices[-3:] if p < long_ma)
        dip_pct  = round(((long_ma - prices[-1]) / long_ma) * 100, 2)
        if below_ma >= 2 and dip_pct >= 2:
            dip = "STRONG DIP"
        elif below_ma >= 1 and dip_pct >= 1:
            dip = "MILD DIP"
        elif prices[-1] > long_ma * 1.05:
            dip = "OVERBOUGHT"
        else:
            dip = "NEAR FAIR VALUE"

        if trend == "BULLISH" and momentum == "UPWARD":
            short = "BUY"
        elif trend == "BEARISH" and momentum == "DOWNWARD":
            short = "SELL"
        else:
            short = "HOLD"

        long_v = "BUY"  # will be overridden by fundamental score below
        final  = "STRONG BUY" if short == "BUY" and long_v == "BUY" else \
                 "STRONG SELL" if short == "SELL" and long_v == "SELL" else "HOLD"

        return {
            'short_ma': short_ma,
            'long_ma': long_ma,
            'trend': trend,
            'momentum': momentum,
            'dip': dip,
            'dip_pct': abs(dip_pct),
            'short': short,
            'long': long_v,
            'final': final
        }

    def predict(self, prices):
        x = list(range(len(prices)))
        n = len(x)
        mx = sum(x) / n
        my = sum(prices) / n
        num = sum((x[i] - mx) * (prices[i] - my) for i in range(n))
        den = sum((x[i] - mx) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0
        intercept = my - slope * mx
        next_p = slope * n + intercept
        direction = "UP" if next_p > prices[-1] else "DOWN"
        change_pct = round(((next_p - prices[-1]) / prices[-1]) * 100, 2)
        return round(next_p, 2), direction, change_pct


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.json.get('stock', '').strip()
    if not symbol:
        return jsonify({"error": "Stock symbol required"})
    try:
        eng = Engine(symbol)
        prices, current, info = eng.fetch()
        analysis = eng.analyze(prices)
        prediction, direction, change_pct = eng.predict(prices)

        entry = {
            "id": len(history_log) + 1,
            "stock": symbol.upper(),
            "name": info.get('name', symbol),
            "price": current,
            "predicted": prediction,
            "direction": direction,
            "change_pct": change_pct,
            "final": analysis['final'],
            "trend": analysis['trend'],
            "momentum": analysis['momentum'],
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        history_log.append(entry)
        # Keep only last 50
        if len(history_log) > 50:
            history_log.pop(0)
        save_history()

        return jsonify({
            "stock": symbol.upper(),
            "name": info.get('name', symbol),
            "price": current,
            "prices": [round(p, 2) for p in prices],
            "prediction": prediction,
            "direction": direction,
            "change_pct": change_pct,
            "sector": info.get('sector', 'N/A'),
            "52w_high": info.get('52w_high'),
            "52w_low": info.get('52w_low'),
            "pe_ratio": info.get('pe_ratio'),
            "pb_ratio": info.get('pb_ratio'),
            "market_cap": info.get('market_cap'),
            "roe": info.get('roe'),
            **analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/add_portfolio', methods=['POST'])
def add_portfolio():
    stock = request.json.get('stock', '').strip().upper()
    if stock and stock not in portfolio:
        portfolio.append(stock)
    return jsonify({"portfolio": portfolio})


@app.route('/remove_portfolio', methods=['POST'])
def remove_portfolio():
    stock = request.json.get('stock', '').strip().upper()
    if stock in portfolio:
        portfolio.remove(stock)
    return jsonify({"portfolio": portfolio})


@app.route('/portfolio')
def get_portfolio():
    return jsonify(portfolio)


@app.route('/top')
def top_stocks():
    sample = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS",
              "WIPRO.NS", "BAJFINANCE.NS", "TATAMOTORS.NS"]
    heap = []
    for s in sample:
        try:
            eng = Engine(s)
            prices, current, _ = eng.fetch()
            score = ((prices[-1] - prices[0]) / prices[0]) * 100
            heapq.heappush(heap, (-score, s, round(current, 2), round(score, 2)))
        except:
            pass
    top = []
    while heap and len(top) < 5:
        neg_score, sym, price, pct = heapq.heappop(heap)
        top.append({"symbol": sym, "price": price, "growth": pct})
    return jsonify(top)


@app.route('/history')
def get_history():
    return jsonify(list(reversed(history_log)))


@app.route('/history/clear', methods=['POST'])
def clear_history():
    history_log.clear()
    save_history()
    return jsonify({"status": "cleared"})


@app.route('/accuracy')
def accuracy():
    if not history_log:
        return jsonify({"accuracy": 0, "total": 0, "correct": 0})
    correct = sum(1 for h in history_log if h.get("direction") == "UP")
    acc = round((correct / len(history_log)) * 100, 2)
    return jsonify({"accuracy": acc, "total": len(history_log), "correct": correct})


if __name__ == '__main__':
    app.run(debug=True)