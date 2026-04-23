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
        info = {}
        try:
            raw = stock.info
            info = {
                'pe_ratio':      raw.get('trailingPE', None),
                'pb_ratio':      raw.get('priceToBook', None),
                'market_cap':    raw.get('marketCap', None),
                'roe':           raw.get('returnOnEquity', None),
                'debt_to_equity':raw.get('debtToEquity', None),
                'name':          raw.get('longName', self.symbol),
                'sector':        raw.get('sector', 'N/A'),
                '52w_high':      raw.get('fiftyTwoWeekHigh', None),
                '52w_low':       raw.get('fiftyTwoWeekLow', None),
            }
        except:
            info = {'name': self.symbol}
        return prices, current, info

    def fetch_extended(self, period="6mo"):
        stock = yf.Ticker(self.symbol)
        hist = stock.history(period=period)
        if hist.empty:
            raise Exception("Invalid stock symbol or no data found.")
        closes = hist['Close'].dropna().tolist()
        dates  = [str(d.date()) for d in hist.index]
        return closes, dates

    def analyze(self, prices):
        short_ma = round(sum(prices[-3:]) / 3, 2)
        long_ma  = round(sum(prices) / len(prices), 2)
        trend    = "BULLISH" if short_ma > long_ma else "BEARISH"
        momentum = "UPWARD"  if prices[-1] > prices[-2] else "DOWNWARD"
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
        long_v = "BUY"
        final  = "STRONG BUY"  if short == "BUY"  and long_v == "BUY"  else \
                 "STRONG SELL" if short == "SELL" and long_v == "SELL" else "HOLD"
        return {'short_ma': short_ma, 'long_ma': long_ma, 'trend': trend,
                'momentum': momentum, 'dip': dip, 'dip_pct': abs(dip_pct),
                'short': short, 'long': long_v, 'final': final}

    def predict(self, prices):
        x = list(range(len(prices)))
        n = len(x)
        mx = sum(x) / n
        my = sum(prices) / n
        num = sum((x[i]-mx)*(prices[i]-my) for i in range(n))
        den = sum((x[i]-mx)**2 for i in range(n))
        slope = num/den if den != 0 else 0
        intercept = my - slope*mx
        next_p = slope*n + intercept
        direction  = "UP" if next_p > prices[-1] else "DOWN"
        change_pct = round(((next_p - prices[-1]) / prices[-1]) * 100, 2)
        return round(next_p, 2), direction, change_pct


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
            "id": len(history_log)+1, "stock": symbol.upper(),
            "name": info.get('name', symbol), "price": current,
            "predicted": prediction, "direction": direction,
            "change_pct": change_pct, "final": analysis['final'],
            "trend": analysis['trend'], "momentum": analysis['momentum'],
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        history_log.append(entry)
        if len(history_log) > 50: history_log.pop(0)
        save_history()
        return jsonify({
            "stock": symbol.upper(), "name": info.get('name', symbol),
            "price": current, "prices": [round(p,2) for p in prices],
            "prediction": prediction, "direction": direction,
            "change_pct": change_pct, "sector": info.get('sector','N/A'),
            "52w_high": info.get('52w_high'), "52w_low": info.get('52w_low'),
            "pe_ratio": info.get('pe_ratio'), "pb_ratio": info.get('pb_ratio'),
            "market_cap": info.get('market_cap'), "roe": info.get('roe'),
            **analysis
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

        trades = []
        wins = losses = 0
        total_return = 0.0
        in_trade = False
        buy_price = buy_date = 0

        for i in range(6, len(closes)):
            window   = closes[i-6:i]
            short_ma = sum(window[-3:]) / 3
            long_ma  = sum(window) / 6
            trend    = "BULLISH" if short_ma > long_ma else "BEARISH"
            momentum = "UPWARD"  if window[-1] > window[-2] else "DOWNWARD"
            signal   = "BUY"  if trend == "BULLISH" and momentum == "UPWARD" else \
                       "SELL" if trend == "BEARISH" and momentum == "DOWNWARD" else "HOLD"
            price = closes[i]
            date  = dates[i]

            if signal == "BUY" and not in_trade:
                in_trade = True; buy_price = price; buy_date = date
            elif signal == "SELL" and in_trade:
                in_trade = False
                pnl = round(((price - buy_price) / buy_price) * 100, 2)
                won = pnl > 0
                wins   += 1 if won else 0
                losses += 0 if won else 1
                total_return += pnl
                trades.append({"buy_date": buy_date, "sell_date": date,
                                "buy_price": round(buy_price,2), "sell_price": round(price,2),
                                "pnl": pnl, "result": "WIN" if won else "LOSS"})

        if in_trade:
            price = closes[-1]
            pnl   = round(((price - buy_price) / buy_price) * 100, 2)
            won   = pnl > 0
            wins  += 1 if won else 0
            losses+= 0 if won else 1
            total_return += pnl
            trades.append({"buy_date": buy_date, "sell_date": dates[-1]+" (open)",
                           "buy_price": round(buy_price,2), "sell_price": round(price,2),
                           "pnl": pnl, "result": "WIN" if won else "LOSS"})

        total_trades = wins + losses
        win_rate     = round((wins/total_trades)*100, 1) if total_trades else 0
        avg_return   = round(total_return/total_trades, 2) if total_trades else 0
        step         = max(1, len(closes)//60)
        chart_prices = [round(closes[i],2) for i in range(0, len(closes), step)]
        chart_dates  = [dates[i] for i in range(0, len(dates), step)]

        return jsonify({
            "stock": symbol.upper(), "period": period,
            "total_trades": total_trades, "wins": wins, "losses": losses,
            "win_rate": win_rate, "total_return": round(total_return,2),
            "avg_return": avg_return, "trades": trades[-20:],
            "chart_prices": chart_prices, "chart_dates": chart_dates,
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
            analysis = eng.analyze(prices)
            prediction, direction, change_pct = eng.predict(prices)
            score = 0
            if analysis['trend']    == 'BULLISH': score += 2
            if analysis['momentum'] == 'UPWARD':  score += 2
            if direction            == 'UP':       score += 1
            if analysis['dip'] in ('STRONG DIP','MILD DIP'): score += 2
            results.append({
                "stock": sym, "name": info.get('name', sym), "price": current,
                "prediction": prediction, "direction": direction, "change_pct": change_pct,
                "trend": analysis['trend'], "momentum": analysis['momentum'],
                "dip": analysis['dip'], "final": analysis['final'],
                "short_ma": analysis['short_ma'], "long_ma": analysis['long_ma'],
                "pe_ratio": info.get('pe_ratio'), "pb_ratio": info.get('pb_ratio'),
                "market_cap": info.get('market_cap'), "roe": info.get('roe'),
                "sector": info.get('sector','N/A'),
                "52w_high": info.get('52w_high'), "52w_low": info.get('52w_low'),
                "prices": [round(p,2) for p in prices], "score": score, "error": None
            })
        except Exception as e:
            results.append({"stock": sym, "error": str(e)})

    valid = [r for r in results if not r.get('error')]
    valid.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(valid): r['rank'] = i + 1
    return jsonify(results)


@app.route('/add_portfolio', methods=['POST'])
def add_portfolio():
    stock = request.json.get('stock','').strip().upper()
    if stock and stock not in portfolio:
        portfolio.append(stock)
    return jsonify({"portfolio": portfolio})

@app.route('/remove_portfolio', methods=['POST'])
def remove_portfolio():
    stock = request.json.get('stock','').strip().upper()
    if stock in portfolio: portfolio.remove(stock)
    return jsonify({"portfolio": portfolio})

@app.route('/portfolio')
def get_portfolio():
    return jsonify(portfolio)

@app.route('/top')
def top_stocks():
    sample = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ITC.NS",
              "WIPRO.NS","BAJFINANCE.NS","TATAMOTORS.NS"]
    heap = []
    for s in sample:
        try:
            eng = Engine(s)
            prices, current, _ = eng.fetch()
            score = ((prices[-1]-prices[0])/prices[0])*100
            heapq.heappush(heap, (-score, s, round(current,2), round(score,2)))
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
    history_log.clear(); save_history()
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True)