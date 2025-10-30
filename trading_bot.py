import os, time, json, traceback, threading
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import pytz

load_dotenv()

# Timezone support - IST
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    return datetime.now(IST)

def utc_to_ist(utc_dt):
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(IST)

def format_ist_time(dt):
    if dt is None:
        return "N/A"
    ist_dt = utc_to_ist(dt) if dt.tzinfo is None or dt.tzinfo == pytz.utc else dt
    return ist_dt.strftime('%Y-%m-%d %I:%M:%S %p IST')

# CONFIG - 5min/15min TESTING WITH REALISTIC PARAMETERS
MODE = os.getenv("MODE", "paper").lower()
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kucoin")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
ENTRY_TF = os.getenv("ENTRY_TF", "5m")      # 🎯 CHANGED: 5min for testing
HTF = os.getenv("HTF", "15m")               # 🎯 CHANGED: 15min for testing

TOTAL_PORTFOLIO_CAPITAL = float(os.getenv("TOTAL_PORTFOLIO_CAPITAL", "10000"))
PER_COIN_ALLOCATION = float(os.getenv("PER_COIN_ALLOCATION", "0.20"))
PER_COIN_CAP_USD = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

# 🎯 REALISTIC PARAMETERS FOR 5min/15min TESTING
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.003"))    # 0.3% risk (even lower for high frequency)
RR_FIXED = float(os.getenv("RR_FIXED", "1.5"))              # 1:1.5 R:R (achievable on short TF)
DYNAMIC_RR = os.getenv("DYNAMIC_RR", "false").lower() == "true"
MIN_RR = float(os.getenv("MIN_RR", "1.2"))
MAX_RR = float(os.getenv("MAX_RR", "1.8"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.2"))        # Tighter stops for short TF
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "true").lower() == "true"
USE_H1_FILTER = os.getenv("USE_H1_FILTER", "true").lower() == "true"

USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "true").lower() == "true"  # Enable for short TF
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "1.2"))    # Higher volume requirement
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))

# 🎯 ADJUSTED FOR HIGH FREQUENCY TRADING
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "0.5"))   # 30 minutes between trades
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "8")) # More trades allowed
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))     # 10% max drawdown (tighter)
MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "100000"))

# 🎯 REALISTIC COSTS - NO SLIPPAGE DEDUCTIONS
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% fees only

SLEEP_CAP = int(os.getenv("SLEEP_CAP", "30"))     # Shorter sleep for 5min
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

API_KEY = os.getenv("KUCOIN_API_KEY", "")
API_SECRET = os.getenv("KUCOIN_SECRET", "")
API_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LOG_PREFIX = "[BOT]"

if MODE == "live":
    if not API_KEY or not API_SECRET or not API_PASSPHRASE:
        raise ValueError("KuCoin API credentials not found!")

# Utilities
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM] No credentials - would send: {msg}")
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
        if response.status_code == 200:
            print(f"[TELEGRAM] ✅ Sent")
        else:
            print(f"[TELEGRAM] ❌ Failed ({response.status_code})")
    except Exception as e:
        print(f"[TELEGRAM] ❌ Error: {e}")

def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")

def get_exchange():
    if MODE == "live":
        ex = getattr(ccxt, EXCHANGE_ID)({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    else:
        ex = getattr(ccxt, EXCHANGE_ID)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    return ex

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            print(f"[DATA] ❌ {symbol} {timeframe}: No OHLCV data")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df_clean = df.dropna()
        if DEBUG_MODE:
            latest_ist = utc_to_ist(df_clean.index[-1])
            print(f"[DATA] ✅ {symbol} {timeframe}: {len(df_clean)} bars | Latest: {latest_ist.strftime('%Y-%m-%d %I:%M %p IST')} | Close: {df_clean['Close'].iloc[-1]:.2f}")
        return df_clean
    except Exception as e:
        print(f"[DATA] ❌ {symbol} {timeframe} error: {e}")
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

def calculate_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def state_files_for_symbol(symbol: str):
    tag = symbol.replace("/", "_")
    return f"state_{tag}.json", f"{tag}_live_trades.csv"

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            s = json.load(f)
        s["entry_time"] = pd.to_datetime(s["entry_time"]) if s.get("entry_time") else None
        s["last_processed_ts"] = pd.to_datetime(s["last_processed_ts"]) if s.get("last_processed_ts") else None
        s["last_exit_time"] = pd.to_datetime(s["last_exit_time"]) if s.get("last_exit_time") else None
        
        if "total_trades" not in s:
            s["total_trades"] = 0
        if "winning_trades" not in s:
            s["winning_trades"] = 0
        if "losing_trades" not in s:
            s["losing_trades"] = 0
        if "total_pnl" not in s:
            s["total_pnl"] = 0.0
        if "total_fees_paid" not in s:
            s["total_fees_paid"] = 0.0
        if "trades_today" not in s:
            s["trades_today"] = 0
        if "last_trade_date" not in s:
            s["last_trade_date"] = None
        
        return s
    return {
        "capital": PER_COIN_CAP_USD,
        "position": 0,
        "entry_price": 0.0,
        "entry_sl": 0.0,
        "entry_tp": 0.0,
        "entry_time": None,
        "entry_size": 0.0,
        "peak_equity": PER_COIN_CAP_USD,
        "last_processed_ts": None,
        "last_exit_time": None,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "total_fees_paid": 0.0,
        "trades_today": 0,
        "last_trade_date": None
    }

def save_state(state_file, state):
    s = dict(state)
    s["entry_time"] = state["entry_time"].isoformat() if state["entry_time"] is not None else None
    s["last_processed_ts"] = state["last_processed_ts"].isoformat() if state["last_processed_ts"] is not None else None
    s["last_exit_time"] = state["last_exit_time"].isoformat() if state.get("last_exit_time") is not None else None
    s["last_trade_date"] = state["last_trade_date"].isoformat() if state.get("last_trade_date") is not None else None
    with open(state_file, "w") as f:
        json.dump(s, f, indent=2)

def append_trade(csv_file, row):
    write_header = not os.path.exists(csv_file)
    pd.DataFrame([row]).to_csv(csv_file, mode="a", header=write_header, index=False)

class MarketInfo:
    def __init__(self, exchange, symbol):
        mkts = exchange.load_markets()
        m = mkts[symbol]
        self.base = m["base"]
        self.quote = m["quote"]
        self.amount_min = m["limits"]["amount"]["min"] or 0.0
        self.cost_min = (m["limits"].get("cost") or {}).get("min") or 0.0
        self.amount_prec = m.get("precision", {}).get("amount", 6)
        self.price_prec = m.get("precision", {}).get("price", 6)
        self.symbol = symbol
        self.exchange = exchange
    def round_amount(self, amt): return float(self.exchange.amount_to_precision(self.symbol, amt))
    def round_price(self, px): return float(self.exchange.price_to_precision(self.symbol, px))

def place_market_buy(exchange, mi: MarketInfo, base_qty: float):
    base_qty = mi.round_amount(max(base_qty, mi.amount_min))
    if base_qty <= 0: raise ValueError(f"Amount too small: {base_qty}")
    return exchange.create_market_buy_order(mi.symbol, base_qty)

def place_market_sell(exchange, mi: MarketInfo, base_qty: float):
    base_qty = mi.round_amount(base_qty)
    if base_qty <= 0: raise ValueError(f"Sell amount too small: {base_qty}")
    return exchange.create_market_sell_order(mi.symbol, base_qty)

def avg_fill_price_from_order(order):
    price = order.get("average") or order.get("price")
    if price: return float(price)
    if "trades" in order and order["trades"]:
        notional = 0.0; qty = 0.0
        for t in order["trades"]:
            p = float(t["price"]); a = float(t["amount"])
            notional += p*a; qty += a
        if qty > 0: return notional / qty
    return None

# 🎯 REALISTIC Spot Trading Position Sizing for 5min/15min
def calculate_spot_position_size(price, sl, capital, risk_percent, max_trade_size):
    """Calculate REALISTIC position size for short timeframe trading"""
    risk_per_trade = capital * risk_percent  # e.g., $2000 × 0.3% = $6
    risk_per_coin = abs(price - sl)          # e.g., $100 - $99.80 = $0.20
    
    # Maximum shares based on risk management
    max_by_risk = risk_per_trade / risk_per_coin if risk_per_coin > 0 else 0
    
    # Maximum shares based on available capital (SPOT TRADING CONSTRAINT)
    max_by_capital = capital / price
    
    # Take the minimum - you're limited by actual capital in spot trading!
    size_base = min(max_by_risk, max_by_capital, max_trade_size)
    
    # Calculate ACTUAL risk percentage (for reporting)
    actual_risk_pct = (size_base * risk_per_coin) / capital * 100
    
    if DEBUG_MODE:
        print(f"[POSITION SIZING] Desired risk: {risk_percent*100}% (${risk_per_trade:.2f})")
        print(f"[POSITION SIZING] Actual risk: {actual_risk_pct:.2f}% (${size_base * risk_per_coin:.2f})")
    
    return max(size_base, 0)

# 🎯 ENHANCED: Better entry filters for 5min/15min timeframe
def is_high_quality_5min_setup(current_bar, prev_bar, htf_trend):
    """Enhanced filters for 5min/15min timeframe trading"""
    
    # 1. Minimum candle size filter (avoid noise)
    candle_size_pct = (current_bar['High'] - current_bar['Low']) / current_bar['Low']
    if candle_size_pct < 0.003:  # Minimum 0.3% candle size
        if DEBUG_MODE:
            print(f"❌ Rejected: Candle too small ({candle_size_pct:.3%})")
        return False
    
    # 2. Strong volume confirmation (critical for short TF)
    if USE_VOLUME_FILTER and 'Avg_Volume' in current_bar and not np.isnan(current_bar['Avg_Volume']):
        volume_ratio = current_bar['Volume'] / current_bar['Avg_Volume']
        if volume_ratio < VOL_MIN_RATIO:
            if DEBUG_MODE:
                print(f"❌ Rejected: Low volume (ratio: {volume_ratio:.2f})")
            return False
    
    # 3. RSI momentum filter (avoid extremes)
    rsi = current_bar['RSI']
    if not (25 < rsi < 75):  # Tighter range for short TF
        if DEBUG_MODE:
            print(f"❌ Rejected: RSI at extreme ({rsi:.1f})")
        return False
    
    # 4. Higher timeframe trend alignment
    if USE_H1_FILTER and htf_trend != 1:
        if DEBUG_MODE:
            print(f"❌ Rejected: Not in 15min uptrend (Trend: {htf_trend})")
        return False
    
    return True

# 🎯 FIXED: Realistic cost calculations - NO SLIPPAGE DEDUCTIONS
def process_bar(symbol, entry_df, htf_df, state, exchange=None, market_info: MarketInfo=None):
    """
    🔥 OPTIMIZED for 5min/15min timeframe testing
    """
    if len(entry_df) < 3:
        return state, None
    
    # Reset daily trade counter if new day
    current_date = datetime.now().date()
    if state.get("last_trade_date") != current_date:
        state["trades_today"] = 0
        state["last_trade_date"] = current_date
    
    # Check daily trade limit
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        if DEBUG_MODE:
            print(f"[TRADE LIMIT] {symbol}: Already {state['trades_today']}/{MAX_TRADES_PER_DAY} trades today")
        state["last_processed_ts"] = entry_df.index[-1]
        return state, None
    
    entry_df_work = entry_df.copy()
    entry_df_work["Bias"] = 0
    entry_df_work.loc[entry_df_work["Close"] > entry_df_work["Close"].shift(1), "Bias"] = 1
    entry_df_work.loc[entry_df_work["Close"] < entry_df_work["Close"].shift(1), "Bias"] = -1
    
    h = htf_df.copy()
    h["Trend"] = 0
    h.loc[h["Close"] > h["Close"].shift(1), "Trend"] = 1
    h.loc[h["Close"] < h["Close"].shift(1), "Trend"] = -1
    
    entry_df_work["HTF_Trend"] = h["Trend"].reindex(entry_df_work.index, method="ffill").fillna(0).astype(int)
    entry_df_work["ATR"] = calculate_atr(entry_df_work, ATR_PERIOD) if USE_ATR_STOPS else np.nan
    
    if USE_VOLUME_FILTER:
        entry_df_work["Avg_Volume"] = entry_df_work["Volume"].rolling(VOL_LOOKBACK).mean()
    
    entry_df_work["RSI"] = calculate_rsi(entry_df_work["Close"], RSI_PERIOD)
    
    i = len(entry_df_work) - 1
    current_bar = entry_df_work.iloc[i]
    ts = entry_df_work.index[i]
    
    price = float(current_bar["Close"])
    open_price = float(current_bar["Open"])
    bias = int(current_bar["Bias"])
    htf_trend = int(current_bar["HTF_Trend"])
    
    if i >= 1:
        prev_close = float(entry_df_work['Close'].iloc[i-1])
        bullish_sweep = (price > open_price) and (price > prev_close)
    else:
        bullish_sweep = False
    
    if DEBUG_MODE:
        ts_ist = utc_to_ist(ts)
        print(f"\n{'='*80}")
        print(f"[DEBUG] {symbol} | 5min Bar[{i}] @ {ts_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"[DEBUG] OHLC: O={open_price:.4f} H={current_bar['High']:.4f} L={current_bar['Low']:.4f} C={price:.4f}")
        if i >= 1:
            print(f"[DEBUG] Prev Close: {prev_close:.4f} | Sweep: {bullish_sweep}")
        print(f"[DEBUG] RSI: {current_bar['RSI']:.1f} | Bias: {bias} | 15min Trend: {htf_trend}")
        print(f"[DEBUG] Position: {state['position']} | Cap: ${state['capital']:.2f}")
        print(f"[DEBUG] Stats: {state['winning_trades']}W / {state['losing_trades']}L | Total PnL: ${state['total_pnl']:.2f}")
        print(f"[DEBUG] Trades Today: {state['trades_today']}/{MAX_TRADES_PER_DAY}")
        print(f"{'='*80}\n")
    
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    curr_dd = (state["peak_equity"] - state["capital"]) / state["peak_equity"] if state["peak_equity"] > 0 else 0.0
    blocked = curr_dd >= MAX_DRAWDOWN
    
    trade_row = None
    
    # Permanent stop
    if blocked and not state.get("permanently_stopped", False):
        state["permanently_stopped"] = True
        
        if state["position"] == 1:
            if MODE == "live":
                try:
                    order = place_market_sell(exchange, market_info, state["entry_size"])
                    exit_price = float(avg_fill_price_from_order(order) or price)
                except Exception as e:
                    send_telegram(f"❌ {symbol} FORCED EXIT error: {e}")
                    raise
            else:
                exit_price = price
            
            # 🎯 FIXED: Only deduct ACTUAL fees (no slippage deductions)
            position_value_at_exit = exit_price * state["entry_size"]
            exit_fee = position_value_at_exit * FEE_RATE
            
            position_value_at_entry = state["entry_price"] * state["entry_size"]
            entry_fee = position_value_at_entry * FEE_RATE
            
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            total_fees = entry_fee + exit_fee
            # ✅ CORRECT: Deduct BOTH entry and exit fees
            net_pnl = gross_pnl - entry_fee - exit_fee  # Only deduct exit fee
            
            state["capital"] += net_pnl
            state["total_trades"] += 1
            state["trades_today"] += 1
            state["total_pnl"] += net_pnl
            state["total_fees_paid"] += total_fees
            
            if net_pnl > 0:
                state["winning_trades"] += 1
            else:
                state["losing_trades"] += 1
            
            trade_row = {
                "Symbol": symbol,
                "Trade_ID": int(time.time()),
                "Entry_DateTime": state["entry_time"].isoformat(),
                "Exit_DateTime": ts.isoformat(),
                "Position": "Long",
                "Entry_Price": round(state["entry_price"], 6),
                "Exit_Price": round(exit_price, 6),
                "Take_Profit": round(state["entry_tp"], 6),
                "Stop_Loss": round(state["entry_sl"], 6),
                "Position_Size_Base": round(state["entry_size"], 8),
                "Gross_PnL_$": round(gross_pnl, 2),
                "Entry_Fee_$": round(entry_fee, 2),
                "Exit_Fee_$": round(exit_fee, 2),
                "Total_Fees_$": round(total_fees, 2),
                "Net_PnL_$": round(net_pnl, 2),
                "Win": 1 if net_pnl > 0 else 0,
                "Exit_Reason": "MAX DRAWDOWN",
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }
            
            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts
            
            send_telegram(f"🛑 {symbol} PERMANENTLY STOPPED!")
            return state, trade_row
    
    blocked = state.get("permanently_stopped", False)
    
    # Exit logic - Only TP or SL
    if state["position"] == 1 and not blocked:
        exit_flag = False
        exit_price = price
        exit_reason = ""
        
        if price >= state["entry_tp"]:
            exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
        elif price <= state["entry_sl"]:
            exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
        
        if exit_flag:
            if MODE == "live":
                try:
                    order = place_market_sell(exchange, market_info, state["entry_size"])
                    exit_price = float(avg_fill_price_from_order(order) or price)
                except Exception as e:
                    send_telegram(f"❌ {symbol} Exit error: {e}")
                    raise
            
            # 🎯 FIXED: Only deduct ACTUAL fees (no slippage deductions)
            position_value_at_exit = exit_price * state["entry_size"]
            exit_fee = position_value_at_exit * FEE_RATE
            
            position_value_at_entry = state["entry_price"] * state["entry_size"]
            entry_fee = position_value_at_entry * FEE_RATE
            
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            total_fees = entry_fee + exit_fee
            net_pnl = gross_pnl - exit_fee  # Only deduct exit fee
            
            state["capital"] += net_pnl
            state["total_trades"] += 1
            state["trades_today"] += 1
            state["total_pnl"] += net_pnl
            state["total_fees_paid"] += total_fees
            
            if net_pnl > 0:
                state["winning_trades"] += 1
            else:
                state["losing_trades"] += 1
            
            win_rate = (state["winning_trades"] / state["total_trades"] * 100) if state["total_trades"] > 0 else 0
            
            trade_row = {
                "Symbol": symbol,
                "Trade_ID": int(time.time()),
                "Entry_DateTime": state["entry_time"].isoformat(),
                "Exit_DateTime": ts.isoformat(),
                "Position": "Long",
                "Entry_Price": round(state["entry_price"], 6),
                "Exit_Price": round(exit_price, 6),
                "Take_Profit": round(state["entry_tp"], 6),
                "Stop_Loss": round(state["entry_sl"], 6),
                "Position_Size_Base": round(state["entry_size"], 8),
                "Gross_PnL_$": round(gross_pnl, 2),
                "Entry_Fee_$": round(entry_fee, 2),
                "Exit_Fee_$": round(exit_fee, 2),
                "Total_Fees_$": round(total_fees, 2),
                "Net_PnL_$": round(net_pnl, 2),
                "Win": 1 if net_pnl > 0 else 0,
                "Exit_Reason": exit_reason,
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }
            
            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts
            
            pnl_emoji = "💚" if net_pnl > 0 else "❤️"
            exit_msg = f"""
{pnl_emoji} EXIT {symbol} | {exit_reason}
━━━━━━━━━━━━━━━━━━━━━━━
📍 Exit Price: ${exit_price:.4f}
💰 Gross PnL: ${gross_pnl:.2f}
💸 Fees: ${total_fees:.2f} ({FEE_RATE*100}%)
💵 Net PnL: ${net_pnl:.2f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Capital: ${state['capital']:.2f}
📊 Stats: {state['winning_trades']}W / {state['losing_trades']}L
📈 Win Rate: {win_rate:.1f}%
💎 Total PnL: ${state['total_pnl']:.2f}
🎯 Trades Today: {state['trades_today']}/{MAX_TRADES_PER_DAY}
"""
            print(exit_msg)
            send_telegram(exit_msg)
    
    # Entry logic with ENHANCED filters
    if state["position"] == 0 and not blocked and state["trades_today"] < MAX_TRADES_PER_DAY:
        if COOLDOWN_HOURS > 0 and state.get("last_exit_time") is not None:
            time_diff_hours = (ts - state["last_exit_time"]).total_seconds() / 3600
            if time_diff_hours < COOLDOWN_HOURS:
                if DEBUG_MODE:
                    print(f"[COOLDOWN] {symbol}: {time_diff_hours:.1f}h since last trade, need {COOLDOWN_HOURS}h")
                state["last_processed_ts"] = ts
                return state, None
        
        # 🎯 ENHANCED: Use better filters for 5min/15min
        basic_conditions = bias == 1 and bullish_sweep
        quality_conditions = is_high_quality_5min_setup(current_bar, current_bar, htf_trend)
        
        if DEBUG_MODE:
            print(f"[ENTRY CHECK] {symbol}")
            print(f"  Basic Conditions: {basic_conditions}")
            print(f"  Quality Filters: {quality_conditions}")
            print(f"  Trades Today: {state['trades_today']}/{MAX_TRADES_PER_DAY}")
        
        if basic_conditions and quality_conditions:
            if USE_ATR_STOPS:
                atr_val = float(current_bar["ATR"])
                if np.isnan(atr_val) or atr_val <= 0:
                    state["last_processed_ts"] = ts
                    return state, trade_row
                sl = price - (ATR_MULT_SL * atr_val)
            else:
                sweep_buffer = min(max(price * 0.0005, 0.0005), 0.0015)
                sl = price * (1 - sweep_buffer)
            
            risk = abs(price - sl)
            if risk <= 0:
                state["last_processed_ts"] = ts
                return state, trade_row
            
            rr_ratio = RR_FIXED
            
            tp = price + rr_ratio * risk
            
            # 🎯 FIXED: Use realistic spot trading position sizing
            size_base = calculate_spot_position_size(
                price, sl, state["capital"], RISK_PERCENT, MAX_TRADE_SIZE
            )
            
            if DEBUG_MODE:
                risk_amount = state["capital"] * RISK_PERCENT
                max_by_capital = state["capital"] / price
                actual_risk_amount = size_base * risk
                print(f"[POSITION SIZING] Price: ${price:.4f}, SL: ${sl:.4f}")
                print(f"  Desired Risk: ${risk_amount:.2f} ({RISK_PERCENT*100}%)")
                print(f"  Actual Risk: ${actual_risk_amount:.2f} ({actual_risk_amount/state['capital']*100:.2f}%)")
                print(f"  Max by Capital: {max_by_capital:.2f} coins")
                print(f"  Final Size: {size_base:.6f} coins")
                print(f"[ENTRY] Setup: Entry=${price:.4f} SL=${sl:.4f} TP=${tp:.4f} RR={rr_ratio:.1f}")
            
            if size_base > 0:
                entry_price_used = price
                
                if MODE == "live":
                    try:
                        size_base = max(size_base, market_info.amount_min)
                        size_base = market_info.round_amount(size_base)
                        order = place_market_buy(exchange, market_info, size_base)
                        entry_price_used = float(avg_fill_price_from_order(order) or price)
                    except Exception as e:
                        send_telegram(f"❌ {symbol} Entry error: {e}")
                        raise
                
                # 🎯 FIXED: Only deduct ACTUAL fees (no slippage)
                position_value = entry_price_used * size_base
                entry_fee = position_value * FEE_RATE
                entry_costs = entry_fee  # Only fees
                
                state["position"] = 1
                state["entry_price"] = entry_price_used
                state["entry_sl"] = sl
                state["entry_tp"] = tp
                state["entry_time"] = ts
                state["entry_size"] = size_base
                
                # 🎯 FIXED: Deduct only fees (no slippage)
                state["capital"] -= entry_costs
                
                actual_risk_amount = size_base * abs(entry_price_used - sl)
                actual_risk_pct = (actual_risk_amount / (state["capital"] + entry_costs)) * 100
                
                entry_msg = f"""
🚀 ENTRY {symbol} (5min/15min)
━━━━━━━━━━━━━━━━━━━━━━━
📍 Entry Price: ${entry_price_used:.4f}
🎯 Stop Loss: ${sl:.4f}
🎯 Take Profit: ${tp:.4f}
📊 Risk:Reward: 1:{rr_ratio:.1f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Position Size: {size_base:.6f} {symbol.split('/')[0]}
💰 Position Value: ${position_value:.2f}
💸 Entry Fee: ${entry_fee:.2f} (${position_value:.2f} × {FEE_RATE*100}%)
💵 Total Entry Cost: ${entry_costs:.2f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Capital Before: ${state['capital'] + entry_costs:.2f}
💼 Capital After: ${state['capital']:.2f}
💎 Actual Risk: ${actual_risk_amount:.2f} ({actual_risk_pct:.2f}%)
━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Trades: {state['total_trades']}
📈 Record: {state['winning_trades']}W / {state['losing_trades']}L
💰 Total PnL: ${state['total_pnl']:.2f}
🎯 Trades Today: {state['trades_today']+1}/{MAX_TRADES_PER_DAY}
"""
                print(entry_msg)
                send_telegram(entry_msg)
        else:
            if DEBUG_MODE:
                missing = []
                if bias != 1: missing.append(f"Bias({bias})")
                if not bullish_sweep: missing.append("Sweep")
                if not quality_conditions: missing.append("QualityFilters")
                print(f"❌ Missing: {', '.join(missing)}")
    
    state["last_processed_ts"] = ts
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    return state, trade_row

def worker(symbol):
    state_file, trades_csv = state_files_for_symbol(symbol)
    exchange = get_exchange()
    market_info = MarketInfo(exchange, symbol) if MODE == "live" else None
    state = load_state(state_file)
    tf_minutes = timeframe_to_minutes(ENTRY_TF)

    initial_msg = f"""
🤖 {symbol} Worker Started (5min/15min)
━━━━━━━━━━━━━━━━━━━━━━━
💼 Starting Capital: ${state['capital']:.2f}
📊 Previous Stats:
  • Total Trades: {state['total_trades']}
  • Wins: {state['winning_trades']}
  • Losses: {state['losing_trades']}
  • Total PnL: ${state['total_pnl']:.2f}
  • Fees Paid: ${state['total_fees_paid']:.2f}
🎯 5min/15min Settings:
  • Max Trades/Day: {MAX_TRADES_PER_DAY}
  • Risk/Trade: {RISK_PERCENT*100}%
  • Risk:Reward: 1:{RR_FIXED}
  • Cooldown: {COOLDOWN_HOURS}h
━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(initial_msg)
    send_telegram(initial_msg.replace('━', '-'))

    while True:
        try:
            now_utc = datetime.now(pytz.utc)
            
            entry_df = fetch_ohlcv_df(exchange, symbol, ENTRY_TF, limit=500)
            htf_df = fetch_ohlcv_df(exchange, symbol, HTF, limit=600)

            if entry_df.empty or htf_df.empty or len(entry_df) < 3:
                print(f"{LOG_PREFIX} {symbol} | No data; wait 30s")
                time.sleep(30)
                continue

            closed_candle_ts = entry_df.index[-2]
            forming_candle_ts = entry_df.index[-1]

            if state["last_processed_ts"] is None:
                state["last_processed_ts"] = closed_candle_ts
                save_state(state_file, state)
                time.sleep(10)
                continue

            if closed_candle_ts > state["last_processed_ts"]:
                if DEBUG_MODE:
                    print(f"\n[WORKER] {symbol} | 🆕 5min CANDLE @ {utc_to_ist(closed_candle_ts).strftime('%I:%M %p')}")
                
                state, trade = process_bar(symbol, entry_df, htf_df, state, exchange=exchange, market_info=market_info)
                
                if trade is not None:
                    append_trade(trades_csv, trade)
                    
                    trade_summary = f"""
{'='*80}
📋 TRADE #{state['total_trades']} COMPLETED - {symbol} (5min/15min)
{'='*80}
Entry: {format_ist_time(pd.to_datetime(trade['Entry_DateTime']))}
Exit:  {format_ist_time(pd.to_datetime(trade['Exit_DateTime']))}

💰 PRICES:
  Entry: ${trade['Entry_Price']:.4f}
  Exit:  ${trade['Exit_Price']:.4f}
  Stop Loss: ${trade['Stop_Loss']:.4f}
  Take Profit: ${trade['Take_Profit']:.4f}

💵 P&L BREAKDOWN:
  Gross PnL:       ${trade['Gross_PnL_$']:.2f}
  Entry Fee:      -${trade['Entry_Fee_$']:.2f}
  Exit Fee:       -${trade['Exit_Fee_$']:.2f}
  ─────────────────────────────
  Net PnL:         ${trade['Net_PnL_$']:.2f} {'✅ WIN' if trade['Win'] else '❌ LOSS'}

📊 STATISTICS ({symbol}):
  Total Trades: {state['total_trades']}
  Wins: {state['winning_trades']} | Losses: {state['losing_trades']}
  Win Rate: {(state['winning_trades']/state['total_trades']*100) if state['total_trades'] > 0 else 0:.1f}%
  Total PnL: ${state['total_pnl']:.2f}
  Total Fees Paid: ${state['total_fees_paid']:.2f}
  Trades Today: {state['trades_today']}/{MAX_TRADES_PER_DAY}

💼 CAPITAL:
  Capital After Trade: ${trade['Capital_After']:.2f}
  Starting Capital: ${PER_COIN_CAP_USD:.2f}
  Return: {((trade['Capital_After']/PER_COIN_CAP_USD - 1) * 100):.2f}%

Exit Reason: {trade['Exit_Reason']}
{'='*80}
"""
                    print(trade_summary)
                
                save_state(state_file, state)

            next_close_utc = forming_candle_ts + timedelta(minutes=tf_minutes)
            safe_check_time_utc = next_close_utc + timedelta(minutes=1)
            sleep_sec = (safe_check_time_utc - now_utc.replace(tzinfo=None)).total_seconds()
            
            if sleep_sec < 10:
                sleep_sec = 10
            elif sleep_sec > 3600:
                sleep_sec = SLEEP_CAP
            
            if DEBUG_MODE:
                print(f"[SLEEP] {symbol} | Next 5min check in {sleep_sec/60:.1f}m\n")
            
            time.sleep(sleep_sec)

        except ccxt.RateLimitExceeded:
            print(f"{LOG_PREFIX} {symbol} | Rate limit; sleep 10s")
            time.sleep(10)
        except Exception as e:
            err = f"{LOG_PREFIX} {symbol} | ERROR: {e}"
            print(err)
            send_telegram(err)
            traceback.print_exc()
            time.sleep(60)

def main():
    now_ist = get_ist_time()
    startup_msg = f"""
🚀 Guardeer Trading Bot - 5min/15min TESTING MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time: {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}
📊 Mode: {MODE.upper()}
💰 Capital per coin: ${PER_COIN_CAP_USD:,.2f}
🪙 Symbols: {', '.join(SYMBOLS)}
🎯 TIMEFRAMES: {ENTRY_TF} / {HTF} (HIGH FREQUENCY TESTING)

⚙️ OPTIMIZED FOR 5min/15min:
  • Risk per Trade: {RISK_PERCENT*100}% (lower for high frequency)
  • Risk:Reward: 1:{RR_FIXED} (achievable on short TF)
  • Max Trades/Day: {MAX_TRADES_PER_DAY}
  • Cooldown: {COOLDOWN_HOURS}h between trades
  • Max Drawdown: {MAX_DRAWDOWN*100}% (tighter)

🎯 ENHANCED FILTERS:
  • Volume Filter: Enabled (min {VOL_MIN_RATIO}x avg volume)
  • Candle Size: Min 0.3% body required
  • RSI Range: 25-75 (tighter momentum)
  • 15min Trend: Must be aligned

✅ CRITICAL FIXES APPLIED:
  ✓ Realistic 0.3% risk per trade
  ✓ Realistic 1:1.5 Risk:Reward
  ✓ REMOVED double-counted slippage deductions
  ✓ Enhanced quality filters for short timeframe
  ✓ Only actual fees deducted

⚠️  HIGH FREQUENCY WARNING:
  • More noise on 5min charts
  • Higher fee impact potential
  • Requires stricter filters
  • TEST CAREFULLY!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(startup_msg)
    send_telegram(startup_msg.replace('━', '-'))
    
    threads = []
    for sym in SYMBOLS:
        t = threading.Thread(target=worker, args=(sym,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)
    
    print(f"\n✅ {len(threads)} worker threads started for 5min/15min testing!\n")
    
    last_report_time = time.time()
    report_interval = 1800  # 30-minute reports for high frequency
    
    while True:
        time.sleep(60)
        
        if time.time() - last_report_time >= report_interval:
            try:
                now_ist = get_ist_time()
                report_lines = [f"\n📊 30-MIN REPORT - {now_ist.strftime('%I:%M %p IST')} (5min/15min)", "="*60]
                
                total_capital = 0
                total_trades_all = 0
                total_wins_all = 0
                total_losses_all = 0
                total_pnl_all = 0
                total_trades_today = 0
                
                for sym in SYMBOLS:
                    state_file, _ = state_files_for_symbol(sym)
                    if os.path.exists(state_file):
                        state = load_state(state_file)
                        total_capital += state['capital']
                        total_trades_all += state['total_trades']
                        total_wins_all += state['winning_trades']
                        total_losses_all += state['losing_trades']
                        total_pnl_all += state['total_pnl']
                        total_trades_today += state.get('trades_today', 0)
                        
                        win_rate = (state['winning_trades']/state['total_trades']*100) if state['total_trades'] > 0 else 0
                        roi = ((state['capital']/PER_COIN_CAP_USD - 1) * 100)
                        
                        report_lines.append(f"\n{sym}:")
                        report_lines.append(f"  Capital: ${state['capital']:.2f} ({roi:+.2f}%)")
                        report_lines.append(f"  Trades: {state['total_trades']} | {state['winning_trades']}W/{state['losing_trades']}L | WR: {win_rate:.1f}%")
                        report_lines.append(f"  PnL: ${state['total_pnl']:.2f} | Fees: ${state['total_fees_paid']:.2f}")
                        report_lines.append(f"  Today: {state.get('trades_today', 0)}/{MAX_TRADES_PER_DAY}")
                        report_lines.append(f"  Position: {'OPEN' if state['position'] == 1 else 'CLOSED'}")
                
                overall_win_rate = (total_wins_all/total_trades_all*100) if total_trades_all > 0 else 0
                overall_roi = ((total_capital/(PER_COIN_CAP_USD * len(SYMBOLS)) - 1) * 100)
                
                report_lines.append(f"\n{'='*60}")
                report_lines.append(f"PORTFOLIO TOTAL:")
                report_lines.append(f"  Total Capital: ${total_capital:.2f} ({overall_roi:+.2f}%)")
                report_lines.append(f"  Total Trades: {total_trades_all} | {total_wins_all}W/{total_losses_all}L")
                report_lines.append(f"  Win Rate: {overall_win_rate:.1f}%")
                report_lines.append(f"  Total PnL: ${total_pnl_all:.2f}")
                report_lines.append(f"  Trades Today: {total_trades_today}/{MAX_TRADES_PER_DAY * len(SYMBOLS)}")
                report_lines.append("="*60)
                
                report_msg = "\n".join(report_lines)
                print(report_msg)
                send_telegram(report_msg)
                
                last_report_time = time.time()
            except Exception as e:
                print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()

