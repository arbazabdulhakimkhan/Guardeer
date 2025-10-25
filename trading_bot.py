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

# CONFIG
MODE = os.getenv("MODE", "paper").lower()
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kucoin")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
ENTRY_TF = os.getenv("ENTRY_TF", "1h")
HTF = os.getenv("HTF", "4h")

TOTAL_PORTFOLIO_CAPITAL = float(os.getenv("TOTAL_PORTFOLIO_CAPITAL", "10000"))
PER_COIN_ALLOCATION = float(os.getenv("PER_COIN_ALLOCATION", "0.20"))
PER_COIN_CAP_USD = TOTAL_PORTFOLIO_CAPITAL * PER_COIN_ALLOCATION

RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
RR_FIXED = float(os.getenv("RR_FIXED", "5.0"))
DYNAMIC_RR = os.getenv("DYNAMIC_RR", "true").lower() == "true"
MIN_RR = float(os.getenv("MIN_RR", "4.0"))
MAX_RR = float(os.getenv("MAX_RR", "6.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "true").lower() == "true"
USE_H1_FILTER = os.getenv("USE_H1_FILTER", "true").lower() == "true"

# 🔧 FIX: Disable volume filter to match backtest
USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "false").lower() == "true"

VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "0.5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "25"))
BIAS_CONFIRM_BEAR = int(os.getenv("BIAS_CONFIRM_BEAR", "2"))

# 🔧 FIX: Add cooldown parameter (set to 0 to match backtest with no cooldown)
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "0.0"))

MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))
MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "100000"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
SLEEP_CAP = int(os.getenv("SLEEP_CAP", "60"))

DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

if MODE == "live":
    SLIPPAGE_RATE = 0.0
    FEE_RATE = 0.0

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
            print(f"[TELEGRAM] ✅ Sent: {msg[:50]}...")
        else:
            print(f"[TELEGRAM] ❌ Failed ({response.status_code}): {msg[:50]}...")
    except Exception as e:
        print(f"[TELEGRAM] ❌ Error: {e}")

def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")

def now_utc_naive():
    return datetime.utcnow().replace(tzinfo=None)

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
        "bearish_count": 0
    }

def save_state(state_file, state):
    s = dict(state)
    s["entry_time"] = state["entry_time"].isoformat() if state["entry_time"] is not None else None
    s["last_processed_ts"] = state["last_processed_ts"].isoformat() if state["last_processed_ts"] is not None else None
    s["last_exit_time"] = state["last_exit_time"].isoformat() if state.get("last_exit_time") is not None else None
    with open(state_file, "w") as f:
        json.dump(s, f, indent=2)

def append_trade(csv_file, row):
    write_header = not os.path.exists(csv_file)
    pd.DataFrame([row]).to_csv(csv_file, mode="a", header=write_header, index=False)

# Precision
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

# 🔧 FIXED: Strategy core per bar - NOW MATCHES BACKTEST 100%
def process_bar(symbol, entry_df, htf_df, state, exchange=None, market_info: MarketInfo=None):
    if len(entry_df) < 3:  # Need at least 3 bars
        if DEBUG_MODE:
            print(f"[PROCESS] {symbol} | Need at least 3 bars")
        return state, None
    
    # Use full dataframes for calculations
    h = htf_df.copy()

    # 🔧 FIX: Calculate Bias on FULL entry_df to match backtest
    entry_df_work = entry_df.copy()
    entry_df_work["Bias"] = 0
    entry_df_work.loc[entry_df_work["Close"] > entry_df_work["Close"].shift(1), "Bias"] = 1
    entry_df_work.loc[entry_df_work["Close"] < entry_df_work["Close"].shift(1), "Bias"] = -1

    h["Trend"] = 0
    h.loc[h["Close"] > h["Close"].shift(1), "Trend"] = 1
    h.loc[h["Close"] < h["Close"].shift(1), "Trend"] = -1
    
    entry_df_work["H4_Trend"] = h["Trend"].reindex(entry_df_work.index, method="ffill").fillna(0).astype(int)
    entry_df_work["ATR"] = calculate_atr(entry_df, ATR_PERIOD) if USE_ATR_STOPS else np.nan
    
    if USE_VOLUME_FILTER:
        entry_df_work["Avg_Volume"] = entry_df["Volume"].rolling(VOL_LOOKBACK).mean()
    
    entry_df_work["RSI"] = calculate_rsi(entry_df["Close"], RSI_PERIOD)

    # 🔧 FIX: Process the PREVIOUS bar (i-1), enter at current bar (i)
    # This matches backtest exactly
    closed_bar = entry_df_work.iloc[-2]  # Previous closed bar
    ts = entry_df_work.index[-2]
    prev_close = float(closed_bar["Close"])
    prev_open = float(closed_bar["Open"])
    
    # Get the bar before previous for sweep check
    if len(entry_df_work) >= 3:
        prev_prev_close = float(entry_df_work['Close'].iloc[-3])
    else:
        prev_prev_close = prev_open
    
    bias = int(closed_bar["Bias"])
    h4_trend = int(closed_bar["H4_Trend"])

    # Current price for exits (from most recent bar)
    current_price = float(entry_df_work['Close'].iloc[-1])

    if DEBUG_MODE:
        ts_ist = utc_to_ist(ts)
        print(f"\n{'='*60}")
        print(f"[DEBUG] {symbol} {ts_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')} | Processing CLOSED candle")
        print(f"[DEBUG] {symbol} | Prev Bar OHLC: O={prev_open:.2f} H={closed_bar['High']:.2f} L={closed_bar['Low']:.2f} C={prev_close:.2f}")
        print(f"[DEBUG] {symbol} | Current Price: {current_price:.2f}")
        print(f"[DEBUG] {symbol} | Volume: {closed_bar['Volume']:.0f}")
        rsi_val = closed_bar['RSI']
        print(f"[DEBUG] {symbol} | RSI: {rsi_val:.1f} | Bias: {bias} | H4_Trend: {h4_trend}")
        print(f"[DEBUG] {symbol} | Position: {state['position']} | Capital: ${state['capital']:.2f}")
        print(f"{'='*60}\n")

    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    curr_dd = (state["peak_equity"] - state["capital"]) / state["peak_equity"] if state["peak_equity"] > 0 else 0.0
    blocked = curr_dd >= MAX_DRAWDOWN

    trade_row = None

    # Permanent stop logic
    if blocked and not state.get("permanently_stopped", False):
        state["permanently_stopped"] = True
        
        if state["position"] == 1:
            if MODE == "live":
                try:
                    base_qty = state["entry_size"]
                    order = place_market_sell(exchange, market_info, base_qty)
                    fill_px = avg_fill_price_from_order(order) or current_price
                    exit_price = float(fill_px)
                except Exception as e:
                    send_telegram(f"{symbol} FORCED EXIT error: {e}")
                    raise
            else:
                exit_price = current_price
                
            pnl = state["entry_size"] * (exit_price - state["entry_price"])
            pnl -= state["entry_size"] * SLIPPAGE_RATE
            pnl -= (exit_price * state["entry_size"]) * FEE_RATE
            state["capital"] += pnl
            
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
                "PnL_$": round(pnl, 2),
                "Win": 1 if pnl > 0 else 0,
                "Exit_Reason": "MAX DRAWDOWN - PERMANENT STOP",
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }
            
            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts
            
            msg = f"{LOG_PREFIX} {symbol} PERMANENTLY STOPPED | Cap={state['capital']:.2f}"
            print(msg)
            send_telegram(f"🛑 {symbol} PERMANENTLY STOPPED - Max Drawdown!")
            
            return state, trade_row

    blocked = state.get("permanently_stopped", False)

    # Exits (using current_price)
    if state["position"] == 1 and not blocked:
        if DEBUG_MODE:
            entry_ist = utc_to_ist(state['entry_time'])
            print(f"[EXIT CHECK] {symbol} | In position since {entry_ist.strftime('%I:%M %p IST')} | Entry: ${state['entry_price']:.2f}")
            
        exit_flag = False
        exit_price = current_price
        exit_reason = ""

        if current_price >= state["entry_tp"]:
            exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
            state["bearish_count"] = 0
        elif current_price <= state["entry_sl"]:
            exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
            state["bearish_count"] = 0
        elif USE_H1_FILTER and h4_trend < 0:
            exit_flag, exit_price, exit_reason = True, current_price, "4H Trend Reversal"
            state["bearish_count"] = 0
        elif bias < 0:
            state["bearish_count"] += 1
            if DEBUG_MODE:
                print(f"[EXIT CHECK] {symbol} | Bearish count: {state['bearish_count']}/{BIAS_CONFIRM_BEAR}")
            if state["bearish_count"] >= BIAS_CONFIRM_BEAR:
                exit_flag, exit_price, exit_reason = True, current_price, "Bias Reversal"
                state["bearish_count"] = 0
        else:
            state["bearish_count"] = 0

        if exit_flag:
            if MODE == "live":
                try:
                    base_qty = state["entry_size"]
                    order = place_market_sell(exchange, market_info, base_qty)
                    fill_px = avg_fill_price_from_order(order) or current_price
                    exit_price = float(fill_px)
                except Exception as e:
                    send_telegram(f"{symbol} Exit SELL error: {e}")
                    raise

            pnl = state["entry_size"] * (exit_price - state["entry_price"])
            pnl -= state["entry_size"] * SLIPPAGE_RATE
            pnl -= (exit_price * state["entry_size"]) * FEE_RATE
            state["capital"] += pnl

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
                "PnL_$": round(pnl, 2),
                "Win": 1 if pnl > 0 else 0,
                "Exit_Reason": exit_reason,
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }

            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts

            ts_ist = utc_to_ist(ts)
            msg = f"{LOG_PREFIX} {symbol} {ts_ist.strftime('%I:%M %p IST')} EXIT {exit_reason} @ {exit_price:.4f} | PnL={pnl:.2f} | Cap={state['capital']:.2f}"
            print(msg)
            send_telegram(msg)

    # 🔧 FIXED: Entry logic - NOW MATCHES BACKTEST EXACTLY
    if state["position"] == 0 and not blocked:
        # 🔧 FIX: Cooldown check (now configurable, default 0 to match backtest)
        if COOLDOWN_HOURS > 0 and state.get("last_exit_time") is not None:
            time_diff_hours = (ts - state["last_exit_time"]).total_seconds() / 3600
            if time_diff_hours < COOLDOWN_HOURS:
                if DEBUG_MODE:
                    print(f"🚫 [COOLDOWN] {symbol} | Blocked: Only {time_diff_hours:.1f}h since last exit (need {COOLDOWN_HOURS}h)")
                state["last_processed_ts"] = ts
                return state, None
        
        # 🔧 FIX: Bullish sweep - EXACTLY matches backtest
        # Checks: prev_close > prev_open AND prev_close > prev_prev_close
        bullish_sweep = (prev_close > prev_open) and (prev_close > prev_prev_close)
        
        # Volume filter (disabled by default to match backtest)
        vol_ok = True
        if USE_VOLUME_FILTER and not np.isnan(closed_bar.get("Avg_Volume", np.nan)):
            vol_ok = closed_bar["Volume"] >= VOL_MIN_RATIO * closed_bar["Avg_Volume"]
        
        rsi_ok = True if np.isnan(closed_bar["RSI"]) else closed_bar["RSI"] > RSI_OVERSOLD
        h4_ok = (not USE_H1_FILTER) or (h4_trend == 1)

        if DEBUG_MODE:
            print(f"[ENTRY CHECK] {symbol} | Looking for entry...")
            print(f"[ENTRY CHECK] {symbol} | Prev Close > Prev Open: {prev_close:.2f} > {prev_open:.2f} = {prev_close > prev_open}")
            print(f"[ENTRY CHECK] {symbol} | Prev Close > Prev-Prev Close: {prev_close:.2f} > {prev_prev_close:.2f} = {prev_close > prev_prev_close}")
            print(f"[ENTRY CHECK] {symbol} | Bullish Sweep: {bullish_sweep}")
            print(f"[ENTRY CHECK] {symbol} | Bias: {bias} (need 1)")
            print(f"[ENTRY CHECK] {symbol} | Volume OK: {vol_ok}")
            print(f"[ENTRY CHECK] {symbol} | RSI OK: {rsi_ok} (RSI: {closed_bar['RSI']:.1f} > {RSI_OVERSOLD})")
            print(f"[ENTRY CHECK] {symbol} | H4 Trend OK: {h4_ok} (Trend: {h4_trend})")

        # Entry conditions - matches backtest
        if bias == 1 and bullish_sweep and vol_ok and rsi_ok and h4_ok:
            ts_ist = utc_to_ist(ts)
            print(f"🟢 [ENTRY TRIGGERED] {symbol} at {ts_ist.strftime('%I:%M %p IST')} | All conditions met!")
            send_telegram(f"🟢 ENTRY SIGNAL: {symbol} at ${prev_close:.2f}")
            
            # Calculate SL/TP based on previous bar
            if USE_ATR_STOPS:
                atr_val = float(closed_bar["ATR"])
                if np.isnan(atr_val) or atr_val <= 0:
                    if DEBUG_MODE:
                        print(f"[ENTRY] {symbol} | ATR invalid: {atr_val}")
                    state["last_processed_ts"] = ts
                    return state, trade_row
                sl = prev_close - (ATR_MULT_SL * atr_val)
            else:
                sweep_buffer = min(max(prev_close * 0.0005, 0.0005), 0.0015)
                sl = prev_close * (1 - sweep_buffer)

            risk = abs(prev_close - sl)
            if risk <= 0:
                if DEBUG_MODE:
                    print(f"[ENTRY] {symbol} | Risk invalid: {risk}")
                state["last_processed_ts"] = ts
                return state, trade_row

            rr_ratio = RR_FIXED
            if DYNAMIC_RR and USE_ATR_STOPS:
                atr_series = calculate_atr(entry_df, ATR_PERIOD)
                if len(atr_series) >= 7:
                    recent_atr = float(atr_series.iloc[-7:-2].mean())
                    curr_atr = float(closed_bar["ATR"])
                    if not np.isnan(recent_atr) and recent_atr > 0:
                        if curr_atr > recent_atr * 1.2: rr_ratio = MIN_RR
                        elif curr_atr < recent_atr * 0.8: rr_ratio = MAX_RR
            
            tp = prev_close + rr_ratio * risk

            per_coin_cap = PER_COIN_CAP_USD
            available_cap = min(state["capital"], per_coin_cap)
            size_base = (available_cap * RISK_PERCENT) / risk
            size_base = min(size_base, MAX_TRADE_SIZE / prev_close)
            size_base = min(size_base, per_coin_cap / prev_close)

            if DEBUG_MODE:
                print(f"[ENTRY] {symbol} | Setup Price: ${prev_close:.2f}")
                print(f"[ENTRY] {symbol} | Risk: ${risk:.2f} | RR: {rr_ratio} | SL: ${sl:.2f} | TP: ${tp:.2f}")
                print(f"[ENTRY] {symbol} | Size: {size_base:.6f}")

            if size_base > 0:
                # 🔧 FIX: Enter at next bar's OPEN (simulated with prev_close)
                # In live, this will be filled at market on next bar
                entry_price_used = prev_close
                
                if MODE == "live":
                    try:
                        mi = market_info
                        size_base = max(size_base, mi.amount_min)
                        size_base = mi.round_amount(size_base)
                        order = place_market_buy(exchange, mi, size_base)
                        fill_px = avg_fill_price_from_order(order) or prev_close
                        entry_price_used = float(fill_px)
                    except Exception as e:
                        send_telegram(f"{symbol} Entry BUY error: {e}")
                        raise

                state["position"] = 1
                state["entry_price"] = entry_price_used
                state["entry_sl"] = sl
                state["entry_tp"] = tp
                state["entry_time"] = ts
                state["entry_size"] = size_base
                state["bearish_count"] = 0

                state["capital"] -= (size_base * SLIPPAGE_RATE)
                state["capital"] -= (entry_price_used * size_base * FEE_RATE)

                ts_ist = utc_to_ist(ts)
                msg = f"{LOG_PREFIX} {symbol} {ts_ist.strftime('%I:%M %p IST')} ENTRY Long @ {entry_price_used:.4f} | SL={sl:.4f} TP={tp:.4f} RR={rr_ratio:.2f} Size={size_base:.6f} Cap={state['capital']:.2f} Mode={MODE}"
                print(msg)
                send_telegram(msg)
        else:
            if DEBUG_MODE:
                missing = []
                if bias != 1: missing.append(f"Bias({bias}≠1)")
                if not bullish_sweep: missing.append("BullishSweep")
                if not vol_ok: missing.append("Volume")
                if not rsi_ok: missing.append("RSI")
                if not h4_ok: missing.append("H4Trend")
                print(f"❌ [NO ENTRY] {symbol} | Missing: {', '.join(missing)}")

    state["last_processed_ts"] = ts
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    return state, trade_row

# Worker per symbol
def worker(symbol):
    state_file, trades_csv = state_files_for_symbol(symbol)
    exchange = get_exchange()
    market_info = MarketInfo(exchange, symbol) if MODE == "live" else None
    state = load_state(state_file)
    tf_minutes = timeframe_to_minutes(ENTRY_TF)

    print(f"{LOG_PREFIX} Start | {symbol} | TF={ENTRY_TF}/{HTF} | Mode={MODE} | Capital={state['capital']:.2f} | Cooldown={COOLDOWN_HOURS}h")
    send_telegram(f"🤖 Started {symbol} {ENTRY_TF}/{HTF} Mode={MODE} Cap=${PER_COIN_CAP_USD} Cooldown={COOLDOWN_HOURS}h")

    while True:
        try:
            now_ist = get_ist_time()
            now_utc = datetime.now(pytz.utc)
            
            if DEBUG_MODE:
                print(f"[TIME] {symbol} | IST: {now_ist.strftime('%I:%M:%S %p')} | UTC: {now_utc.strftime('%H:%M:%S')}")
            
            entry_df = fetch_ohlcv_df(exchange, symbol, ENTRY_TF, limit=500)
            htf_df = fetch_ohlcv_df(exchange, symbol, HTF, limit=600)

            if entry_df.empty or htf_df.empty or len(entry_df) < 3:
                print(f"{LOG_PREFIX} {symbol} No data; wait 30s")
                time.sleep(30)
                continue

            closed_candle_ts = entry_df.index[-2]
            forming_candle_ts = entry_df.index[-1]

            closed_candle_ist = utc_to_ist(closed_candle_ts)
            forming_candle_ist = utc_to_ist(forming_candle_ts)

            if DEBUG_MODE:
                print(f"[DATA] {symbol} | Closed candle: {closed_candle_ist.strftime('%Y-%m-%d %I:%M %p IST')}")
                print(f"[DATA] {symbol} | Forming candle: {forming_candle_ist.strftime('%Y-%m-%d %I:%M %p IST')}")
                if state['last_processed_ts']:
                    last_proc_ist = utc_to_ist(state['last_processed_ts'])
                    print(f"[DATA] {symbol} | Last processed: {last_proc_ist.strftime('%Y-%m-%d %I:%M %p IST')}")

            if state["last_processed_ts"] is None:
                state["last_processed_ts"] = closed_candle_ts
                save_state(state_file, state)
                if DEBUG_MODE:
                    print(f"[INIT] {symbol} | Initial: {closed_candle_ist.strftime('%I:%M %p IST')}")
                time.sleep(10)
                continue

            if closed_candle_ts > state["last_processed_ts"]:
                if DEBUG_MODE:
                    prev_ist = utc_to_ist(state['last_processed_ts'])
                    print(f"[WORKER] {symbol} | 🆕 NEW CLOSED CANDLE: {closed_candle_ist.strftime('%I:%M %p IST')} (prev: {prev_ist.strftime('%I:%M %p IST')})")
                
                state, trade = process_bar(symbol, entry_df, htf_df, state, exchange=exchange, market_info=market_info)
                
                if trade is not None:
                    append_trade(trades_csv, trade)
                    entry_time_ist = format_ist_time(pd.to_datetime(trade['Entry_DateTime']))
                    print(f"✅ [TRADE LOGGED] {symbol} | Entry: {entry_time_ist}")
                
                save_state(state_file, state)
                last_proc_ist = utc_to_ist(state['last_processed_ts'])
                print(f"💾 [STATE SAVED] {symbol} | Last processed: {last_proc_ist.strftime('%I:%M %p IST')}")
            else:
                if DEBUG_MODE and symbol == "BTC/USDT":
                    print(f"[WORKER] {symbol} | ⏳ Already processed")

            next_close_utc = forming_candle_ts + timedelta(minutes=tf_minutes)
            safe_check_time_utc = next_close_utc + timedelta(minutes=1)
            
            next_close_ist = utc_to_ist(next_close_utc)
            safe_check_ist = utc_to_ist(safe_check_time_utc)
            
            sleep_sec = (safe_check_time_utc - now_utc.replace(tzinfo=None)).total_seconds()
            
            if sleep_sec < 10:
                sleep_sec = 10
            elif sleep_sec > 3600:
                sleep_sec = SLEEP_CAP
            
            if DEBUG_MODE:
                print(f"[SLEEP] {symbol} | Next closes: {next_close_ist.strftime('%I:%M %p IST')}")
                print(f"[SLEEP] {symbol} | Will check: {safe_check_ist.strftime('%I:%M %p IST')} (in {sleep_sec:.0f}s)")
            
            time.sleep(sleep_sec)

        except ccxt.RateLimitExceeded:
            print(f"{LOG_PREFIX} {symbol} Rate limit; sleep 10s")
            time.sleep(10)
        except Exception as e:
            err = f"{LOG_PREFIX} {symbol} ERROR: {e}"
            print(err)
            send_telegram(err)
            traceback.print_exc()
            time.sleep(60)

# Main
def main():
    now_ist = get_ist_time()
    startup_msg = f"🚀 Guardeer Bot Started!\nTime: {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}\nMode: {MODE}\nCoins: {', '.join(SYMBOLS)}\nCap/coin: ${PER_COIN_CAP_USD}\nCooldown: {COOLDOWN_HOURS}h\nDebug: {DEBUG_MODE}"
    print(startup_msg)
    send_telegram(startup_msg)
    
    threads = []
    for sym in SYMBOLS:
        t = threading.Thread(target=worker, args=(sym,), daemon=True)
        t.start()
        threads.append(t)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
