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

USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "false").lower() == "true"
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MIN_RATIO = float(os.getenv("VOL_MIN_RATIO", "0.5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "25"))
BIAS_CONFIRM_BEAR = int(os.getenv("BIAS_CONFIRM_BEAR", "2"))

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
        
        # ✅ NEW: Load trade statistics
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
        if "total_slippage_paid" not in s:
            s["total_slippage_paid"] = 0.0
        
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
        "bearish_count": 0,
        # ✅ NEW: Track statistics
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "total_fees_paid": 0.0,
        "total_slippage_paid": 0.0
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

# ✅ ENHANCED: Now tracks detailed fees, slippage, and statistics
def process_bar(symbol, entry_df, htf_df, state, exchange=None, market_info: MarketInfo=None):
    """
    🔥 ENHANCED: Detailed capital tracking with fees & slippage breakdown
    """
    if len(entry_df) < 3:
        return state, None
    
    entry_df_work = entry_df.copy()
    entry_df_work["Bias"] = 0
    entry_df_work.loc[entry_df_work["Close"] > entry_df_work["Close"].shift(1), "Bias"] = 1
    entry_df_work.loc[entry_df_work["Close"] < entry_df_work["Close"].shift(1), "Bias"] = -1
    
    h = htf_df.copy()
    h["Trend"] = 0
    h.loc[h["Close"] > h["Close"].shift(1), "Trend"] = 1
    h.loc[h["Close"] < h["Close"].shift(1), "Trend"] = -1
    
    entry_df_work["H4_Trend"] = h["Trend"].reindex(entry_df_work.index, method="ffill").fillna(0).astype(int)
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
    h4_trend = int(current_bar["H4_Trend"])
    
    if i >= 1:
        prev_close = float(entry_df_work['Close'].iloc[i-1])
        bullish_sweep = (price > open_price) and (price > prev_close)
    else:
        bullish_sweep = False
    
    if DEBUG_MODE:
        ts_ist = utc_to_ist(ts)
        print(f"\n{'='*80}")
        print(f"[DEBUG] {symbol} | Bar[{i}] @ {ts_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"[DEBUG] OHLC: O={open_price:.4f} H={current_bar['High']:.4f} L={current_bar['Low']:.4f} C={price:.4f}")
        if i >= 1:
            print(f"[DEBUG] Prev Close: {prev_close:.4f} | Sweep: {bullish_sweep}")
        print(f"[DEBUG] RSI: {current_bar['RSI']:.1f} | Bias: {bias} | H4: {h4_trend}")
        print(f"[DEBUG] Position: {state['position']} | Cap: ${state['capital']:.2f}")
        print(f"[DEBUG] Stats: {state['winning_trades']}W / {state['losing_trades']}L | Total PnL: ${state['total_pnl']:.2f}")
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
            
            # ✅ FIXED: Standard trading cost calculation (position value based)
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            
            # Exit costs (calculated on position value at exit)
            position_value_at_exit = exit_price * state["entry_size"]
            exit_slippage = position_value_at_exit * SLIPPAGE_RATE
            exit_fee = position_value_at_exit * FEE_RATE
            
            # Entry costs (for reporting only - already deducted at entry)
            position_value_at_entry = state["entry_price"] * state["entry_size"]
            entry_slippage = position_value_at_entry * SLIPPAGE_RATE
            entry_fee = position_value_at_entry * FEE_RATE
            
            # Net PnL = Gross PnL - EXIT costs only
            total_fees = entry_fee + exit_fee  # For reporting
            total_slippage = entry_slippage + exit_slippage  # For reporting
            net_pnl = gross_pnl - exit_slippage - exit_fee  # Only deduct exit costs!
            
            state["capital"] += net_pnl
            state["total_trades"] += 1
            state["total_pnl"] += net_pnl
            state["total_fees_paid"] += total_fees
            state["total_slippage_paid"] += total_slippage
            
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
                "Entry_Slippage_$": round(entry_slippage, 2),
                "Exit_Slippage_$": round(exit_slippage, 2),
                "Total_Slippage_$": round(total_slippage, 2),
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
    
    # Exit logic with detailed tracking
    if state["position"] == 1 and not blocked:
        exit_flag = False
        exit_price = price
        exit_reason = ""
        
        if price >= state["entry_tp"]:
            exit_flag, exit_price, exit_reason = True, state["entry_tp"], "Take Profit"
            state["bearish_count"] = 0
        elif price <= state["entry_sl"]:
            exit_flag, exit_price, exit_reason = True, state["entry_sl"], "Stop Loss"
            state["bearish_count"] = 0
        elif USE_H1_FILTER and h4_trend < 0:
            exit_flag, exit_price, exit_reason = True, price, "4H Trend Reversal"
            state["bearish_count"] = 0
        elif bias < 0:
            state["bearish_count"] += 1
            if state["bearish_count"] >= BIAS_CONFIRM_BEAR:
                exit_flag, exit_price, exit_reason = True, price, "Bias Reversal"
                state["bearish_count"] = 0
        else:
            state["bearish_count"] = 0
        
        if exit_flag:
            if MODE == "live":
                try:
                    order = place_market_sell(exchange, market_info, state["entry_size"])
                    exit_price = float(avg_fill_price_from_order(order) or price)
                except Exception as e:
                    send_telegram(f"❌ {symbol} Exit error: {e}")
                    raise
            
            # ✅ FIXED: Standard trading cost calculation (position value based)
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            
            # Exit costs (calculated on position value at exit)
            position_value_at_exit = exit_price * state["entry_size"]
            exit_slippage = position_value_at_exit * SLIPPAGE_RATE
            exit_fee = position_value_at_exit * FEE_RATE
            
            # Entry costs (for reporting only - already deducted at entry)
            position_value_at_entry = state["entry_price"] * state["entry_size"]
            entry_slippage = position_value_at_entry * SLIPPAGE_RATE
            entry_fee = position_value_at_entry * FEE_RATE
            
            # Net PnL = Gross PnL - EXIT costs only
            total_fees = entry_fee + exit_fee  # For reporting
            total_slippage = entry_slippage + exit_slippage  # For reporting
            net_pnl = gross_pnl - exit_slippage - exit_fee  # Only deduct exit costs!
            
            state["capital"] += net_pnl
            state["total_trades"] += 1
            state["total_pnl"] += net_pnl
            state["total_fees_paid"] += total_fees
            state["total_slippage_paid"] += total_slippage
            
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
                "Entry_Slippage_$": round(entry_slippage, 2),
                "Exit_Slippage_$": round(exit_slippage, 2),
                "Total_Slippage_$": round(total_slippage, 2),
                "Net_PnL_$": round(net_pnl, 2),
                "Win": 1 if net_pnl > 0 else 0,
                "Exit_Reason": exit_reason,
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }
            
            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts
            
            # ✅ ENHANCED: Detailed exit notification
            pnl_emoji = "💚" if net_pnl > 0 else "❤️"
            exit_msg = f"""
{pnl_emoji} EXIT {symbol} | {exit_reason}
━━━━━━━━━━━━━━━━━━━━━━━
📍 Exit Price: ${exit_price:.4f}
💰 Gross PnL: ${gross_pnl:.2f}
💸 Fees: ${total_fees:.2f} ({FEE_RATE*100}%)
📉 Slippage: ${total_slippage:.2f}
💵 Net PnL: ${net_pnl:.2f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Capital: ${state['capital']:.2f}
📊 Stats: {state['winning_trades']}W / {state['losing_trades']}L
📈 Win Rate: {win_rate:.1f}%
💎 Total PnL: ${state['total_pnl']:.2f}
"""
            print(exit_msg)
            send_telegram(exit_msg)
    
    # Entry logic with detailed tracking
    if state["position"] == 0 and not blocked:
        if COOLDOWN_HOURS > 0 and state.get("last_exit_time") is not None:
            time_diff_hours = (ts - state["last_exit_time"]).total_seconds() / 3600
            if time_diff_hours < COOLDOWN_HOURS:
                state["last_processed_ts"] = ts
                return state, None
        
        vol_ok = True
        if USE_VOLUME_FILTER and not np.isnan(current_bar.get("Avg_Volume", np.nan)):
            vol_ok = current_bar["Volume"] >= VOL_MIN_RATIO * current_bar["Avg_Volume"]
        
        rsi_ok = True if np.isnan(current_bar["RSI"]) else current_bar["RSI"] > RSI_OVERSOLD
        h4_ok = (not USE_H1_FILTER) or (h4_trend == 1)
        
        if DEBUG_MODE:
            print(f"[ENTRY CHECK] {symbol}")
            print(f"  Bias==1: {bias == 1}")
            print(f"  Bullish Sweep: {bullish_sweep}")
            print(f"  Volume: {vol_ok}")
            print(f"  RSI: {rsi_ok} ({current_bar['RSI']:.1f})")
            print(f"  H4: {h4_ok}")
        
        if bias == 1 and bullish_sweep and vol_ok and rsi_ok and h4_ok:
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
            if DYNAMIC_RR and USE_ATR_STOPS and not np.isnan(current_bar["ATR"]):
                if len(entry_df_work) >= 6:
                    recent_atr = float(entry_df_work['ATR'].iloc[-6:-1].mean())
                    current_atr = float(current_bar["ATR"])
                    if recent_atr > 0:
                        if current_atr > recent_atr * 1.2:
                            rr_ratio = MIN_RR
                        elif current_atr < recent_atr * 0.8:
                            rr_ratio = MAX_RR
            
            tp = price + rr_ratio * risk
            size_base = min(state["capital"] * RISK_PERCENT / risk if risk > 0 else 0, MAX_TRADE_SIZE)
            
            if DEBUG_MODE:
                print(f"[ENTRY] Setup: Entry=${price:.4f} SL=${sl:.4f} TP=${tp:.4f} RR={rr_ratio:.1f} Size={size_base:.6f}")
            
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
                
                state["position"] = 1
                state["entry_price"] = entry_price_used
                state["entry_sl"] = sl
                state["entry_tp"] = tp
                state["entry_time"] = ts
                state["entry_size"] = size_base
                state["bearish_count"] = 0
                
                # ✅ FIXED: Standard trading cost calculation (position value based)
                position_value = entry_price_used * size_base
                entry_slippage = position_value * SLIPPAGE_RATE
                entry_fee = position_value * FEE_RATE
                entry_costs = entry_slippage + entry_fee
                
                state["capital"] -= entry_costs
                
                position_value = entry_price_used * size_base
                risk_amount = state["capital"] * RISK_PERCENT
                
                # ✅ ENHANCED: Detailed entry notification
                entry_msg = f"""
🚀 ENTRY {symbol}
━━━━━━━━━━━━━━━━━━━━━━━
📍 Entry Price: ${entry_price_used:.4f}
🎯 Stop Loss: ${sl:.4f}
🎯 Take Profit: ${tp:.4f}
📊 Risk:Reward: 1:{rr_ratio:.1f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Position Size: {size_base:.6f} {symbol.split('/')[0]}
💰 Position Value: ${position_value:.2f}
💸 Entry Fee: ${entry_fee:.2f}
📉 Entry Slippage: ${entry_slippage:.2f}
💵 Total Entry Cost: ${entry_costs:.2f}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Capital Before: ${state['capital'] + entry_costs:.2f}
💼 Capital After: ${state['capital']:.2f}
💎 Risking: ${risk_amount:.2f} ({RISK_PERCENT*100}%)
━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Trades: {state['total_trades']}
📈 Record: {state['winning_trades']}W / {state['losing_trades']}L
💰 Total PnL: ${state['total_pnl']:.2f}
"""
                print(entry_msg)
                send_telegram(entry_msg)
        else:
            if DEBUG_MODE:
                missing = []
                if bias != 1: missing.append(f"Bias({bias})")
                if not bullish_sweep: missing.append("Sweep")
                if not vol_ok: missing.append("Vol")
                if not rsi_ok: missing.append("RSI")
                if not h4_ok: missing.append("H4")
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

    # ✅ ENHANCED: Show initial statistics
    initial_msg = f"""
🤖 {symbol} Worker Started
━━━━━━━━━━━━━━━━━━━━━━━
💼 Starting Capital: ${state['capital']:.2f}
📊 Previous Stats:
  • Total Trades: {state['total_trades']}
  • Wins: {state['winning_trades']}
  • Losses: {state['losing_trades']}
  • Total PnL: ${state['total_pnl']:.2f}
  • Fees Paid: ${state['total_fees_paid']:.2f}
  • Slippage: ${state['total_slippage_paid']:.2f}
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
                    print(f"\n[WORKER] {symbol} | 🆕 NEW CANDLE @ {utc_to_ist(closed_candle_ts).strftime('%I:%M %p')}")
                
                state, trade = process_bar(symbol, entry_df, htf_df, state, exchange=exchange, market_info=market_info)
                
                if trade is not None:
                    append_trade(trades_csv, trade)
                    
                    # ✅ ENHANCED: Show detailed trade log
                    trade_summary = f"""
{'='*80}
📋 TRADE #{state['total_trades']} COMPLETED - {symbol}
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
  Entry Slippage: -${trade['Entry_Slippage_$']:.2f}
  Exit Slippage:  -${trade['Exit_Slippage_$']:.2f}
  ─────────────────────────────
  Net PnL:         ${trade['Net_PnL_$']:.2f} {'✅ WIN' if trade['Win'] else '❌ LOSS'}

📊 STATISTICS ({symbol}):
  Total Trades: {state['total_trades']}
  Wins: {state['winning_trades']} | Losses: {state['losing_trades']}
  Win Rate: {(state['winning_trades']/state['total_trades']*100) if state['total_trades'] > 0 else 0:.1f}%
  Total PnL: ${state['total_pnl']:.2f}
  Total Fees Paid: ${state['total_fees_paid']:.2f}
  Total Slippage: ${state['total_slippage_paid']:.2f}

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
                print(f"[SLEEP] {symbol} | Next check in {sleep_sec/60:.1f}m\n")
            
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
🚀 Guardeer Trading Bot Started! (Enhanced Tracking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time: {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}
📊 Mode: {MODE.upper()}
💰 Capital per coin: ${PER_COIN_CAP_USD:,.2f}
🪙 Symbols: {', '.join(SYMBOLS)}
📈 Timeframes: {ENTRY_TF} / {HTF}

⚙️ Strategy Settings:
  • Risk per Trade: {RISK_PERCENT*100}%
  • Risk:Reward: {RR_FIXED}x (Dynamic: {MIN_RR}-{MAX_RR}x)
  • Max Drawdown: {MAX_DRAWDOWN*100}%
  • ATR Stops: {USE_ATR_STOPS}
  • H4 Filter: {USE_H1_FILTER}
  • Volume Filter: {USE_VOLUME_FILTER}
  • Cooldown: {COOLDOWN_HOURS}h

💸 Cost Settings:
  • Slippage Rate: {SLIPPAGE_RATE*100}%
  • Fee Rate: {FEE_RATE*100}%

✅ Enhanced Features:
  ✓ Detailed fee & slippage tracking
  ✓ Per-coin statistics (W/L, PnL)
  ✓ Capital breakdown on entry/exit
  ✓ Real-time performance metrics
  ✓ Complete trade history in CSV

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
    
    print(f"\n✅ {len(threads)} worker threads started!\n")
    
    # ✅ NEW: Periodic statistics reporting
    last_report_time = time.time()
    report_interval = 3600  # Report every hour
    
    while True:
        time.sleep(60)
        
        # Generate hourly report
        if time.time() - last_report_time >= report_interval:
            try:
                now_ist = get_ist_time()
                report_lines = [f"\n📊 HOURLY REPORT - {now_ist.strftime('%I:%M %p IST')}", "="*60]
                
                total_capital = 0
                total_trades_all = 0
                total_wins_all = 0
                total_losses_all = 0
                total_pnl_all = 0
                
                for sym in SYMBOLS:
                    state_file, _ = state_files_for_symbol(sym)
                    if os.path.exists(state_file):
                        state = load_state(state_file)
                        total_capital += state['capital']
                        total_trades_all += state['total_trades']
                        total_wins_all += state['winning_trades']
                        total_losses_all += state['losing_trades']
                        total_pnl_all += state['total_pnl']
                        
                        win_rate = (state['winning_trades']/state['total_trades']*100) if state['total_trades'] > 0 else 0
                        roi = ((state['capital']/PER_COIN_CAP_USD - 1) * 100)
                        
                        report_lines.append(f"\n{sym}:")
                        report_lines.append(f"  Capital: ${state['capital']:.2f} ({roi:+.2f}%)")
                        report_lines.append(f"  Trades: {state['total_trades']} | {state['winning_trades']}W/{state['losing_trades']}L | WR: {win_rate:.1f}%")
                        report_lines.append(f"  PnL: ${state['total_pnl']:.2f} | Fees: ${state['total_fees_paid']:.2f}")
                        report_lines.append(f"  Position: {'OPEN' if state['position'] == 1 else 'CLOSED'}")
                
                overall_win_rate = (total_wins_all/total_trades_all*100) if total_trades_all > 0 else 0
                overall_roi = ((total_capital/(PER_COIN_CAP_USD * len(SYMBOLS)) - 1) * 100)
                
                report_lines.append(f"\n{'='*60}")
                report_lines.append(f"PORTFOLIO TOTAL:")
                report_lines.append(f"  Total Capital: ${total_capital:.2f} ({overall_roi:+.2f}%)")
                report_lines.append(f"  Total Trades: {total_trades_all} | {total_wins_all}W/{total_losses_all}L")
                report_lines.append(f"  Win Rate: {overall_win_rate:.1f}%")
                report_lines.append(f"  Total PnL: ${total_pnl_all:.2f}")
                report_lines.append("="*60)
                
                report_msg = "\n".join(report_lines)
                print(report_msg)
                send_telegram(report_msg)
                
                last_report_time = time.time()
            except Exception as e:
                print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
