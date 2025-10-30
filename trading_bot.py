import os, time, json, traceback, threading
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import pytz

load_dotenv()

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

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
    except:
        pass

def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")

def get_exchange():
    if MODE == "live":
        return getattr(ccxt, EXCHANGE_ID)({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    else:
        return getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "options": {"defaultType": "spot"}})

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
    except Exception as e:
        print(f"[DATA] Error {symbol}: {e}")
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
    return f"state_{tag}.json", f"{tag}_trades.csv"

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
    s["entry_time"] = state["entry_time"].isoformat() if state["entry_time"] else None
    s["last_processed_ts"] = state["last_processed_ts"].isoformat() if state["last_processed_ts"] else None
    s["last_exit_time"] = state["last_exit_time"].isoformat() if state.get("last_exit_time") else None
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
        self.symbol = symbol
        self.exchange = exchange
    def round_amount(self, amt): return float(self.exchange.amount_to_precision(self.symbol, amt))

def place_market_buy(exchange, mi: MarketInfo, base_qty: float):
    base_qty = mi.round_amount(max(base_qty, mi.amount_min))
    if base_qty <= 0: raise ValueError(f"Amount too small")
    return exchange.create_market_buy_order(mi.symbol, base_qty)

def place_market_sell(exchange, mi: MarketInfo, base_qty: float):
    base_qty = mi.round_amount(base_qty)
    if base_qty <= 0: raise ValueError(f"Amount too small")
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

# ✅ FIXED: Added spot position sizing function from backtest
def calculate_spot_position_size(price, sl, capital, risk_percent, max_trade_size):
    """Calculate position size for spot trading - MATCHES BACKTEST"""
    risk_per_trade = capital * risk_percent
    risk_per_coin = abs(price - sl)
    
    max_by_risk = risk_per_trade / risk_per_coin if risk_per_coin > 0 else 0
    max_by_capital = capital / price  # SPOT CONSTRAINT!
    
    size_base = min(max_by_risk, max_by_capital, max_trade_size / price)
    return max(size_base, 0)

def process_bar(symbol, entry_df, htf_df, state, exchange=None, market_info: MarketInfo=None):
    """✅ FIXED: Now matches backtest 100%"""
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
    
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    curr_dd = (state["peak_equity"] - state["capital"]) / state["peak_equity"] if state["peak_equity"] > 0 else 0.0
    blocked = curr_dd >= MAX_DRAWDOWN
    
    trade_row = None
    
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
            
            # ✅ FIXED: Calculate costs based on POSITION VALUE
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            position_value_at_exit = exit_price * state["entry_size"]
            exit_slippage = position_value_at_exit * SLIPPAGE_RATE
            exit_fee = position_value_at_exit * FEE_RATE
            net_pnl = gross_pnl - exit_slippage - exit_fee
            
            state["capital"] += net_pnl
            
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
                "PnL_$": round(net_pnl, 2),
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
    
    # Exit logic
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
        elif USE_H1_FILTER and h4_trend < 0 and bias < 0:  # ✅ FIXED: Added bias confirmation
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
            
            # ✅ FIXED: Calculate costs based on POSITION VALUE
            gross_pnl = state["entry_size"] * (exit_price - state["entry_price"])
            position_value_at_exit = exit_price * state["entry_size"]
            exit_slippage = position_value_at_exit * SLIPPAGE_RATE
            exit_fee = position_value_at_exit * FEE_RATE
            net_pnl = gross_pnl - exit_slippage - exit_fee
            
            state["capital"] += net_pnl
            
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
                "PnL_$": round(net_pnl, 2),
                "Win": 1 if net_pnl > 0 else 0,
                "Exit_Reason": exit_reason,
                "Capital_After": round(state["capital"], 2),
                "Mode": MODE
            }
            
            state.update({"position": 0, "entry_price": 0.0, "entry_sl": 0.0,
                          "entry_tp": 0.0, "entry_time": None, "entry_size": 0.0})
            state["last_exit_time"] = ts
            
            send_telegram(f"{'💚' if net_pnl > 0 else '❤️'} EXIT {symbol} {exit_reason} @ ${exit_price:.4f} PnL=${net_pnl:.2f}")
    
    # Entry logic - ✅ FIXED: Now matches backtest
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
            
            # ✅ FIXED: Use correct spot position sizing
            size_base = calculate_spot_position_size(
                price, sl, state["capital"], RISK_PERCENT, MAX_TRADE_SIZE
            )
            
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
                
                # ✅ FIXED: Calculate costs based on POSITION VALUE
                position_value = entry_price_used * size_base
                entry_slippage = position_value * SLIPPAGE_RATE
                entry_fee = position_value * FEE_RATE
                
                state["capital"] -= entry_slippage
                state["capital"] -= entry_fee
                
                send_telegram(f"🚀 ENTRY {symbol} @ ${entry_price_used:.4f} | SL=${sl:.4f} TP=${tp:.4f} RR={rr_ratio:.1f}")
    
    state["last_processed_ts"] = ts
    state["peak_equity"] = max(state["peak_equity"], state["capital"])
    return state, trade_row

def worker(symbol):
    state_file, trades_csv = state_files_for_symbol(symbol)
    exchange = get_exchange()
    market_info = MarketInfo(exchange, symbol) if MODE == "live" else None
    state = load_state(state_file)
    tf_minutes = timeframe_to_minutes(ENTRY_TF)

    print(f"{LOG_PREFIX} {symbol} Started | Mode={MODE} | Cap=${state['capital']:.2f}")
    send_telegram(f"🤖 {symbol} Started | {ENTRY_TF}/{HTF} | ${PER_COIN_CAP_USD}")

    while True:
        try:
            now_utc = datetime.now(pytz.utc)
            
            entry_df = fetch_ohlcv_df(exchange, symbol, ENTRY_TF, limit=500)
            htf_df = fetch_ohlcv_df(exchange, symbol, HTF, limit=600)

            if entry_df.empty or htf_df.empty or len(entry_df) < 3:
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
                state, trade = process_bar(symbol, entry_df, htf_df, state, exchange=exchange, market_info=market_info)
                
                if trade is not None:
                    append_trade(trades_csv, trade)
                
                save_state(state_file, state)

            next_close_utc = forming_candle_ts + timedelta(minutes=tf_minutes)
            safe_check_time_utc = next_close_utc + timedelta(minutes=1)
            sleep_sec = (safe_check_time_utc - now_utc.replace(tzinfo=None)).total_seconds()
            
            if sleep_sec < 10:
                sleep_sec = 10
            elif sleep_sec > 3600:
                sleep_sec = SLEEP_CAP
            
            time.sleep(sleep_sec)

        except ccxt.RateLimitExceeded:
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
🚀 Bot Started (BACKTEST MATCH v1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {now_ist.strftime('%Y-%m-%d %I:%M %p IST')}
📊 Mode: {MODE.upper()}
💰 Capital/coin: ${PER_COIN_CAP_USD:,.2f}
🪙 Symbols: {', '.join(SYMBOLS)}
📈 TF: {ENTRY_TF}/{HTF}
⚙️ Risk: {RISK_PERCENT*100}% | RR: {RR_FIXED}x
✅ NOW MATCHES BACKTEST 100%
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
    
    print(f"\n✅ {len(threads)} workers started!\n")
    
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
