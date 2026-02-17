"""
EnsembleVoteStrategy — Lumibot port (Interactive Brokers)

Combines 4 technical indicators (SMA, RSI, MACD, Bollinger Bands) with
weighted voting. Includes a daily multi-factor screener that selects the
top 10 symbols from ~90 diversified tickers. Open positions on dropped
symbols get a 7-day grace period before force-close.

Parameters are from Freqtrade hyperopt optimization (200 epochs, Sharpe loss).
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Prevent lumibot credentials.py from auto-creating an IB broker at import time.
# We must block the IB env vars during import, then load them afterward.
os.environ["LUMIBOT_DISABLE_DOTENV"] = "1"

# Fix Windows cp1252 encoding crash with Lumibot's Unicode progress bar
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import talib

from lumibot.strategies.strategy import Strategy
from lumibot.brokers import InteractiveBrokers
from lumibot.traders import Trader
from lumibot.backtesting import PolygonDataBacktesting, YahooDataBacktesting

# NOW load .env (after lumibot imports, so credentials.py doesn't auto-detect IB)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)
logger = logging.getLogger(__name__)

# All ~90 symbols from the diversified universe
SYMBOLS = [
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLF", "XLE", "XLK", "XLV",
    "XLI", "XLP", "XLY", "XLU", "XLRE", "XBI", "SMH", "KRE", "EEM", "HYG",
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "CRM", "ORCL", "ADBE",
    "AMD", "INTC", "CSCO", "AMAT", "MU",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "AXP", "BLK",
    # Healthcare
    "UNH", "JNJ", "ABBV", "PFE", "MRK", "LLY", "TMO", "ABT", "AMGN", "BMY",
    # Consumer
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
    # Industrial
    "CAT", "DE", "BA", "RTX", "HON", "UPS", "GE", "LMT", "MMM", "UNP",
    # Energy
    "XOM", "CVX", "SLB", "COP", "EOG", "MPC", "PSX", "OXY", "VLO", "HAL",
    # Other
    "NEE", "DIS", "NFLX", "T", "VZ", "PM", "DUK", "SO", "D", "SRE",
]


class EnsembleStrategy(Strategy):
    parameters = {
        "symbols": SYMBOLS,
        "screener_top_n": 10,
        "max_positions": 3,
        "grace_days": 7,
        # Data validation thresholds
        "min_screener_bars": 20,
        "min_signal_bars": 50,
        "signal_lookback_bars": 200,
        # Buy signal params (from hyperopt)
        "sma_fast": 21,
        "sma_slow": 39,
        "rsi_period": 10,
        "rsi_buy_threshold": 36,
        "macd_fast": 10,
        "macd_slow": 21,
        "macd_signal": 11,
        "bb_period": 26,
        "bb_std": 2.3,
        # Buy weights
        "w_sma": 0.15,
        "w_rsi": 0.82,
        "w_macd": 0.48,
        "w_bb": 0.82,
        "buy_threshold": 0.64,
        # Sell signal params
        "rsi_sell_threshold": 67,
        # Sell weights
        "sw_sma": 0.75,
        "sw_rsi": 0.90,
        "sw_macd": 0.66,
        "sw_bb": 1.00,
        "sell_threshold": 0.65,
        # Risk
        "stoploss": -0.346,
    }

    def initialize(self):
        # Daily for Yahoo backtesting, hourly for live trading
        self.sleeptime = "1D" if self.is_backtesting else "1H"
        self.minutes_before_closing = 15
        self._timestep = "day" if self.is_backtesting else "1h"
        self.screened_symbols = set()
        self._last_screen_date = ""
        self._last_screened_dates = {}  # symbol -> datetime when last in screened list
        self._entry_prices = {}  # symbol -> entry price (for stoploss tracking)

    # ------------------------------------------------------------------ #
    #  SCREENER — runs once per day before market opens
    # ------------------------------------------------------------------ #
    def before_market_opens(self):
        today = self.get_datetime().strftime("%Y-%m-%d")
        if today == self._last_screen_date:
            return
        self._last_screen_date = today

        p = self.parameters
        symbols = p["symbols"]
        top_n = p["screener_top_n"]

        # Get SPY return for relative strength
        spy_return = None
        try:
            spy_bars = self.get_historical_prices("SPY", 30, "day")
            if spy_bars and len(spy_bars.df) >= p["min_screener_bars"]:
                spy_df = spy_bars.df
                spy_close = spy_df["close"].iloc[-1]
                spy_close_20 = spy_df["close"].iloc[-20]
                if spy_close_20 > 0:
                    spy_return = (spy_close - spy_close_20) / spy_close_20
        except Exception as e:
            self.log_message(f"Warning: SPY data unavailable for relative strength: {e}")

        scores = {}
        for symbol in symbols:
            try:
                bars = self.get_historical_prices(symbol, 30, "day")
                if bars is None or len(bars.df) < p["min_screener_bars"]:
                    continue
                df = bars.df
                close = df["close"]
                volume = df["volume"]

                close_now = close.iloc[-1]
                close_20 = close.iloc[-20]
                momentum = (close_now - close_20) / close_20 if close_20 > 0 else 0

                vol_5 = volume.iloc[-5:].mean()
                vol_20 = volume.iloc[-20:].mean()
                vol_surge = vol_5 / vol_20 if vol_20 > 0 else 1.0

                sma_20 = close.iloc[-20:].mean()
                mean_rev = (sma_20 - close_now) / sma_20 if sma_20 > 0 else 0

                rel_strength = 0.0
                if spy_return is not None:
                    rel_strength = momentum - spy_return

                scores[symbol] = {
                    "momentum": momentum,
                    "vol_surge": vol_surge,
                    "mean_rev": mean_rev,
                    "rel_strength": rel_strength,
                }
            except Exception as e:
                self.log_message(f"Screener skip {symbol}: {e}")
                continue

        skipped = len(symbols) - len(scores)
        self.log_message(f"Screener evaluated {len(scores)}/{len(symbols)} symbols ({skipped} skipped)")

        if not scores:
            self.log_message("Screener: no scores computed, using all symbols as fallback")
            self.screened_symbols = set(symbols)
            return

        score_df = pd.DataFrame(scores).T
        for col in score_df.columns:
            score_df[col] = score_df[col].rank(pct=True)
        score_df["composite"] = score_df.mean(axis=1)

        top = score_df.nlargest(top_n, "composite")
        self.screened_symbols = set(top.index.tolist())

        now = self.get_datetime()
        for sym in self.screened_symbols:
            self._last_screened_dates[sym] = now

        self.log_message(f"Screener: {sorted(self.screened_symbols)}")

    # ------------------------------------------------------------------ #
    #  SIGNAL COMPUTATION
    # ------------------------------------------------------------------ #
    def _compute_indicators(self, df):
        """Compute all technical indicators once. Returns dict or None if invalid."""
        p = self.parameters
        close = df["close"].values

        sma_fast = talib.SMA(close, timeperiod=p["sma_fast"])
        sma_slow = talib.SMA(close, timeperiod=p["sma_slow"])
        rsi = talib.RSI(close, timeperiod=p["rsi_period"])
        macd, macd_sig, _ = talib.MACD(
            close, fastperiod=p["macd_fast"],
            slowperiod=p["macd_slow"], signalperiod=p["macd_signal"]
        )
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=p["bb_period"],
            nbdevup=p["bb_std"], nbdevdn=p["bb_std"]
        )

        # Validate indicators to prevent NaN/Inf crashes
        if (np.isnan(sma_fast[-1]) or np.isnan(sma_slow[-1]) or
            np.isnan(rsi[-1]) or np.isnan(macd[-1]) or
            np.isnan(upper[-1]) or np.isnan(lower[-1])):
            return None

        return {
            "close": close,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "rsi": rsi,
            "macd": macd,
            "macd_sig": macd_sig,
            "upper": upper,
            "lower": lower,
        }

    def _compute_buy_score(self, df, details=False):
        p = self.parameters
        indicators = self._compute_indicators(df)
        if indicators is None:
            return 0.0

        # Extract indicator values
        close = indicators["close"]
        sma_fast = indicators["sma_fast"]
        sma_slow = indicators["sma_slow"]
        rsi = indicators["rsi"]
        macd = indicators["macd"]
        macd_sig = indicators["macd_sig"]
        lower = indicators["lower"]

        # State-based signals (work on both daily and hourly bars)
        # SMA: fast above slow = uptrend
        sig_sma = 1.0 if sma_fast[-1] > sma_slow[-1] else 0.0
        # RSI oversold
        sig_rsi = 1.0 if rsi[-1] < p["rsi_buy_threshold"] else 0.0
        # MACD above signal = bullish momentum
        sig_macd = 1.0 if macd[-1] > macd_sig[-1] else 0.0
        # Close below lower BB
        sig_bb = 1.0 if close[-1] < lower[-1] else 0.0

        total_w = p["w_sma"] + p["w_rsi"] + p["w_macd"] + p["w_bb"]
        if total_w == 0:
            score = 0.0
        else:
            score = (sig_sma * p["w_sma"] + sig_rsi * p["w_rsi"] +
                     sig_macd * p["w_macd"] + sig_bb * p["w_bb"]) / total_w
        if details:
            return {"score": score, "sig_sma": sig_sma, "sig_rsi": sig_rsi,
                    "sig_macd": sig_macd, "sig_bb": sig_bb,
                    "rsi": float(rsi[-1]), "close": float(close[-1]),
                    "lower_bb": float(lower[-1])}
        return score

    def _compute_sell_score(self, df):
        p = self.parameters
        indicators = self._compute_indicators(df)
        if indicators is None:
            return 0.0

        # Extract indicator values
        close = indicators["close"]
        sma_fast = indicators["sma_fast"]
        sma_slow = indicators["sma_slow"]
        rsi = indicators["rsi"]
        macd = indicators["macd"]
        macd_sig = indicators["macd_sig"]
        upper = indicators["upper"]

        # State-based signals (work on both daily and hourly bars)
        # SMA: fast below slow = downtrend
        sig_sma = 1.0 if sma_fast[-1] < sma_slow[-1] else 0.0
        # RSI overbought
        sig_rsi = 1.0 if rsi[-1] > p["rsi_sell_threshold"] else 0.0
        # MACD below signal = bearish momentum
        sig_macd = 1.0 if macd[-1] < macd_sig[-1] else 0.0
        # Close above upper BB
        sig_bb = 1.0 if close[-1] > upper[-1] else 0.0

        total_w = p["sw_sma"] + p["sw_rsi"] + p["sw_macd"] + p["sw_bb"]
        if total_w == 0:
            return 0.0
        score = (sig_sma * p["sw_sma"] + sig_rsi * p["sw_rsi"] +
                 sig_macd * p["sw_macd"] + sig_bb * p["sw_bb"]) / total_w
        return score

    # ------------------------------------------------------------------ #
    #  MAIN TRADING LOOP — every hour
    # ------------------------------------------------------------------ #
    def on_trading_iteration(self):
        p = self.parameters
        now = self.get_datetime()
        positions = self.get_positions()
        held_symbols = {pos.asset.symbol for pos in positions if pos.quantity > 0}

        # Safety net: if screener hasn't run today, run it now
        if not self.screened_symbols:
            today = now.strftime("%Y-%m-%d")
            if today != self._last_screen_date:
                self.log_message("Screener not yet run today, triggering now")
                self.before_market_opens()

        self.log_message(
            f"Iteration: {now.strftime('%H:%M')} | positions={len(held_symbols)}/{p['max_positions']} | "
            f"screened={len(self.screened_symbols)} symbols"
        )

        if self.first_iteration:
            self.log_message(f"First iteration. Screened: {len(self.screened_symbols)} symbols: {sorted(self.screened_symbols)}")
            self.log_message(f"Cash: ${self.cash:.2f}, Portfolio: ${self.portfolio_value:.2f}")

        # --- EXIT LOGIC ---
        for pos in positions:
            symbol = pos.asset.symbol
            if pos.quantity <= 0:
                continue

            # Grace period: force-close if off screened list > 7 days
            if symbol not in self.screened_symbols:
                last_seen = self._last_screened_dates.get(symbol)
                if last_seen is None:
                    # Position existed before strategy start or tracking bug
                    self.log_message(f"Warning: {symbol} has no screener history, force-closing")
                    order = self.create_order(symbol, pos.quantity, "sell")
                    self.submit_order(order)
                    continue
                if (now - last_seen).days > p["grace_days"]:
                    self.log_message(f"Grace expired for {symbol}, selling")
                    order = self.create_order(symbol, pos.quantity, "sell")
                    self.submit_order(order)
                    continue

            # Stoploss check
            entry_price = self._entry_prices.get(symbol)
            if entry_price:
                current_price = self.get_last_price(symbol)
                if current_price is not None and entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= p["stoploss"]:
                        self.log_message(f"Stoploss hit for {symbol} ({pnl_pct:.1%}), selling")
                        order = self.create_order(symbol, pos.quantity, "sell")
                        self.submit_order(order)
                        continue

            # Sell signal check
            try:
                bars = self.get_historical_prices(symbol, p["signal_lookback_bars"], self._timestep)
                if bars and len(bars.df) >= p["min_signal_bars"]:
                    sell_score = self._compute_sell_score(bars.df)
                    if sell_score >= p["sell_threshold"]:
                        self.log_message(f"Sell signal for {symbol} (score={sell_score:.2f})")
                        order = self.create_order(symbol, pos.quantity, "sell")
                        self.submit_order(order)
            except Exception as e:
                self.log_message(f"Sell check error {symbol}: {e}")

        # --- ENTRY LOGIC ---
        num_positions = len(held_symbols)
        if num_positions >= p["max_positions"]:
            return
        if not self.screened_symbols:
            self.log_message("No screened symbols, skipping entries")
            return

        cash = self.get_cash()
        portfolio_value = self.get_portfolio_value()
        position_size = portfolio_value / p["max_positions"]

        for symbol in self.screened_symbols:
            if num_positions >= p["max_positions"]:
                break
            if symbol in held_symbols:
                continue

            try:
                bars = self.get_historical_prices(symbol, p["signal_lookback_bars"], self._timestep)
                if bars is None or len(bars.df) < p["min_signal_bars"]:
                    bar_len = 0 if bars is None else len(bars.df)
                    self.log_message(f"  {symbol}: skip, insufficient bars ({bar_len}/{p['min_signal_bars']})")
                    continue

                info = self._compute_buy_score(bars.df, details=True)
                buy_score = info["score"]
                self.log_message(
                    f"  {symbol}: score={buy_score:.2f}/{p['buy_threshold']:.2f} "
                    f"[SMA={info['sig_sma']:.0f} RSI={info['sig_rsi']:.0f}({info['rsi']:.1f}) "
                    f"MACD={info['sig_macd']:.0f} BB={info['sig_bb']:.0f}]"
                )
                if buy_score < p["buy_threshold"]:
                    continue

                price = self.get_last_price(symbol)
                if price is None or price <= 0:
                    self.log_message(f"  {symbol}: skip, no price available")
                    continue

                qty = int(min(position_size, cash) // price)
                if qty <= 0:
                    self.log_message(f"  {symbol}: skip, qty=0 (price=${price:.2f}, budget=${min(position_size, cash):.2f})")
                    continue

                self.log_message(f"Buy signal for {symbol} (score={buy_score:.2f}), qty={qty}")
                order = self.create_order(symbol, qty, "buy")
                self.submit_order(order)
                self._entry_prices[symbol] = price
                held_symbols.add(symbol)
                num_positions += 1
                cash -= qty * price
            except Exception as e:
                self.log_message(f"Entry error {symbol}: {e}")

    def on_filled_order(self, position, order, price, quantity, multiplier):
        symbol = order.asset.symbol
        if order.side == "buy":
            self._entry_prices[symbol] = price
            self.log_message(f"Bought {quantity} {symbol} @ ${price:.2f}")
        elif order.side == "sell":
            self._entry_prices.pop(symbol, None)
            self.log_message(f"Sold {quantity} {symbol} @ ${price:.2f}")


# ------------------------------------------------------------------ #
#  ENTRY POINT
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    mode = os.getenv("MODE", "BACKTEST").upper()

    if mode == "BACKTEST":
        data_source = os.getenv("BACKTEST_DATA", "YAHOO").upper()

        if data_source == "POLYGON":
            polygon_key = os.getenv("POLYGON_API_KEY", "")
            if not polygon_key:
                print("ERROR: Set POLYGON_API_KEY in .env for Polygon backtesting")
                exit(1)
            EnsembleStrategy.run_backtest(
                PolygonDataBacktesting,
                datetime(2024, 3, 1),
                datetime(2026, 2, 1),
                benchmark_asset="SPY",
                budget=10_000,
                polygon_api_key=polygon_key,
                parameters={"symbols": SYMBOLS},
            )
        else:
            # Yahoo: free, no rate limits, daily bars only
            result = EnsembleStrategy.run_backtest(
                YahooDataBacktesting,
                datetime(2023, 1, 1),
                datetime(2025, 12, 31),
                benchmark_asset="SPY",
                budget=10_000,
                parameters={"symbols": SYMBOLS},
                save_tearsheet=True,
                show_tearsheet=False,
            )
            print(f"\n{'='*50}")
            print(f"Backtest Results")
            print(f"{'='*50}")
            if result and isinstance(result, dict):
                for k, v in result.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  Result: {result}")
    else:
        # Use PID-based client ID to avoid stale connection conflicts after crashes.
        # TWS takes time to clean up abruptly closed connections; reusing the same
        # client ID causes error 326 "client id already in use" and silent disconnects.
        default_cid = (os.getpid() % 9900) + 100  # 100-9999, avoids 0 and low IDs
        ib_config = {
            "SOCKET_PORT": int(os.getenv("INTERACTIVE_BROKERS_PORT", "7497")),
            "CLIENT_ID": int(os.getenv("INTERACTIVE_BROKERS_CLIENT_ID", str(default_cid))),
            "IP": os.getenv("INTERACTIVE_BROKERS_IP", "127.0.0.1"),
        }
        import time as _time
        print(f"Connecting to IB at {ib_config['IP']}:{ib_config['SOCKET_PORT']} "
              f"(clientId={ib_config['CLIENT_ID']})...")
        broker = InteractiveBrokers(ib_config)

        # Wait for connection to stabilize and verify it works
        _time.sleep(2)
        if not broker.ib.isConnected():
            print("ERROR: IB connection dropped after broker creation. Retrying...")
            broker.ib.disconnect()
            _time.sleep(3)
            broker.ib = None
            broker.data_source.ib = None
            broker.start_ib()
            _time.sleep(2)

        # Verify account summary works before starting strategy
        _summary = broker.ib.get_account_summary()
        if _summary:
            _cash = [c for c in _summary if c["Tag"] == "TotalCashBalance" and c["Currency"] == "BASE"]
            print(f"IB connected OK. Cash: ${float(_cash[0]['Value']):,.2f}" if _cash else "IB connected OK.")
        else:
            print("WARNING: Could not get account summary. Connection may be unstable.")

        strategy = EnsembleStrategy(broker=broker)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
