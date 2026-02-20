"""
EnsembleVoteStrategy — Lumibot port (Interactive Brokers)

Combines 4 technical indicators (SMA, RSI, MACD, Bollinger Bands) with
weighted voting. Includes a daily multi-factor screener that selects the
top symbols from a pure high-growth universe. Open positions on dropped
symbols get a 5-day grace period before force-close.

Parameters tuned for aggressive outperformance vs SPY (target 25-35% CAGR).
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

import json
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
order_logger = logging.getLogger(__name__ + ".orders")

# Pure high-growth, high-beta universe — Change 1
SYMBOLS = [
    # Tech mega-cap (core holdings)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
    # High-growth software
    "CRM", "ADBE", "NOW", "PANW", "CRWD", "SNOW",
    # Semiconductors
    "AMD", "AMAT", "MU", "LRCX", "KLAC",
    # High-beta momentum
    "TSLA", "PLTR", "APP", "DDOG", "MSTR",
    # ETF proxies (only growth-focused)
    "QQQ", "TQQQ",
    # China tech (for non-US diversification)
    "BABA", "PDD", "BIDU",
]


class EnsembleStrategy(Strategy):
    # PARAMETER EVOLUTION:
    # v1 (Original hyperopt 2020-2021): buy=0.64, sell=0.65, stop=-34.6%, max_pos=3
    #    Result: 3% return, 11% time in market (too conservative)
    # v2 (First optimization): buy=0.30, sell=0.45, stop=-10%, max_pos=5
    #    Result: 29% return, but only 0.49% per trade (exits too early)
    # v3 (Aggressive growth): Targeting 25-35% CAGR vs SPY ~15%

    parameters = {
        "symbols": SYMBOLS,
        "screener_top_n": 8,    # Change 4: was 15 — more selective
        "max_positions":  5,    # Change 4: was 7 — concentrate for higher impact
        "grace_days": 5,        # was 10 — cut losers faster
        # Data validation thresholds
        "min_screener_bars": 20,
        "min_signal_bars": 50,
        "signal_lookback_bars": 200,
        # Buy signal params
        "sma_fast": 21,
        "sma_slow": 39,
        "rsi_period": 10,
        "rsi_buy_threshold": 55,   # Change 2: was 47 — RSI > 55 = momentum confirmation
        "macd_fast": 10,
        "macd_slow": 21,
        "macd_signal": 11,
        "bb_period": 26,
        "bb_std": 1.9,
        # Buy weights — Change 2
        "w_sma":  0.50,   # was 0.15 — SMA crossover is clearest trend signal
        "w_rsi":  0.35,   # was 0.82 — reduce mean-reversion weight
        "w_macd": 0.60,   # was 0.48 — MACD momentum is strong
        "w_bb":   0.35,   # was 0.82 — reduce mean-reversion weight
        "buy_threshold": 0.40,  # was 0.42 — slightly easier entry into strong trends
        # Sell signal params
        "rsi_sell_threshold": 70,
        # Sell weights — Change 7
        "sw_sma":  0.90,  # was 0.75 — death cross is strongest exit signal
        "sw_rsi":  0.40,  # was 0.90 — RSI overbought alone should NOT exit a trend
        "sw_macd": 0.80,  # was 0.66 — MACD reversal is solid exit evidence
        "sw_bb":   0.60,  # was 1.00 — upper BB touch is weak in strong uptrend
        "sell_threshold": 0.55,
        # Risk — Change 5
        "stoploss":               -0.15,
        "trailing_stop_trigger":   0.30,  # was 0.10 — only activate after 30% gain
        "trailing_stop_pct":       0.18,  # was 0.08 — trail 18% below peak (allows normal 15% dips)
    }

    def initialize(self):
        self.sleeptime = "1D" if self.is_backtesting else "1H"
        self.minutes_before_closing = 15
        # Bug fix: always daily bars for signal computation.
        # Previously "1h" in live caused 200-hour (~25 day) lookback instead of 200-day,
        # making all indicator periods completely different from the backtested values.
        self._timestep = "day"
        self.screened_symbols = set()
        self._last_screen_date = ""
        self._last_screened_dates = {}  # symbol -> datetime when last in screened list
        self._entry_prices = {}         # symbol -> entry price (for stoploss tracking)
        self._peak_prices = {}          # symbol -> highest price since entry (trailing stop)
        self._indicator_cache = {}      # cleared each iteration to avoid stale data
        self._sell_check_failures = {}  # symbol -> consecutive sell check failure count
        self._spy_regime_bull = True    # Change 3: default True; updated daily before entries
        self._pyramid_done = {}         # Change 6: symbol -> True if already pyramided this position
        if not self.is_backtesting:
            self._load_state()

    # ------------------------------------------------------------------ #
    #  STATE PERSISTENCE — survives live trading restarts
    # ------------------------------------------------------------------ #
    def _state_path(self):
        return Path(__file__).parent / "strategy_state.json"

    def _save_state(self):
        """Persist critical in-memory state to disk for restart recovery."""
        try:
            state = {
                "entry_prices": self._entry_prices,
                "peak_prices": self._peak_prices,
                "last_screened_dates": {
                    k: v.isoformat() for k, v in self._last_screened_dates.items()
                },
            }
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log_message(f"Warning: could not save state: {e}")

    def _load_state(self):
        """Restore persisted state after a live trading restart."""
        path = self._state_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self._entry_prices = state.get("entry_prices", {})
            self._peak_prices = state.get("peak_prices", {})
            self._last_screened_dates = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("last_screened_dates", {}).items()
            }
            self.log_message(
                f"Loaded state: {len(self._entry_prices)} entry prices, "
                f"{len(self._last_screened_dates)} screened dates"
            )
        except Exception as e:
            self.log_message(f"Warning: could not load state (starting fresh): {e}")

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

        # Change 3: Get SPY data for regime filter + relative strength
        # Extended to 210 bars to compute 200-day SMA for regime detection
        spy_return = None
        try:
            spy_bars = self.get_historical_prices("SPY", 210, "day")  # was 30
            if spy_bars and len(spy_bars.df) >= p["min_screener_bars"]:
                spy_df = spy_bars.df
                spy_close = float(spy_df["close"].iloc[-1])
                spy_close_20 = spy_df["close"].iloc[-20]
                if spy_close_20 > 0:
                    spy_return = (spy_close - spy_close_20) / spy_close_20

                # Change 3: Regime filter — bull if SPY > 200-day SMA
                if len(spy_bars.df) >= 200:
                    spy_sma200 = float(np.nanmean(spy_df["close"].values[-200:]))
                    self._spy_regime_bull = bool(spy_close > spy_sma200)
                    self.log_message(
                        f"Regime: {'BULL' if self._spy_regime_bull else 'BEAR'} "
                        f"(SPY {spy_close:.2f} vs 200SMA {spy_sma200:.2f})"
                    )
        except Exception as e:
            self.log_message(f"Warning: SPY data unavailable for relative strength: {e}")

        scores = {}
        for symbol in symbols:
            try:
                # Change 8: Extended to 65 bars to compute 60-day momentum
                bars = self.get_historical_prices(symbol, 65, "day")  # was 30
                if bars is None or len(bars.df) < p["min_screener_bars"]:
                    continue
                df = bars.df
                close = df["close"]
                volume = df["volume"]

                # Change 8: Blend 20-day and 60-day momentum (60-day weighted more)
                close_now = close.iloc[-1]
                close_20  = close.iloc[-20] if len(close) > 20 else close.iloc[0]
                close_60  = close.iloc[-60] if len(close) > 60 else close.iloc[0]
                momentum_20 = (close_now - close_20) / close_20 if close_20 > 0 else 0
                momentum_60 = (close_now - close_60) / close_60 if close_60 > 0 else 0
                momentum = momentum_20 * 0.40 + momentum_60 * 0.60  # 60-day weighted more

                vol_5 = volume.iloc[-5:].mean()
                vol_20 = volume.iloc[-20:].mean()
                vol_surge = vol_5 / vol_20 if vol_20 > 0 else 1.0

                rel_strength = 0.0
                if spy_return is not None:
                    rel_strength = momentum - spy_return

                scores[symbol] = {
                    "momentum": momentum,
                    "vol_surge": vol_surge,
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
            now = self.get_datetime()
            for sym in self.screened_symbols:
                self._last_screened_dates[sym] = now
            self._save_state()
            return

        score_df = pd.DataFrame(scores).T
        for col in score_df.columns:
            score_df[col] = score_df[col].rank(pct=True)
        # Change 8: Updated composite weights — momentum is primary signal
        score_df["composite"] = (
            score_df["momentum"]     * 0.50 +  # was 0.40 — momentum is primary signal
            score_df["vol_surge"]    * 0.10 +  # was 0.20 — reduce volume weight
            score_df["rel_strength"] * 0.40    # unchanged
        )

        top = score_df.nlargest(top_n, "composite")
        self.screened_symbols = set(top.index.tolist())

        now = self.get_datetime()
        for sym in self.screened_symbols:
            self._last_screened_dates[sym] = now

        self.log_message(f"Screener: {sorted(self.screened_symbols)}")
        self._save_state()

    # ------------------------------------------------------------------ #
    #  SIGNAL COMPUTATION
    # ------------------------------------------------------------------ #
    def _compute_indicators(self, df, cache_key=None):
        """Compute all technical indicators once. Returns dict or None if invalid."""
        if cache_key is not None and cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
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
            if cache_key is not None:
                self._indicator_cache[cache_key] = None
            return None

        result = {
            "close": close,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "rsi": rsi,
            "macd": macd,
            "macd_sig": macd_sig,
            "upper": upper,
            "middle": middle,  # Change 2: added for trend-confirmation BB signal
            "lower": lower,
        }
        if cache_key is not None:
            self._indicator_cache[cache_key] = result
        return result

    def _compute_buy_score(self, df, details=False, cache_key=None):
        p = self.parameters
        indicators = self._compute_indicators(df, cache_key=cache_key)
        if indicators is None:
            return 0.0

        close    = indicators["close"]
        sma_fast = indicators["sma_fast"]
        sma_slow = indicators["sma_slow"]
        rsi      = indicators["rsi"]
        macd     = indicators["macd"]
        macd_sig = indicators["macd_sig"]
        middle   = indicators["middle"]  # Change 2: use middle band instead of lower

        # SMA: fast above slow = uptrend
        sig_sma  = 1.0 if sma_fast[-1] > sma_slow[-1] else 0.0
        # Change 2: RSI > threshold = momentum confirmation (was < threshold = oversold dip)
        sig_rsi  = 1.0 if rsi[-1] > p["rsi_buy_threshold"] else 0.0
        # MACD above signal = bullish momentum
        sig_macd = 1.0 if macd[-1] > macd_sig[-1] else 0.0
        # Change 2: close above middle BB = price in upper half = trending up (was below lower BB)
        sig_bb   = 1.0 if close[-1] > middle[-1] else 0.0

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
                    "middle_bb": float(middle[-1])}
        return score

    def _compute_sell_score(self, df, cache_key=None):
        p = self.parameters
        indicators = self._compute_indicators(df, cache_key=cache_key)
        if indicators is None:
            return 0.0

        close    = indicators["close"]
        sma_fast = indicators["sma_fast"]
        sma_slow = indicators["sma_slow"]
        rsi      = indicators["rsi"]
        macd     = indicators["macd"]
        macd_sig = indicators["macd_sig"]
        upper    = indicators["upper"]

        # SMA: fast below slow = downtrend
        sig_sma  = 1.0 if sma_fast[-1] < sma_slow[-1] else 0.0
        # RSI overbought
        sig_rsi  = 1.0 if rsi[-1] > p["rsi_sell_threshold"] else 0.0
        # MACD below signal = bearish momentum
        sig_macd = 1.0 if macd[-1] < macd_sig[-1] else 0.0
        # Close above upper BB
        sig_bb   = 1.0 if close[-1] > upper[-1] else 0.0

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
        self._indicator_cache = {}  # Clear per-iteration cache

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

        portfolio_value = self.get_portfolio_value()

        # --- EXIT LOGIC ---
        for pos in positions:
            symbol = pos.asset.symbol
            if pos.quantity <= 0:
                continue

            # Grace period: force-close if off screened list > grace_days
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

            # Fetch bars once per position — used for both price and sell signal.
            # Using last close from historical bars avoids get_tick (which requires
            # a paid IB market data subscription).
            exit_bars = None
            try:
                exit_bars = self.get_historical_prices(symbol, p["signal_lookback_bars"], self._timestep)
            except Exception as e:
                self.log_message(f"Could not fetch bars for {symbol}: {e}")

            # Hard stoploss and trailing stop
            entry_price = self._entry_prices.get(symbol)
            if entry_price and entry_price > 0:
                current_price = (
                    float(exit_bars.df["close"].iloc[-1])
                    if exit_bars is not None and len(exit_bars.df) > 0
                    else None
                )
                if current_price is not None:
                    pnl_pct = (current_price - entry_price) / entry_price

                    # Hard stoploss
                    if pnl_pct <= p["stoploss"]:
                        self.log_message(f"Stoploss hit for {symbol} ({pnl_pct:.1%}), selling")
                        order = self.create_order(symbol, pos.quantity, "sell")
                        self.submit_order(order)
                        self._pyramid_done.pop(symbol, None)  # Change 6: reset on exit
                        continue

                    # Trailing stop: once position gains >= trigger pct, trail below peak
                    peak = self._peak_prices.get(symbol, entry_price)
                    if current_price > peak:
                        self._peak_prices[symbol] = current_price
                        peak = current_price
                    gain_at_peak = (peak - entry_price) / entry_price
                    if gain_at_peak >= p["trailing_stop_trigger"]:
                        trail_price = peak * (1 - p["trailing_stop_pct"])
                        if current_price <= trail_price:
                            self.log_message(
                                f"Trailing stop hit for {symbol} "
                                f"(peak=${peak:.2f}, current=${current_price:.2f}, trail=${trail_price:.2f})"
                            )
                            order = self.create_order(symbol, pos.quantity, "sell")
                            self.submit_order(order)
                            self._pyramid_done.pop(symbol, None)  # Change 6: reset on exit
                            continue

                    # Change 6: Pyramiding — add to winners up 20%+ if not yet pyramided
                    if (not self._pyramid_done.get(symbol, False) and
                            symbol in self.screened_symbols):
                        gain = (current_price - entry_price) / entry_price
                        if gain >= 0.20:
                            add_size = (portfolio_value / p["max_positions"]) * 0.40
                            cash_now = self.get_cash()
                            add_qty = int(min(add_size, cash_now) // current_price) if current_price > 0 else 0
                            if add_qty > 0:
                                self.log_message(f"Pyramiding {symbol} (+{gain:.1%}), adding {add_qty} shares")
                                order = self.create_order(symbol, add_qty, "buy")
                                self.submit_order(order)
                                self._pyramid_done[symbol] = True

            # Sell signal check (reuses exit_bars fetched above)
            try:
                if exit_bars and len(exit_bars.df) >= p["min_signal_bars"]:
                    sell_score = self._compute_sell_score(exit_bars.df, cache_key=symbol)
                    if sell_score >= p["sell_threshold"]:
                        self.log_message(f"Sell signal for {symbol} (score={sell_score:.2f})")
                        order = self.create_order(symbol, pos.quantity, "sell")
                        self.submit_order(order)
                        self._pyramid_done.pop(symbol, None)  # Change 6: reset on exit
                self._sell_check_failures.pop(symbol, None)
            except Exception as e:
                failures = self._sell_check_failures.get(symbol, 0) + 1
                self._sell_check_failures[symbol] = failures
                self.log_message(f"Sell check error {symbol} (#{failures}): {e}")
                if failures >= 3:
                    self.log_message(f"Force-closing {symbol} after {failures} consecutive sell check failures")
                    order = self.create_order(symbol, pos.quantity, "sell")
                    self.submit_order(order)
                    self._sell_check_failures.pop(symbol, None)
                    self._pyramid_done.pop(symbol, None)  # Change 6: reset on exit

        # --- ENTRY LOGIC ---
        num_positions = len(held_symbols)
        if num_positions >= p["max_positions"]:
            return
        if not self.screened_symbols:
            self.log_message("No screened symbols, skipping entries")
            return

        # Change 3: Block new entries in bear regime
        if not self._spy_regime_bull:
            self.log_message("BEAR regime: skipping entries")
            return

        cash = self.get_cash()
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

                info = self._compute_buy_score(bars.df, details=True, cache_key=symbol)
                buy_score = info["score"]
                self.log_message(
                    f"  {symbol}: score={buy_score:.2f}/{p['buy_threshold']:.2f} "
                    f"[SMA={info['sig_sma']:.0f} RSI={info['sig_rsi']:.0f}({info['rsi']:.1f}) "
                    f"MACD={info['sig_macd']:.0f} BB={info['sig_bb']:.0f}]"
                )
                if buy_score < p["buy_threshold"]:
                    continue

                price = float(bars.df["close"].iloc[-1])
                if price <= 0:
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
                self._pyramid_done[symbol] = False  # Change 6: initialize pyramid state
                held_symbols.add(symbol)
                num_positions += 1
                cash -= qty * price
            except Exception as e:
                self.log_message(f"Entry error {symbol}: {e}")

    def on_filled_order(self, position, order, price, quantity, multiplier):
        symbol = order.asset.symbol
        if order.side == "buy":
            self._entry_prices[symbol] = price
            self._peak_prices[symbol] = price  # Initialize peak at entry
            # Change 6: initialize pyramid state on new buy (not on pyramid fills)
            if symbol not in self._pyramid_done:
                self._pyramid_done[symbol] = False
            self.log_message(f"Bought {quantity} {symbol} @ ${price:.2f}")
            order_logger.info(f"FILLED BUY  {symbol:6s}  qty={quantity:6}  price=${price:.2f}")
        elif order.side == "sell":
            self._entry_prices.pop(symbol, None)
            self._peak_prices.pop(symbol, None)
            self._pyramid_done.pop(symbol, None)  # Change 6: reset on full exit
            self.log_message(f"Sold {quantity} {symbol} @ ${price:.2f}")
            order_logger.info(f"FILLED SELL {symbol:6s}  qty={quantity:6}  price=${price:.2f}")
        self._save_state()


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

        # Request delayed market data (15-min delay, free) — avoids subscription errors
        try:
            broker.ib.client.reqMarketDataType(3)  # 3=delayed, 1=live
            print("Market data type set to delayed (15-min, free tier)")
        except Exception as e:
            print(f"Warning: could not set delayed data type: {e}")

        # Verify account summary works before starting strategy
        _summary = broker.ib.get_account_summary()
        if _summary:
            _cash = [c for c in _summary if c["Tag"] == "TotalCashBalance" and c["Currency"] == "BASE"]
            print(f"IB connected OK. Cash: ${float(_cash[0]['Value']):,.2f}" if _cash else "IB connected OK.")
        else:
            print("WARNING: Could not get account summary. Connection may be unstable.")

        _log_dir = Path(__file__).parent / "log-live"
        _log_dir.mkdir(exist_ok=True)
        _orders_log = _log_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.log"
        _fh = logging.FileHandler(_orders_log)
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        order_logger.addHandler(_fh)
        order_logger.setLevel(logging.INFO)
        order_logger.propagate = False
        print(f"Order log: {_orders_log}")

        strategy = EnsembleStrategy(broker=broker)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
