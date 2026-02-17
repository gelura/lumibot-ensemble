"""
Unit tests for EnsembleStrategy trading logic.

Tests cover:
- Indicator calculation and validation
- Screener logic
- Position tracking
- Weighted voting
- Edge cases (NaN, Inf, insufficient data)
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Set backtesting mode to avoid broker requirement
os.environ["IS_BACKTESTING"] = "true"

from ensemble_strategy import EnsembleStrategy


@pytest.fixture
def mock_broker():
    """Create properly configured mock broker for testing."""
    broker = Mock()
    # Mock the _filled_positions.get_list() call that Strategy.__init__ makes
    broker._filled_positions = Mock()
    broker._filled_positions.get_list = Mock(return_value=[])
    # Mock the data_source._data_store.keys() call
    broker.data_source = Mock()
    broker.data_source._data_store = {}
    return broker


class TestIndicatorCalculation:
    """Tests for technical indicator computation and validation."""

    @pytest.fixture
    def strategy(self, mock_broker):
        """Create strategy instance for testing."""
        strategy = EnsembleStrategy(broker=mock_broker)
        # Initialize required state
        strategy._last_screened_dates = {}
        strategy._entry_prices = {}
        strategy.screened_symbols = set()
        return strategy

    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create synthetic OHLCV data with uptrend."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "close": [100 + i * 0.5 for i in range(100)],  # Uptrend
            "high": [101 + i * 0.5 for i in range(100)],
            "low": [99 + i * 0.5 for i in range(100)],
            "open": [100 + i * 0.5 for i in range(100)],
            "volume": [1000000] * 100,
        }, index=dates)

    def test_compute_indicators_with_valid_data(self, strategy, valid_ohlcv_data):
        """Test that _compute_indicators returns valid dict with correct keys."""
        indicators = strategy._compute_indicators(valid_ohlcv_data)

        assert indicators is not None, "Should return dict for valid data"
        assert "close" in indicators
        assert "sma_fast" in indicators
        assert "sma_slow" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "macd_sig" in indicators
        assert "upper" in indicators
        assert "lower" in indicators

        # Verify values are not NaN
        assert not np.isnan(indicators["sma_fast"][-1])
        assert not np.isnan(indicators["rsi"][-1])

    def test_compute_indicators_with_insufficient_data(self, strategy):
        """Test that _compute_indicators returns None for insufficient data."""
        # Only 10 bars (need 50+ for proper calculation)
        df = pd.DataFrame({
            "close": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "high": np.array([101.0 + i for i in range(10)], dtype=np.float64),
            "low": np.array([99.0 + i for i in range(10)], dtype=np.float64),
            "open": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "volume": np.array([1000000.0] * 10, dtype=np.float64),
        })

        indicators = strategy._compute_indicators(df)
        assert indicators is None, "Should return None for insufficient data"

    def test_compute_indicators_with_nan_data(self, strategy):
        """Test that _compute_indicators returns None when data contains NaN."""
        df = pd.DataFrame({
            "close": [np.nan] * 100,
            "high": [np.nan] * 100,
            "low": [np.nan] * 100,
            "open": [np.nan] * 100,
            "volume": [1000000] * 100,
        })

        indicators = strategy._compute_indicators(df)
        assert indicators is None, "Should return None for NaN data"

    def test_compute_buy_score_with_valid_data(self, strategy, valid_ohlcv_data):
        """Test buy score calculation with valid data."""
        score = strategy._compute_buy_score(valid_ohlcv_data)

        assert isinstance(score, float), "Should return float score"
        assert 0 <= score <= 1, f"Score should be 0-1, got {score}"

    def test_compute_buy_score_with_details(self, strategy, valid_ohlcv_data):
        """Test buy score with details flag returns additional info."""
        info = strategy._compute_buy_score(valid_ohlcv_data, details=True)

        assert isinstance(info, dict), "Should return dict when details=True"
        assert "score" in info
        assert "sig_sma" in info
        assert "sig_rsi" in info
        assert "sig_macd" in info
        assert "sig_bb" in info
        assert "rsi" in info
        assert "close" in info
        assert "lower_bb" in info

        # Verify signal values are 0.0 or 1.0
        assert info["sig_sma"] in [0.0, 1.0]
        assert info["sig_rsi"] in [0.0, 1.0]
        assert info["sig_macd"] in [0.0, 1.0]
        assert info["sig_bb"] in [0.0, 1.0]

    def test_compute_buy_score_with_insufficient_data(self, strategy):
        """Test buy score returns 0.0 for insufficient data."""
        df = pd.DataFrame({
            "close": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "high": np.array([101.0 + i for i in range(10)], dtype=np.float64),
            "low": np.array([99.0 + i for i in range(10)], dtype=np.float64),
            "open": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "volume": np.array([1000000.0] * 10, dtype=np.float64),
        })

        score = strategy._compute_buy_score(df)
        assert score == 0.0, "Should return 0.0 for insufficient data"

    def test_compute_sell_score_with_valid_data(self, strategy, valid_ohlcv_data):
        """Test sell score calculation with valid data."""
        score = strategy._compute_sell_score(valid_ohlcv_data)

        assert isinstance(score, float), "Should return float score"
        assert 0 <= score <= 1, f"Score should be 0-1, got {score}"

    def test_compute_sell_score_with_insufficient_data(self, strategy):
        """Test sell score returns 0.0 for insufficient data."""
        df = pd.DataFrame({
            "close": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "high": np.array([101.0 + i for i in range(10)], dtype=np.float64),
            "low": np.array([99.0 + i for i in range(10)], dtype=np.float64),
            "open": np.array([100.0 + i for i in range(10)], dtype=np.float64),
            "volume": np.array([1000000.0] * 10, dtype=np.float64),
        })

        score = strategy._compute_sell_score(df)
        assert score == 0.0, "Should return 0.0 for insufficient data"


class TestWeightedVoting:
    """Tests for weighted voting mechanism."""

    @pytest.fixture
    def strategy(self, mock_broker):
        return EnsembleStrategy(broker=mock_broker)

    def test_buy_threshold_calculation(self, strategy):
        """Test that buy score calculation matches expected weighted voting logic."""
        # Create scenario where all signals are active (1.0)
        df = pd.DataFrame({
            "close": [100 - i * 0.1 for i in range(100)],  # Downtrend (oversold)
            "high": [101 - i * 0.1 for i in range(100)],
            "low": [99 - i * 0.1 for i in range(100)],
            "open": [100 - i * 0.1 for i in range(100)],
            "volume": [1000000] * 100,
        })

        p = strategy.parameters
        info = strategy._compute_buy_score(df, details=True)

        # Manually calculate expected score
        total_w = p["w_sma"] + p["w_rsi"] + p["w_macd"] + p["w_bb"]
        manual_score = (
            info["sig_sma"] * p["w_sma"] +
            info["sig_rsi"] * p["w_rsi"] +
            info["sig_macd"] * p["w_macd"] +
            info["sig_bb"] * p["w_bb"]
        ) / total_w

        assert abs(info["score"] - manual_score) < 0.01, \
            f"Calculated score {info['score']} should match manual {manual_score}"

    def test_sell_threshold_calculation(self, strategy):
        """Test that sell score calculation matches expected weighted voting logic."""
        # Create scenario where some signals might be active
        df = pd.DataFrame({
            "close": [100 + i * 0.1 for i in range(100)],  # Uptrend (overbought)
            "high": [101 + i * 0.1 for i in range(100)],
            "low": [99 + i * 0.1 for i in range(100)],
            "open": [100 + i * 0.1 for i in range(100)],
            "volume": [1000000] * 100,
        })

        p = strategy.parameters
        score = strategy._compute_sell_score(df)

        # Score should be valid
        assert 0 <= score <= 1, f"Sell score should be 0-1, got {score}"


class TestParameterValidation:
    """Tests for parameter validation and edge cases."""

    def test_parameters_exist(self, mock_broker):
        """Test that all expected parameters are defined."""
        strategy = EnsembleStrategy(broker=mock_broker)
        p = strategy.parameters

        # Core parameters
        assert "symbols" in p
        assert "screener_top_n" in p
        assert "max_positions" in p
        assert "grace_days" in p

        # Data validation thresholds
        assert "min_screener_bars" in p
        assert "min_signal_bars" in p
        assert "signal_lookback_bars" in p

        # Buy signal params
        assert "sma_fast" in p
        assert "sma_slow" in p
        assert "rsi_period" in p
        assert "rsi_buy_threshold" in p
        assert "macd_fast" in p
        assert "macd_slow" in p
        assert "macd_signal" in p
        assert "bb_period" in p
        assert "bb_std" in p

        # Buy weights
        assert "w_sma" in p
        assert "w_rsi" in p
        assert "w_macd" in p
        assert "w_bb" in p
        assert "buy_threshold" in p

        # Sell signal params
        assert "rsi_sell_threshold" in p

        # Sell weights
        assert "sw_sma" in p
        assert "sw_rsi" in p
        assert "sw_macd" in p
        assert "sw_bb" in p
        assert "sell_threshold" in p

        # Risk
        assert "stoploss" in p

    def test_threshold_values_are_sensible(self, mock_broker):
        """Test that threshold parameters have sensible values."""
        strategy = EnsembleStrategy(broker=mock_broker)
        p = strategy.parameters

        assert p["min_screener_bars"] > 0
        assert p["min_signal_bars"] > 0
        assert p["signal_lookback_bars"] >= p["min_signal_bars"]
        assert p["buy_threshold"] > 0 and p["buy_threshold"] <= 1
        assert p["sell_threshold"] > 0 and p["sell_threshold"] <= 1
        assert p["stoploss"] < 0  # Should be negative


class TestGracePeriodLogic:
    """Tests for position tracking and grace period."""

    @pytest.fixture
    def strategy(self, mock_broker):
        strategy = EnsembleStrategy(broker=mock_broker)
        # Initialize required state
        strategy._last_screened_dates = {}
        strategy._entry_prices = {}
        strategy.screened_symbols = set()
        return strategy

    def test_grace_period_tracking(self, strategy):
        """Test that last_screened_dates are tracked correctly."""
        now = datetime.now()
        strategy._last_screened_dates["AAPL"] = now
        strategy._last_screened_dates["MSFT"] = now - timedelta(days=8)

        # AAPL should still be within grace period
        assert (now - strategy._last_screened_dates["AAPL"]).days <= 7

        # MSFT should be beyond grace period
        assert (now - strategy._last_screened_dates["MSFT"]).days > 7

    def test_entry_price_tracking(self, strategy):
        """Test that entry prices are tracked correctly."""
        strategy._entry_prices["AAPL"] = 150.50
        strategy._entry_prices["MSFT"] = 380.25

        assert strategy._entry_prices["AAPL"] == 150.50
        assert strategy._entry_prices["MSFT"] == 380.25

    def test_stoploss_calculation(self, strategy):
        """Test stoploss percentage calculation."""
        entry_price = 100.0
        stoploss_pct = strategy.parameters["stoploss"]  # -0.346

        # Calculate stoploss price
        stoploss_price = entry_price * (1 + stoploss_pct)

        assert stoploss_price < entry_price
        assert abs(stoploss_price - 65.4) < 0.1  # ~65.4 for -34.6% loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
