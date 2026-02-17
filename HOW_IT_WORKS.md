# How the Ensemble Strategy Works

## The Big Picture

Every day, the bot looks at ~90 US stocks and picks the best 10. Then every hour, it checks those 10 stocks using 4 different technical indicators. If enough indicators agree "buy", it buys. If enough agree "sell", it sells. It holds a maximum of 3 stocks at a time.

## Step 1: The Daily Screener

Before the market opens each day, the bot scores all ~90 stocks on 4 factors:

| Factor | What it measures |
|--------|-----------------|
| **Momentum** | How much the price went up over the last 20 days |
| **Volume surge** | Is recent volume (5-day) higher than normal (20-day)? |
| **Mean reversion** | Is the price below its 20-day average? (potential bounce) |
| **Relative strength** | Is this stock doing better than SPY (the market)? |

Each stock gets a percentile rank for each factor (0-100%), then the 4 ranks are averaged into a composite score. The top 10 make the cut.

## Step 2: Buy Decisions

For each of the top 10 stocks (that the bot doesn't already own), it checks 4 indicators:

| Indicator | Buy signal fires when... | Weight |
|-----------|-------------------------|--------|
| **SMA** (Simple Moving Average) | 21-day SMA is above 39-day SMA (uptrend) | 0.15 (low) |
| **RSI** (Relative Strength Index) | RSI is below 36 (oversold = potential bounce) | 0.82 (high) |
| **MACD** | MACD line is above its signal line (bullish momentum) | 0.48 (medium) |
| **Bollinger Bands** | Price is below the lower band (oversold) | 0.82 (high) |

Each indicator votes 1 (yes) or 0 (no). The votes are multiplied by their weights and averaged. If the final score is **0.64 or higher**, the bot buys.

**Why different weights?** The parameters were optimized by running 200 rounds of automated testing (hyperopt). RSI and Bollinger Bands turned out to be the strongest buy signals, while SMA was less useful for entries.

## Step 3: Sell Decisions

For each stock the bot owns, it checks the same 4 indicators but looking for bearish signals:

| Indicator | Sell signal fires when... | Weight |
|-----------|--------------------------|--------|
| **SMA** | 21-day SMA is below 39-day SMA (downtrend) | 0.75 |
| **RSI** | RSI is above 67 (overbought) | 0.90 |
| **MACD** | MACD line is below its signal line (bearish) | 0.66 |
| **Bollinger Bands** | Price is above the upper band (overbought) | 1.00 |

If the weighted score is **0.65 or higher**, the bot sells.

## Step 4: Safety Rules

Three additional rules protect the portfolio:

### Stoploss (-34.6%)
If a stock drops more than 34.6% from the buy price, sell immediately. This is a wide stoploss — the strategy is designed to hold through moderate dips.

### Grace Period (7 days)
If a stock drops out of the top 10 screener list, the bot doesn't sell right away. It waits 7 days. If the stock comes back to the top 10 within that time, it stays. If not, it gets sold.

### Max 3 Positions
The bot never holds more than 3 stocks at once. Each position gets roughly 1/3 of the portfolio.

## The Flow (Visual)

```
Every morning:
  Score 90 stocks --> Pick top 10

Every hour (during market hours):
  For each stock I OWN:
    - Off the top 10 list for > 7 days?  --> Sell
    - Down more than 34.6%?              --> Sell
    - Sell indicators score >= 0.65?      --> Sell

  For each stock in the TOP 10 that I DON'T own:
    - Already holding 3 stocks?           --> Skip
    - Buy indicators score >= 0.64?       --> Buy
```

## The 90 Stocks

The universe covers 8 sectors to stay diversified:

- **ETFs** (20): SPY, QQQ, IWM, sector ETFs, etc.
- **Tech** (15): AAPL, MSFT, NVDA, GOOGL, AMZN, etc.
- **Finance** (9): JPM, GS, BAC, etc.
- **Healthcare** (10): UNH, JNJ, LLY, etc.
- **Consumer** (10): WMT, COST, HD, etc.
- **Industrial** (10): CAT, BA, HON, etc.
- **Energy** (10): XOM, CVX, SLB, etc.
- **Other** (10): NEE, DIS, NFLX, etc.

## Where the Numbers Come From

All the specific values (SMA periods, RSI thresholds, weights, etc.) were found by **hyperopt optimization** — an automated process that tested thousands of parameter combinations against 3 years of historical data and picked the combination with the best risk-adjusted returns (Sharpe ratio). The optimization ran 200 epochs and found a configuration that returned +89% in backtesting with the original hourly timeframe.
