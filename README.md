# Lumibot Ensemble Strategy

Automated US stock trading strategy using [Lumibot](https://github.com/Lumibot-Org/lumibot) with Interactive Brokers.

Ported from a Freqtrade EnsembleVoteStrategy with hyperopt-optimized parameters (200 epochs, Sharpe loss function).

## What It Does

1. **Screens ~90 stocks daily** — picks the top 10 using momentum, volume, mean reversion, and relative strength vs SPY
2. **Trades using 4 indicators** — SMA, RSI, MACD, and Bollinger Bands vote with weighted scoring to decide buy/sell
3. **Manages risk** — 34.6% stoploss, max 3 positions, 7-day grace period before closing dropped stocks

## Setup

### Prerequisites

- Python 3.10+
- [TA-Lib C library](https://ta-lib.github.io/ta-lib-python/install.html) installed
- For live trading: [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) running

### Install

```bash
cd lumibot-ensemble
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERACTIVE_BROKERS_IP` | TWS/Gateway IP | `127.0.0.1` |
| `INTERACTIVE_BROKERS_PORT` | `7497` paper, `7496` live (TWS) | `7497` |
| `INTERACTIVE_BROKERS_CLIENT_ID` | API client ID | `1` |
| `POLYGON_API_KEY` | For Polygon.io backtesting (optional) | — |
| `MODE` | `BACKTEST` or `LIVE` | `BACKTEST` |

## Usage

### Backtest (Yahoo — free, daily bars)

```bash
python ensemble_strategy.py
```

### Backtest (Polygon — paid, supports hourly bars)

Set in `.env`:
```
BACKTEST_DATA=POLYGON
POLYGON_API_KEY=your_key_here
```

```bash
python ensemble_strategy.py
```

### Live Trading

1. Open TWS or IB Gateway and log in
2. Enable API connections: Edit > Global Configuration > API > Settings > check "Enable ActiveX and Socket Clients"
3. Set in `.env`:
   ```
   MODE=LIVE
   INTERACTIVE_BROKERS_PORT=7497   # 7497=paper, 7496=live
   ```
4. Run:
   ```bash
   python ensemble_strategy.py
   ```

## Project Files

| File | Purpose |
|------|---------|
| `ensemble_strategy.py` | Strategy logic + entry point |
| `.env` | Your configuration (not committed) |
| `.env.example` | Template for `.env` |
| `requirements.txt` | Python dependencies |

## IB Connection

The strategy connects to TWS/IB Gateway running **on your local machine** (`127.0.0.1`). TWS/Gateway handles the actual connection to Interactive Brokers servers over the internet.

```
ensemble_strategy.py  -->  TWS/IB Gateway (localhost)  -->  IB Servers
```

TWS or IB Gateway must be running before you start the strategy in LIVE mode.

tasklist | grep -i python 2>/dev/null || tasklist | findstr -i python ### finds scripts
taskkill /F /IM python.exe  # kill them


taskkill /F /IM python.exe 2>/dev/null; sleep 3; cd "c:/Users/gelur/Desktop/TRADING/HOLY2/lumibot-ensemble" && .venv/Scripts/python ensemble_strategy.py
