# Trading Bot Strategy Documentation

## 1. Overview

This repo implements an hourly long-only trading bot for Trading212. The strategy uses per-symbol machine learning models to decide when to enter long positions in a small set of equities/ETFs. It executes market buy orders when a `LONG` signal is generated and closes positions when the signal flips to `FLAT`.

Key design goals: 
- limit the number of concurrent positions,
- use a weekly retraining cadence,
- apply per-symbol risk sizing,
- enforce a daily drawdown kill switch,
- keep the model features normalized so they generalize across symbols.

## 2. Architecture

The main components are:

- `run.py`
  - entry point and scheduler
  - runs one trade cycle immediately, then sleeps until the next candle close

- `bot.py`
  - trading loop and risk management
  - fetches account cash and portfolio
  - computes signals and decides buys/sells
  - triggers weekly retraining and kill switch events

- `t212_client.py`
  - REST client wrapper for Trading212
  - handles account queries, portfolio queries, market orders, and position closing

- `strategy.py`
  - converts model inference into a trading signal

- `model.py`
  - trains and saves per-symbol RandomForest models
  - loads models and computes predictions

- `data_feed.py`
  - downloads OHLCV data from Yahoo Finance

- `features.py`
  - builds normalized technical features for the ML model

- `risk_manager.py`
  - rule-based position sizing, open position counting, and drawdown protection

- `notifier.py`
  - sends Telegram alerts for key events

## 3. Trading logic

### 3.1 Cycle flow

Each hourly cycle in `bot.py`:

1. checks for an active kill switch
2. verifies US market hours using `risk_manager.is_market_open()`
3. potentially retrains models once per week on Sunday
4. fetches current cash and open positions
5. calculates open P&L for the tracked symbols
6. checks whether the daily loss threshold was exceeded
7. iterates through configured symbols and uses model signals
8. opens new long positions when a `LONG` signal appears and position limits allow
9. closes existing positions when the signal becomes `FLAT`

### 3.2 Signal decisions

A symbol is processed as follows:

- fetch OHLCV data from Yahoo Finance
- compute a model signal and confidence via `strategy.compute_signal()`
- if the symbol is not already held and the signal is `LONG`:
  - require fewer than `cfg.MAX_OPEN_POSITIONS` open positions
  - calculate quantity with `risk_manager.position_size()`
  - submit a market buy
- if the symbol is already held and the signal is `FLAT`:
  - close the existing position with a market sell

## 4. Data and feature engineering

The model works on normalized technical features built from hourly OHLCV history.

### 4.1 Input data

- Source: Yahoo Finance via `yfinance`
- Interval: `cfg.INTERVAL` (default `1h`)
- Bars: recent history from `get_ohlcv()`
- The latest incomplete candle is dropped to avoid using partial bars.

### 4.2 Feature set in `features.py`

The features are designed to be ratio-based and scale-invariant.

1. EMA ratio features
   - `close_to_ema20 = close / EMA(20) - 1`
   - `ema20_to_ema50 = EMA(20) / EMA(50) - 1`
   - `close_to_ema200 = close / EMA(200) - 1`

   These features capture trend alignment and crossovers.

2. RSI features
   - `rsi_7 = RSI(7) / 100`
   - `rsi_14 = RSI(14) / 100`

   RSI is computed with exponential moving averages of gains and losses:
   - `delta = close_t - close_{t-1}`
   - `gain = max(delta, 0)`
   - `loss = max(-delta, 0)`
   - `avg_gain = EMA(gain, period)`
   - `avg_loss = EMA(loss, period)`
   - `RS = avg_gain / avg_loss`
   - `RSI = 100 - 100 / (1 + RS)`

3. Volatility feature
   - `atr_14 = ATR(14) / close`
   - ATR uses true range and EMA smoothing:
     - `TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)`
     - `ATR_t = EMA(TR, 14)`
   - Normalizing by close produces a scale-free volatility measure.

4. Volume feature
   - `volume_ratio = volume / volume_ma20`
   - captures whether the current bar has above-average volume.

5. Momentum features
   - `mom_1h = close / close.shift(1) - 1`
   - `mom_4h = close / close.shift(4) - 1`
   - `mom_12h = close / close.shift(12) - 1`
   - `mom_24h = close / close.shift(24) - 1`

   These are simple returns over multiple lookback windows.

6. Bollinger Band position
   - `bb_pos = (close - (SMA(20) - 2*std(20))) / ((SMA(20) + 2*std(20)) - (SMA(20) - 2*std(20)))`
   - maps price position inside the Bollinger band range to `[0, 1]`.

7. Time-of-day features
   - `hour_sin`, `hour_cos` encode hour of day as a cyclic feature
   - `dow_sin`, `dow_cos` encode day-of-week cyclicity

### 4.3 Feature matrix properties

- The feature matrix drops rows with NaN values, which occur due to indicator warmup.
- Most features are expressed as ratios or normalized scalars, allowing use across different price ranges.

## 5. Machine learning model

### 5.1 Target definition

The ML problem is framed as a binary classification:

- Input: features computed from the current bar history.
- Target label: whether the price rises by at least `TARGET_RETURN` over the next `FORWARD_HOURS` candles.

Formally:

- `forward_ret_t = close_{t+FORWARD_HOURS} / close_t - 1`
- `y_t = 1` if `forward_ret_t > TARGET_RETURN`, else `0`

With defaults:
- `FORWARD_HOURS = 4`
- `TARGET_RETURN = 0.003` (0.3%)

Thus the model predicts whether the symbol is likely to gain at least 0.3% over the next 4 hourly bars.

### 5.2 Training process

In `model.py`:

1. call `build_features(df)` to compute features from the historical DataFrame.
2. compute the forward-looking target label using shifted close prices.
3. align features and labels, dropping rows where target is unavailable.
4. split data in time order:
   - first 70% of rows for training
   - final 30% of rows for testing
5. train a `RandomForestClassifier` with:
   - `n_estimators=200`
   - `max_depth=6`
   - `min_samples_leaf=20`
   - `class_weight='balanced'`
   - `random_state=42`
   - `n_jobs=-1`
6. save pickled model data including feature columns.

### 5.3 Inference and signal logic

In `model.predict()`:

- compute the latest feature row from fresh market data
- use the saved model to get class probabilities
- use the probability of the positive class (`LONG`) as `confidence`
- return `signal = 1` if `confidence >= cfg.ML_CONFIDENCE_THRESHOLD`, else `0`

`strategy.compute_signal()` converts the numeric output into:
- `Signal.LONG` when model favors the positive class above the threshold
- `Signal.FLAT` otherwise

The configured threshold is `cfg.ML_CONFIDENCE_THRESHOLD = 0.55` by default.

## 6. Execution and risk management

### 6.1 Position sizing

The bot sizes each trade by fraction of available cash:

- `qty = floor((cash * risk_fraction / price) * 10) / 10`

This means:
- `risk_fraction` is the fraction of available cash to deploy
- the result is rounded down to one decimal place
- example: if cash is $10,000, price is $400, and risk fraction is 0.01,
  then `qty = floor((10000 * 0.01 / 400) * 10) / 10 = 2.5` shares.

### 6.2 Exposure limits

Configured limits:
- `MAX_OPEN_POSITIONS = 2`
- `MAX_DAILY_LOSS = 0.03` (3% daily drawdown)

The bot also counts only positions in the tracked symbols when enforcing the open position limit.

### 6.3 Drawdown kill switch

The kill switch is triggered when:
- `drawdown >= MAX_DAILY_LOSS`
- drawdown is computed as:
  `drawdown = (start_cash - (current_cash + open_pnl)) / start_cash`

When triggered, the bot closes all tracked-symbol positions and sets `state.killed = True` so subsequent cycles do not trade.

### 6.4 Market hours

Trading occurs only during US market hours in New York time:
- open window: 09:35 ET
- close window: 15:50 ET
- weekends are skipped

## 7. Weekly retraining

The bot checks weekly retrain conditions in `bot._maybe_retrain()`:
- retrain only on `cfg.RETRAIN_DAY = 6` (Sunday)
- if the current ISO week differs from `last_retrain_week`
- then it imports `train.main()` and runs training

This means models are refreshed once per calendar week when the bot is running on Sundays.

## 8. Mathematical model summary

### 8.1 Feature equations

Let `C_t`, `H_t`, `L_t`, `V_t` denote the close, high, low, and volume at time `t`.

EMA:
- `EMA_{t}^{(n)} = rac{C_t + (n-1) \, EMA_{t-1}^{(n)}}{n+1}` using exponential weighting

EMA ratios:
- `close_to_ema20_t = C_t / EMA_{t}^{(20)} - 1`
- `ema20_to_ema50_t = EMA_{t}^{(20)} / EMA_{t}^{(50)} - 1`
- `close_to_ema200_t = C_t / EMA_{t}^{(200)} - 1`

RSI:
- `
  \Delta_t = C_t - C_{t-1}`
- `gain_t = max(\Delta_t, 0)`
- `loss_t = max(-\Delta_t, 0)`
- `avg_gain_t = EMA(gain_t, period)`
- `avg_loss_t = EMA(loss_t, period)`
- `RS_t = avg_gain_t / avg_loss_t`
- `RSI_t = 100 - 100 / (1 + RS_t)`
- normalized: `rsi_period_t = RSI_t / 100`
`

ATR:
- `TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)`
- `ATR_t = EMA(TR_t, 14)`
- normalized: `atr_14_t = ATR_t / C_t`

Momentum returns:
- `mom_kh_t = C_t / C_{t-k} - 1`
  for k = 1,4,12,24

Bollinger position:
- `SMA_{20,t} = mean(C_{t-19} ... C_t)`
- `std_{20,t} = std(C_{t-19} ... C_t)`
- `upper_t = SMA_{20,t} + 2 * std_{20,t}`
- `lower_t = SMA_{20,t} - 2 * std_{20,t}`
- `bb_pos_t = (C_t - lower_t) / (upper_t - lower_t)`

Cyclical time encoding:
- `hour_sin_t = sin(2π hour_t / 24)`
- `hour_cos_t = cos(2π hour_t / 24)`
- `dow_sin_t = sin(2π dow_t / 5)`
- `dow_cos_t = cos(2π dow_t / 5)`

### 8.2 Model formulation

The model is a Random Forest classifier. It learns a mapping:

- `f(X_t) -> P(y_t = 1)`

where `X_t` is the feature vector at time `t` and `y_t` is the binary target for future return.

The classifier is trained on time-ordered examples using a 70/30 split so there is no information leakage from future bars into training.

During inference:
- calculate `p = f(X_t)[LONG]`
- signal `LONG` if `p >= threshold`, otherwise `FLAT`

The threshold is set to 0.55, meaning the model must be at least 55% confident that the next 4h return will exceed 0.3%.

## 9. Personal opinion

### 9.1 Strengths

- The architecture is clear and modular.
- Feature engineering is sensible for trend/momentum-based market behavior.
- Normalized ratios and cyclical time features make the model more robust across different symbols.
- Weekly retraining is a good practice for adapting to new market conditions.
- Built-in risk controls such as max positions and daily drawdown are valuable for protection.

### 9.2 Weaknesses and concerns

- The strategy is very simple and may struggle in sideways or choppy markets.
- The target label is binary and fixed at a 0.3% gain over 4 hours; that may not align with actual realized transaction costs and slippage.
- Market buy/sell decisions are made exclusively on model signals without additional exit logic or stop losses.
- The Random Forest model does not explicitly capture sequential dependencies beyond the feature window; it is still a static classifier on engineered features.
- `yfinance` data quality and latency can be unreliable for live execution decisions.
- The bot ignores position-specific risk beyond allocated cash and does not use a trailing stop or volatility-adjusted exit.

### 9.3 Practical advice

- Validate the model with a holdout or walk-forward backtest before live trading.
- Measure transaction costs, slippage, and overnight exposure.
- Consider adding a stop-loss or take-profit rule rather than fully relying on FLAT exit signals.
- Use the Telegram alerts carefully, and make sure the bot has robust error handling around the Trading212 API.
- Keep `MAX_OPEN_POSITIONS` low, as the strategy is designed for a limited number of concurrent trades.

### 9.4 Conclusion

This is a reasonable proof-of-concept system for a small hourly long-only ML trading bot. The strategy is transparent and modular, but it should not be treated as a production-ready black-box model without additional validation, stress testing, and more robust exit/risk management.
