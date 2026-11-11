# Financial & Sequential Time-Series Guardrails

Disclosed domain reference for `dream-rsi`. Load whenever evaluating market data, OHLCV bars, orderbooks, or sequential forecasting.

> **Mandatory Risk Warning:** Always display a clear capital-risk disclaimer alongside any trading metric or forecast. Historical `Walk-Forward` and `Blind Vault` profitability never guarantees future live returns.

## 1. Zero Look-Ahead Causality Audit
- **Higher-Timeframe Joins:** Join macro bars (`1d`, `4h`, `1h`) to micro bars (`15m`, `5m`, `1m`) strictly on **`htf_close_time <= ltf_close_time`**. Assert `lookahead_leak_violations == 0` before scoring any candidate.
- **Purged Walk-Forward Embargo:** Insert an explicit `h_bars` embargo equal to the maximum holding horizon between training and validation blocks inside `Dev`.
- **Sealed Chronological `Blind Vault`:** Hold out the final chronological tail (e.g. last `20%` of history) behind `BlindVaultGuard(..., max_calls=1)`. In multi-timeframe continual sleep, order runs **Macro -> Micro** (`1d -> 4h -> 1h -> 15m -> 5m`) so earlier training spans never overlap a later timeframe's `Blind Vault`.

## 2. Phase-Invariant Event Selection & Friction Accounting
- **Event Cooldowns Over Modulo Grids:** Never subsample entries via `df.iloc[::step]` (creates phase-offset luck). Trigger entries on explicit structural events with a non-overlapping `cooldown >= horizon_bars`.
- **Fee-to-ATR Barrier Gate:** Always deduct realistic round-trip taker fees + slippage (`4.5–6.0 bps`) inside `verify_and_score`. Track the ratio `round_trip_cost / median_ATR`: when compressed volatility pushes friction above `~25–35%` of `ATR`, require an explicit `WAIT / NEUTRAL` regime filter rather than forcing trades.
- **Report Blind Vault Losses Honestly:** Never re-tune parameters after opening the `Blind Vault`; report compressed-volatility regime losses alongside winning swing horizons.
