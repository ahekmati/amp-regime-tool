# AMP Regime Confirmation & C2 Auto-Trader

This tool:
- Scrapes AMP iSystems top ES/NQ systems,
- Interprets the best ES/NQ signal (long/short),
- Confirms with SPY/QQQ composite regime (trend, ML, Monte Carlo, macro),
- Tracks last regime change and age,
- Optionally sends live MES/MNQ orders to Collective2 when >=75% of models agree.

### Usage

```bash
export C2_API_KEY="..."
export C2_SYSTEM_ID=...
export AUTO_TRADE_ENABLED=true
export DRY_RUN=1
python amp_regime_confirmation.py
```
