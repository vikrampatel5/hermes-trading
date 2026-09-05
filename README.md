# Hermes Trading Bot

Local self-improving trading bot running 24/7 in paper mode.

## Structure

- `hermes_trading/` — Python package with all trading logic
- `state/` — Persistent state files
- `services/` — OS service definitions

## Quick Start

```bash
cd ~/hermes-trading-bot
pip install -e .
python -m hermes_trading.run
```