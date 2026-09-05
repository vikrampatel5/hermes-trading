"""Hermes Trading Bot - Professional Dashboard Backend (FastAPI).

Exposes a complete REST API for the multi-pair trading engine:
  - /api/health                      — liveness probe (Render keep-alive hits this)
  - /api/heartbeat                   — worker status + last keepalive
  - /api/performance                 — aggregate performance metrics
  - /api/trades                      — recent trade history
  - /api/equity-curve                — per-tick equity time series
  - /api/pairs                       — current state of every pair in the universe
  - /api/pair/{symbol}               — single-pair detail (candles + state + recent trades)
  - /api/signals                     — last N signal events across all pairs
  - /api/wallet                      — Delta wallet balances
  - /api/positions                   — live positions from Delta
  - /api/universe                    — the trading universe metadata
  - /api/strategy                    — current strategy.yaml + history
  - /api/hypotheses                  — brain reflection log
  - /api/logs                        — tail of worker error log

All endpoints are no-auth and read-only by design — the dashboard does not
mutate trading state. To control the worker, use systemd / Render's dashboard.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add project root to path so we can import hermes_trading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root so wallet/positions endpoints can authenticate
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

app = FastAPI(title="Hermes Trading Bot Dashboard", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).parent
STATE_DIR = PROJECT_ROOT / "state"
TRADING_FILE = STATE_DIR / "trades.jsonl"
PERFORMANCE_FILE = STATE_DIR / "performance.json"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"
LOGS_DIR = STATE_DIR / "logs"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE = STATE_DIR / "goal.yaml"
UNIVERSE_FILE = STATE_DIR / "delta_universe.json"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard.html"
HISTORY_DIR = STATE_DIR / "history"


# ---------- helpers ----------

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def _read_text(path: Path, max_bytes: int = 5000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read()
    except Exception:
        return ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _list_pair_states() -> List[Dict[str, Any]]:
    """Read every state/pair_*.json file."""
    out = []
    for sf in sorted(STATE_DIR.glob("pair_*.json")):
        s = _read_json(sf)
        if s:
            s["_file"] = sf.name
            out.append(s)
    return out


def _compute_metrics_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback metrics when performance.json is empty or stale."""
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "average_pnl_pct": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_losses": 0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
        }
    pnls = [float(t.get("pnl_pct", t.get("pnl", 0)) or 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
    total = sum(pnls)
    avg = total / len(pnls) if pnls else 0.0
    # Consecutive losses at the tail
    consec = 0
    for p in reversed(pnls):
        if p < 0:
            consec += 1
        else:
            break
    # Sharpe
    if len(pnls) > 1:
        mean = avg
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = var ** 0.5
        sharpe = (mean / std) * (len(pnls) ** 0.5) if std > 0 else 0.0
    else:
        sharpe = 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (sum(wins) if wins else 0.0)
    return {
        "trade_count": len(pnls),
        "win_rate": round(win_rate, 2),
        "total_pnl_pct": round(total, 4),
        "average_pnl_pct": round(avg, 4),
        "largest_win": round(max(pnls), 4) if pnls else 0.0,
        "largest_loss": round(min(pnls), 4) if pnls else 0.0,
        "consecutive_losses": consec,
        "sharpe": round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
    }


def _build_equity_curve(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cumulative PnL curve from trade history."""
    out = []
    cum = 0.0
    for t in trades:
        pnl = float(t.get("pnl_pct", t.get("pnl", 0)) or 0)
        cum += pnl
        out.append({
            "timestamp": t.get("timestamp") or t.get("time") or "",
            "symbol": t.get("symbol", ""),
            "pnl_pct": round(pnl, 4),
            "cumulative_pnl_pct": round(cum, 4),
        })
    return out


# ---------- endpoints ----------

@app.get("/", include_in_schema=False)
async def get_dashboard():
    if DASHBOARD_FILE.exists():
        return HTMLResponse(content=DASHBOARD_FILE.read_text())
    return HTMLResponse(content="<h1>Dashboard file missing</h1>", status_code=500)


@app.get("/health", include_in_schema=False)
def health_check():
    """Liveness probe — also writes heartbeat.json with last_keepalive so the
    Render keep-alive cron can verify the worker is alive."""
    now = datetime.now(timezone.utc).isoformat()
    hb = _read_json(HEARTBEAT_FILE, default={}) or {}
    hb["last_keepalive"] = now
    hb["keepalive_source"] = "health_endpoint"
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump(hb, f, indent=2)
    except Exception:
        pass
    return {"status": "ok", "service": "hermes-dashboard", "timestamp": now}


@app.get("/api/heartbeat", include_in_schema=False)
def api_heartbeat():
    hb = _read_json(HEARTBEAT_FILE, default={}) or {}
    age_seconds = None
    last_ts = hb.get("last_keepalive") or hb.get("timestamp")
    if last_ts:
        try:
            ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            pass
    return {
        "worker_status": hb.get("worker_status", "unknown"),
        "strategy_version": hb.get("strategy_version", "?"),
        "closed_trades": hb.get("closed_trades", 0),
        "last_trade": hb.get("last_trade"),
        "last_reflection": hb.get("last_reflection"),
        "current_score": hb.get("current_score", 0.0),
        "last_keepalive": hb.get("last_keepalive"),
        "keepalive_source": hb.get("keepalive_source"),
        "keepalive_age_seconds": age_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/performance", include_in_schema=False)
def api_performance():
    perf = _read_json(PERFORMANCE_FILE, default={}) or {}
    trades = _read_jsonl(TRADING_FILE)
    computed = _compute_metrics_from_trades(trades)
    # Merge: prefer explicit perf file values, fall back to computed
    merged = {**computed, **{
        k: perf.get(k) for k in [
            "return_pct", "sharpe", "sortino", "win_rate", "profit_factor",
            "expectancy", "trade_count", "average_trade", "largest_win",
            "largest_loss", "consecutive_losses", "current_strategy_version",
            "current_score", "maximum_drawdown",
        ] if perf.get(k) is not None
    }}
    if not merged.get("current_strategy_version"):
        strat = _read_json(STRATEGY_FILE, default={}) or {}
        merged["current_strategy_version"] = strat.get("version", "?")
    return merged


@app.get("/api/trades", include_in_schema=False)
def api_trades(limit: int = Query(50, ge=1, le=500)):
    trades = _read_jsonl(TRADING_FILE)
    return trades[-limit:][::-1]  # newest first


@app.get("/api/equity-curve", include_in_schema=False)
def api_equity_curve(limit: int = Query(200, ge=1, le=2000)):
    trades = _read_jsonl(TRADING_FILE)
    return _build_equity_curve(trades[-limit:])


@app.get("/api/pairs", include_in_schema=False)
def api_pairs():
    """All pair states + the universe metadata merged together."""
    states = {s["symbol"]: s for s in _list_pair_states()}
    universe = _read_json(UNIVERSE_FILE, default={}) or {}
    universe_list = universe.get("universe", [])
    rows = []
    for u in universe_list:
        sym = u.get("symbol", "?")
        s = states.get(sym, {})
        rows.append({
            "rank": u.get("volume_24h", 0),
            "symbol": sym,
            "product_id": u.get("product_id"),
            "underlying": u.get("underlying"),
            "mark_price": u.get("mark_price", 0),
            "volume_24h": u.get("volume_24h", 0),
            "max_leverage": u.get("max_leverage"),
            "trading_leverage": u.get("trading_leverage") or s.get("trading_leverage"),
            "tick_size": u.get("tick_size"),
            "contract_value": u.get("contract_value"),
            "position_size": s.get("position_size", 0),
            "position_side": ("long" if (s.get("position_size") or 0) > 0
                              else "short" if (s.get("position_size") or 0) < 0
                              else "flat"),
            "entry_price": s.get("entry_price", 0),
            "last_signal": s.get("last_signal", "none"),
            "last_keepalive_signal": s.get("last_keepalive_signal", "none"),
            "last_signal_time": s.get("last_signal_time", ""),
            "recent_trade_count": len(s.get("recent_trades", [])),
        })
    # Sort by volume desc
    rows.sort(key=lambda r: r.get("rank") or 0, reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "pairs": rows,
        "universe_size": len(rows),
        "last_updated": universe.get("last_updated"),
    }


@app.get("/api/pair/{symbol}", include_in_schema=False)
def api_pair_detail(symbol: str):
    """Single-pair detail: state + recent trades."""
    sf = STATE_DIR / f"pair_{symbol}.json"
    if not sf.exists():
        raise HTTPException(status_code=404, detail=f"no state for {symbol}")
    state = _read_json(sf)
    if not state:
        raise HTTPException(status_code=404, detail=f"unreadable state for {symbol}")
    return state


@app.get("/api/signals", include_in_schema=False)
def api_signals(limit: int = Query(30, ge=1, le=200)):
    """Recent signal events derived from per-pair state files."""
    events = []
    for s in _list_pair_states():
        ts = s.get("last_signal_time")
        if ts and s.get("last_signal", "none") != "none":
            events.append({
                "timestamp": ts,
                "symbol": s.get("symbol"),
                "product_id": s.get("product_id"),
                "signal": s.get("last_signal"),
                "position_size": s.get("position_size", 0),
                "entry_price": s.get("entry_price", 0),
                "leverage": s.get("trading_leverage"),
            })
    # Also include signal-time events from per-pair recent_trades (closed trades)
    for s in _list_pair_states():
        for t in (s.get("recent_trades") or [])[-3:]:
            events.append({
                "timestamp": t.get("timestamp"),
                "symbol": s.get("symbol"),
                "product_id": s.get("product_id"),
                "signal": f"close:{t.get('reason','?')}",
                "pnl_pct": t.get("pnl_pct"),
                "leverage": t.get("leverage") or s.get("trading_leverage"),
            })
    # Newest first
    events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    return events[:limit]


@app.get("/api/wallet", include_in_schema=False)
def api_wallet():
    """Live wallet from Delta (requires API key in .env)."""
    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")
    mode = os.getenv("DELTA_MODE", "testnet")
    if not api_key or not api_secret:
        return {"status": "no_credentials", "balances": {}, "net_equity": 0}
    try:
        from hermes_trading.adapters.delta import get_wallet_balances_sync
        w = get_wallet_balances_sync(mode=mode, api_key=api_key, api_secret=api_secret)
        return w
    except Exception as e:
        return {"status": "error", "error": str(e), "balances": {}, "net_equity": 0}


@app.get("/api/positions", include_in_schema=False)
def api_positions():
    """Live positions from Delta (BTC + ETH only on testnet)."""
    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")
    mode = os.getenv("DELTA_MODE", "testnet")
    if not api_key or not api_secret:
        return {"status": "no_credentials", "positions": []}
    out = []
    for underlying in ["BTC", "ETH"]:
        try:
            from hermes_trading.adapters.delta import get_positions_sync
            r = get_positions_sync(
                symbol=f"{underlying}USD", mode=mode,
                api_key=api_key, api_secret=api_secret,
            )
            for p in r.get("positions", []):
                p["_queried_underlying"] = underlying
                out.append(p)
        except Exception as e:
            out.append({"_queried_underlying": underlying, "_error": str(e)})
    return {"status": "ok", "positions": out, "count": len(out)}


@app.get("/api/universe", include_in_schema=False)
def api_universe():
    return _read_json(UNIVERSE_FILE, default={"universe": [], "total": 0})


@app.get("/api/strategy", include_in_schema=False)
def api_strategy():
    """Current strategy + recent reflection history."""
    current = _read_json(STRATEGY_FILE, default={}) or {}
    history = []
    if HISTORY_DIR.exists():
        for h in sorted(HISTORY_DIR.glob("v*.yaml")):
            history.append({
                "version": h.stem,
                "path": h.name,
                "size": h.stat().st_size,
            })
    history.sort(key=lambda h: h["version"], reverse=True)
    return {
        "current": current,
        "history_versions": [h["version"] for h in history],
        "history_count": len(history),
    }


@app.get("/api/hypotheses", include_in_schema=False)
def api_hypotheses(limit: int = Query(50, ge=1, le=500)):
    return _read_jsonl(HYPOTHESES_FILE)[-limit:][::-1]


@app.get("/api/logs", include_in_schema=False)
def api_logs(max_bytes: int = Query(8000, ge=100, le=50000)):
    candidates = [
        LOGS_DIR / "worker_errors.log",
        LOGS_DIR / "errors.log",
        LOGS_DIR / "worker.log",
    ]
    for c in candidates:
        if c.exists():
            return {"log_file": c.name, "content": _read_text(c, max_bytes)}
    return {"log_file": None, "content": "No logs found"}


@app.get("/api/overview", include_in_schema=False)
def api_overview():
    """One-shot summary call for the dashboard's top stats card."""
    perf = api_performance()
    pairs = api_pairs()
    wallet = api_wallet()
    positions = api_positions()
    signals = api_signals(limit=5)
    hb = api_heartbeat()

    universe = pairs.get("pairs", [])
    open_positions = [p for p in universe if p.get("position_size")]
    long_count = sum(1 for p in open_positions if (p.get("position_size") or 0) > 0)
    short_count = sum(1 for p in open_positions if (p.get("position_size") or 0) < 0)

    return {
        "performance": perf,
        "wallet": wallet,
        "universe_size": pairs.get("universe_size", 0),
        "open_positions": len(open_positions),
        "long_positions": long_count,
        "short_positions": short_count,
        "flat_pairs": pairs.get("universe_size", 0) - len(open_positions),
        "live_positions": positions.get("count", 0),
        "live_positions_list": positions.get("positions", []),
        "recent_signals": signals,
        "heartbeat": hb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Hermes Trading Bot Dashboard API v3.0")
    print("=" * 60)
    print(f"Endpoints:")
    print(f"  GET  /                       - Dashboard HTML")
    print(f"  GET  /health                 - Liveness (writes heartbeat)")
    print(f"  GET  /api/overview           - Top-card summary")
    print(f"  GET  /api/heartbeat          - Worker status")
    print(f"  GET  /api/performance        - Aggregate metrics")
    print(f"  GET  /api/trades             - Recent trades")
    print(f"  GET  /api/equity-curve       - PnL time series")
    print(f"  GET  /api/pairs              - All pair states")
    print(f"  GET  /api/pair/{{symbol}}      - Single pair detail")
    print(f"  GET  /api/signals            - Recent signal events")
    print(f"  GET  /api/wallet             - Delta wallet")
    print(f"  GET  /api/positions          - Live positions (BTC/ETH on testnet)")
    print(f"  GET  /api/universe           - Trading universe metadata")
    print(f"  GET  /api/strategy           - Current + history")
    print(f"  GET  /api/hypotheses         - Brain reflection log")
    print(f"  GET  /api/logs               - Worker log tail")
    print()
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    print(f"Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)