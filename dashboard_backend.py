import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Hermes Trading Bot Dashboard", version="2.0.0")

# Project paths
PROJECT_ROOT = Path(__file__).parent
STATE_DIR = PROJECT_ROOT / "state"
TRADING_FILE = STATE_DIR / "trades.jsonl"
PERFORMANCE_FILE = STATE_DIR / "performance.json"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"
LOGS_DIR = STATE_DIR / "logs"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard.html"


def read_json_file_safe(path: Path, default=None):
    """Safely read a JSON file."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_trades_jsonl_safe():
    """Safely read trades from JSONL file."""
    trades = []
    try:
        with open(TRADING_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return trades


@app.get("/", include_in_schema=False)
async def get_dashboard():
    """Serve the dashboard HTML."""
    return HTMLResponse(content=DASHBOARD_FILE.read_text())


@app.get("/api/heartbeat", include_in_schema=False)
def api_heartbeat():
    """Get worker heartbeat status."""
    hb = read_json_file_safe(HEARTBEAT_FILE)
    if hb is None:
        return {
            "worker_status": "unknown",
            "strategy_version": "unknown",
            "closed_trades": 0,
            "last_trade": None,
            "last_reflection": None,
            "current_score": 0.0,
        }
    
    return {
        "worker_status": hb.get("worker_status", "unknown"),
        "strategy_version": hb.get("strategy_version", "unknown"),
        "closed_trades": hb.get("closed_trades", 0),
        "last_trade": hb.get("last_trade"),
        "last_reflection": hb.get("last_reflection"),
        "current_score": hb.get("current_score", 0.0),
    }


@app.get("/api/performance", include_in_schema=False)
def api_performance():
    """Get performance metrics."""
    perf = read_json_file_safe(PERFORMANCE_FILE)
    trades = read_trades_jsonl_safe()
    
    if perf is None:
        return {
            "return_pct": 0,
            "sharpe": 0,
            "sortino": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "trade_count": 0,
            "average_trade": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "consecutive_losses": 0,
            "current_strategy_version": "unknown",
            "current_score": 0.0,
        }
    
    result = {
        "return_pct": perf.get("return_pct", 0),
        "sharpe": perf.get("sharpe", 0),
        "sortino": perf.get("sortino", 0),
        "win_rate": perf.get("win_rate", 0),
        "profit_factor": perf.get("profit_factor", 0),
        "expectancy": perf.get("expectancy", 0),
        "trade_count": perf.get("trade_count", 0),
        "average_trade": perf.get("average_trade", 0),
        "largest_win": perf.get("largest_win", 0),
        "largest_loss": perf.get("largest_loss", 0),
        "consecutive_losses": perf.get("consecutive_losses", 0),
        "current_strategy_version": perf.get("current_strategy_version", "unknown"),
        "current_score": perf.get("current_score", 0.0),
    }
    
    # If performance file is empty/default, calculate from trades
    if result["trade_count"] == 0 and perf is not None:
        trades = read_trades_jsonl_safe()
        result["trade_count"] = len(trades) if trades else 0
    
    return result


@app.get("/api/trades", include_in_schema=False)
def api_trades():
    """Get recent trades."""
    trades = read_trades_jsonl_safe()
    if trades is None:
        trades = []
    # Return last 20 trades
    return trades[-20:]


@app.post("/api/toggle-worker", include_in_schema=False)
def toggle_worker(action: str):
    """Toggle worker start/stop request."""
    # This is a request handler - actual startup is handled by OS services
    return {
        "status": "request_received",
        "message": f"Worker {action} request received. Use system services to start/stop.",
        "action": action,
    }


@app.post("/api/toggle-reflector", include_in_schema=False)
def toggle_reflector(action: str):
    """Toggle reflector start/stop request."""
    return {
        "status": "request_received",
        "message": f"Reflector {action} request received. Use system services to start/stop.",
        "action": action,
    }


@app.get("/api/logs", include_in_schema=False)
def api_logs():
    """Get system logs."""
    # Try errors log first
    errors_log = LOGS_DIR / "errors.log"
    if errors_log.exists():
        try:
            with open(errors_log, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {"logs": content[-3000:]}
        except OSError:
            pass
    
    # Try worker log
    worker_log = STATE_DIR / "logs" / "worker.log"
    if worker_log.exists():
        try:
            with open(worker_log, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {"logs": content[-3000:]}
        except OSError:
            pass
    
    return {"logs": "No logs found"}


# Health check endpoint
@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok", "message": "Hermes Dashboard API is running"}


if __name__ == "__main__":
    # Create required directories
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Hermes Trading Bot Dashboard API (FastAPI)")
    print("=" * 60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"State: {STATE_DIR}")
    print()
    print("Available endpoints:")
    print("  GET  /              - Dashboard HTML")
    print("  GET  /api/heartbeat - Worker status")
    print("  GET  /api/performance - Performance metrics")
    print("  GET  /api/trades - Recent trades")
    print("  POST /api/toggle-worker - Start/stop worker")
    print("  POST /api/toggle-reflector - Start/stop reflector")
    print("  GET  /api/logs - System logs")
    print("  GET  /health - Health check")
    print()
    print("Starting server on http://127.0.0.1:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)