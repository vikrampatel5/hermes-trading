"""Hermes Trading Bot - Delta Exchange India Adapter (Testnet/Live).

Implements the Delta Exchange v2 REST API:
  - Testnet base: https://cdn-ind.testnet.deltaex.org
  - Live base:    https://api.india.delta.exchange

Auth: HMAC-SHA256 over METHOD + TIMESTAMP + PATH_WITH_QUERY + BODY
  (path_with_query is the full path INCLUDING the leading '?query=...' if any)

Key endpoints used:
  GET  /v2/products                          (public)
  GET  /v2/tickers/{symbol}                  (public)
  GET  /v2/candles/{product_id}              (public)
  GET  /v2/wallet/balances                   (auth)
  POST /v2/orders                            (auth)  -- note: PLURAL
  POST /v2/orders/cancel                     (auth)
  GET  /v2/positions?underlying_asset_symbol (auth)
"""

import os
import hmac
import httpx
import hashlib
import time
import base64
from hermes_trading.adapters import SchemaError

SCHEMA_VERSION = "1.0"

# Delta Exchange endpoints
DELTA_TESTNET_BASE = "https://cdn-ind.testnet.deltaex.org"
DELTA_LIVE_BASE = "https://api.india.delta.exchange"


class DeltaError(Exception):
    """Base exception for Delta Exchange errors."""
    pass


class DeltaAuthError(DeltaError):
    """Raised when authentication fails."""
    pass


def _generate_signature(secret_key: str, method: str, timestamp: str,
                        path_with_query: str, body: str) -> str:
    """
    Generate Delta Exchange HMAC-SHA256 signature.

    Signature message = METHOD + TIMESTAMP + PATH_WITH_QUERY + BODY
    where PATH_WITH_QUERY is the full URL path including any '?key=val' suffix.

    Returns the hex digest.
    """
    message = f"{method}{timestamp}{path_with_query}{body}"
    return hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def _split_path(path: str) -> tuple:
    """
    Split a path into (path_only, query_string) on the first '?'.
    Returns (path, "") if no '?' present.
    """
    if "?" in path:
        p, q = path.split("?", 1)
        return p, q
    return path, ""


def fetch_price_sync(symbol: str = "BTCUSDT", mode: str = "testnet") -> dict:
    """
    Fetch ticker/price data from Delta Exchange (synchronous, public).

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        mode: "testnet" or "live"
    """
    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    formatted_symbol = symbol.replace("/", "").upper().replace("USDT", "USDT")
    path = f"/v2/tickers/{formatted_symbol}"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}{path}")
            if resp.status_code == 200:
                data = resp.json()
                # Delta response: {"success": true, "result": {<ticker fields>}}
                result = data.get("result") if isinstance(data, dict) else None
                if result:
                    last_price = float(result.get("close") or result.get("last_price") or 0)
                    open_24h = float(result.get("open") or result.get("open_24h") or 0)
                    volume_24h = float(result.get("volume") or result.get("volume_24h") or 0)
                    change_24h = ((last_price - open_24h) / open_24h) * 100 if open_24h else 0
                else:
                    last_price = float(data.get("last_price", 0))
                    open_24h = float(data.get("open_24h", 0))
                    volume_24h = float(data.get("volume_24h", 0))
                    change_24h = ((last_price - open_24h) / open_24h) * 100 if open_24h else 0
            else:
                return _ticker_fallback(symbol, mode)
    except Exception:
        return _ticker_fallback(symbol, mode)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "price": last_price,
        "change_24h": change_24h,
        "volume_24h": volume_24h,
        "status": "ok",
        "exchange": "delta_india",
        "mode": mode,
    }
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"Schema mismatch in delta adapter: {payload.get('schema_version')}")
    return payload


def _ticker_fallback(symbol: str, mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "price": 0.0,
        "change_24h": 0.0,
        "volume_24h": 0.0,
        "status": "error",
        "exchange": "delta_india",
        "mode": mode,
    }


def fetch_ohlc_sync(symbol: str = "BTCUSDT", timeframe: str = "1m",
                    limit: int = 100, mode: str = "testnet",
                    api_key: str = None, api_secret: str = None) -> dict:
    """
    Fetch OHLC candles from Delta Exchange.

    On testnet the candles endpoint requires auth. If api_key/secret are
    provided we sign the request; otherwise we return a public-style empty
    response so callers can fall back gracefully.
    """
    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    # Resolve product_id from symbol
    product_id = _resolve_product_id(symbol, mode, api_key, api_secret)
    if not product_id:
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "candles": [],
            "status": "error",
            "exchange": "delta_india",
            "mode": mode,
        }

    # Build path with query
    path_only = f"/v2/candles/{product_id}"
    query_string = f"resolution={timeframe}&limit={limit}"
    path_with_query = f"{path_only}?{query_string}"

    # Sign request
    if not (api_key and api_secret):
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "candles": [],
            "status": "error",
            "exchange": "delta_india",
            "mode": mode,
        }

    timestamp = str(int(time.time()))
    signature = _generate_signature(api_secret, "GET", timestamp, path_with_query, "")

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "Hermes-Trading/1.0",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{base_url}{path_with_query}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                # Response shape: {"success": true, "result": [candles]}
                candles_raw = data.get("result") if isinstance(data, dict) else data
                if isinstance(candles_raw, list):
                    candles = []
                    for c in candles_raw:
                        # Shape: {"open":..., "high":..., "low":..., "close":..., "volume":..., "time":...}
                        if isinstance(c, dict):
                            candles.append({
                                "timestamp": c.get("time") or c.get("t"),
                                "open": float(c.get("open", 0)),
                                "high": float(c.get("high", 0)),
                                "low": float(c.get("low", 0)),
                                "close": float(c.get("close", 0)),
                                "volume": float(c.get("volume", 0)),
                            })
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "symbol": symbol,
                        "candles": candles,
                        "status": "ok",
                        "exchange": "delta_india",
                        "mode": mode,
                    }
    except Exception:
        pass

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "candles": [],
        "status": "error",
        "exchange": "delta_india",
        "mode": mode,
    }


def _resolve_product_id(symbol: str, mode: str,
                        api_key: str = None, api_secret: str = None) -> str:
    """
    Resolve a symbol like 'BTCUSDT' to its Delta product_id.
    Fetches /v2/products and finds the perpetual_futures entry.
    """
    formatted = symbol.replace("/", "").upper().replace("USDT", "USD")
    base_url = DELTA_TESTNET_BASE if mode == "testnet" else DELTA_LIVE_BASE

    try:
        if api_key and api_secret:
            path = "/v2/products"
            timestamp = str(int(time.time()))
            signature = _generate_signature(api_secret, "GET", timestamp, path, "")
            headers = {
                "api-key": api_key,
                "signature": signature,
                "timestamp": timestamp,
                "User-Agent": "Hermes-Trading/1.0",
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}{path}", headers=headers)
        else:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/v2/products")

        if resp.status_code == 200:
            data = resp.json()
            products = data.get("result") if isinstance(data, dict) else data
            if isinstance(products, list):
                for p in products:
                    if (p.get("symbol") == formatted
                            and p.get("contract_type") == "perpetual_futures"):
                        return str(p.get("id"))
    except Exception:
        pass

    # Fallback: hardcoded known perp IDs on testnet
    fallback = {"BTCUSD": "84", "ETHUSD": "85"}
    return fallback.get(formatted, "84")


def place_order_sync(side: str, quantity: float, price: float = None,
                     order_type: str = "market_order",
                     symbol: str = "BTCUSDT", mode: str = "testnet",
                     api_key: str = None, api_secret: str = None,
                     reduce_only: bool = False) -> dict:
    """
    Place an order on Delta Exchange (synchronous, requires auth).

    Args:
        side: "buy" or "sell"
        quantity: number of contracts
        price: required for limit orders
        order_type: "market_order" or "limit_order"
        symbol: e.g. "BTCUSDT"
        mode: "testnet" or "live"
        api_key, api_secret: API credentials
        reduce_only: if True, order can only reduce existing position
    """
    if not api_key or not api_secret:
        raise DeltaAuthError("Delta API key and secret are required for order placement")

    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    product_id = _resolve_product_id(symbol, mode, api_key, api_secret)
    path = "/v2/orders"

    # Build the order body. Delta v2 wants 'size' (not quantity) and
    # 'product_id' (numeric).
    body_dict = {
        "product_id": int(product_id),
        "side": side,
        "size": int(quantity) if quantity == int(quantity) else quantity,
        "order_type": order_type,
    }
    if order_type == "limit_order":
        if price is None:
            raise DeltaError("limit_price required for limit orders")
        body_dict["limit_price"] = str(price)
    if reduce_only:
        body_dict["reduce_only"] = True

    import json as _json
    body = _json.dumps(body_dict)

    timestamp = str(int(time.time()))
    signature = _generate_signature(api_secret, "POST", timestamp, path, body)

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "Hermes-Trading/1.0",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}{path}", headers=headers, content=body)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}

            if resp.status_code in (200, 201) and data.get("success"):
                result = data.get("result", {})
                return {
                    "status": "filled" if order_type == "market_order" else "placed",
                    "order_id": result.get("id"),
                    "product_id": result.get("product_id"),
                    "product_symbol": result.get("product_symbol"),
                    "side": result.get("side"),
                    "size": result.get("size"),
                    "filled_size": result.get("size", 0) - result.get("unfilled_size", 0),
                    "average_fill_price": result.get("average_fill_price"),
                    "order_type": result.get("order_type"),
                    "state": result.get("state"),
                    "created_at": result.get("created_at"),
                    "side": side,
                    "symbol": symbol,
                    "mode": mode,
                }
            else:
                error_msg = data.get("error", data)
                raise DeltaError(f"Delta order failed: {resp.status_code} - {error_msg}")

    except DeltaError:
        raise
    except Exception as e:
        raise DeltaError(f"Delta order placement error: {e}")


def cancel_order_sync(order_id: str, symbol: str = "BTCUSDT",
                      mode: str = "testnet",
                      api_key: str = None, api_secret: str = None) -> dict:
    """Cancel an open order on Delta Exchange."""
    if not api_key or not api_secret:
        raise DeltaAuthError("Delta API key and secret are required for order cancellation")

    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    import json as _json
    body = _json.dumps({"id": int(order_id), "product_id": int(_resolve_product_id(symbol, mode, api_key, api_secret))})
    path = "/v2/orders/cancel"

    timestamp = str(int(time.time()))
    signature = _generate_signature(api_secret, "POST", timestamp, path, body)

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "Hermes-Trading/1.0",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}{path}", headers=headers, content=body)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}

            if resp.status_code == 200 and data.get("success"):
                return {
                    "status": "cancelled",
                    "order_id": order_id,
                    "result": data.get("result"),
                }
            else:
                raise DeltaError(f"Delta cancel failed: {resp.status_code} - {data.get('error', data)}")

    except DeltaError:
        raise
    except Exception as e:
        raise DeltaError(f"Delta cancel order error: {e}")


def get_wallet_balances_sync(mode: str = "testnet", api_key: str = None,
                             api_secret: str = None) -> dict:
    """Get wallet balances from Delta Exchange (synchronous, requires auth)."""
    if not api_key or not api_secret:
        raise DeltaAuthError("Delta API key and secret are required for wallet balance queries")

    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    path = "/v2/wallet/balances"
    timestamp = str(int(time.time()))
    signature = _generate_signature(api_secret, "GET", timestamp, path, "")

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "Hermes-Trading/1.0",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{base_url}{path}", headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}

            if resp.status_code == 200 and data.get("success"):
                # Delta response: {"success": true, "result": [{asset_symbol, available_balance, balance, ...}], "meta": {net_equity, ...}}
                balances = {}
                for bal in data.get("result", []):
                    ccy = bal.get("asset_symbol", "")
                    balances[ccy] = {
                        "available": float(bal.get("available_balance", 0)),
                        "total": float(bal.get("balance", 0)),
                    }
                meta = data.get("meta", {})
                return {
                    "schema_version": SCHEMA_VERSION,
                    "balances": balances,
                    "net_equity": float(meta.get("net_equity", 0)),
                    "status": "ok",
                    "exchange": "delta_india",
                    "mode": mode,
                }
            else:
                raise DeltaError(f"Delta wallet fetch failed: {resp.status_code} - {data.get('error', data)}")

    except DeltaError:
        raise
    except Exception as e:
        raise DeltaError(f"Delta wallet balances error: {e}")


def get_positions_sync(symbol: str = "BTCUSD", mode: str = "testnet",
                       api_key: str = None, api_secret: str = None) -> dict:
    """Get open positions for an underlying asset (BTC, ETH, etc.) on Delta Exchange."""
    if not api_key or not api_secret:
        raise DeltaAuthError("Delta API key and secret required for positions query")

    if mode == "testnet":
        base_url = DELTA_TESTNET_BASE
    else:
        base_url = DELTA_LIVE_BASE

    # Strip USDT/USD suffix to get the underlying asset symbol (BTC, ETH)
    underlying = symbol.replace("/", "").upper().replace("USDT", "").replace("USD", "")
    path_only = "/v2/positions"
    path_with_query = f"{path_only}?underlying_asset_symbol={underlying}"

    timestamp = str(int(time.time()))
    signature = _generate_signature(api_secret, "GET", timestamp, path_with_query, "")

    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "Hermes-Trading/1.0",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{base_url}{path_with_query}", headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}

            if resp.status_code == 200 and data.get("success"):
                positions = data.get("result", [])
                return {
                    "schema_version": SCHEMA_VERSION,
                    "positions": positions,
                    "status": "ok",
                    "exchange": "delta_india",
                    "mode": mode,
                }
            else:
                raise DeltaError(f"Delta positions fetch failed: {resp.status_code} - {data.get('error', data)}")

    except DeltaError:
        raise
    except Exception as e:
        raise DeltaError(f"Delta positions error: {e}")