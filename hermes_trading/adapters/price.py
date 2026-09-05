import os
import httpx
from hermes_trading.adapters import SchemaError

SCHEMA_VERSION = "1.0"

async def fetch(symbol: str = "BTC/USDT") -> dict:
    """Fetch spot price and ticker data using Binance public API or ccxt."""
    formatted_symbol = symbol.replace("/", "").upper()
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={formatted_symbol}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                # Fallback to Coingecko simple price
                coin = "bitcoin" if "BTC" in symbol else "ethereum"
                cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
                cg_resp = await client.get(cg_url)
                cg_data = cg_resp.json()
                price = float(cg_data[coin]["usd"])
                change_24h = float(cg_data[coin].get("usd_24h_change", 0.0))
                vol = float(cg_data[coin].get("usd_24h_vol", 0.0))
            else:
                data = resp.json()
                price = float(data["lastPrice"])
                change_24h = float(data["priceChangePercent"])
                vol = float(data["volume"])
                
        payload = {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "price": price,
            "change_24h": change_24h,
            "volume_24h": vol,
            "status": "ok"
        }
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(f"Schema mismatch in price adapter: {payload.get('schema_version')}")
        return payload
    except Exception as e:
        if isinstance(e, SchemaError):
            raise
        raise RuntimeError(f"Price adapter fetch failed: {e}")
