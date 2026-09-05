import os
import httpx
from hermes_trading.adapters import SchemaError

SCHEMA_VERSION = "1.0"

async def fetch() -> dict:
    """Fetch market sentiment / news signals."""
    # Free alternative: Fear & Greed Index
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                fng_value = int(data["data"][0]["value"])
                fng_classification = data["data"][0]["value_classification"]
            else:
                fng_value = 50
                fng_classification = "Neutral"

        payload = {
            "schema_version": SCHEMA_VERSION,
            "fear_and_greed_index": fng_value,
            "sentiment": fng_classification,
            "status": "ok"
        }
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(f"Schema mismatch in news adapter: {payload.get('schema_version')}")
        return payload
    except Exception as e:
        if isinstance(e, SchemaError):
            raise
        return {
            "schema_version": SCHEMA_VERSION,
            "fear_and_greed_index": 50,
            "sentiment": "Neutral",
            "status": "fallback"
        }
