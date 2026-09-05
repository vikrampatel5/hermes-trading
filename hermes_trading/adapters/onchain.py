import os
import httpx
from hermes_trading.adapters import SchemaError

SCHEMA_VERSION = "1.0"

async def fetch() -> dict:
    """Fetch on-chain metrics (mempool / gas / difficulty)."""
    url = "https://mempool.space/api/v1/fees/recommended"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                fast_fee = data.get("fastestFee", 20)
                half_hour = data.get("halfHourFee", 15)
            else:
                fast_fee = 20
                half_hour = 15

        payload = {
            "schema_version": SCHEMA_VERSION,
            "fastest_fee_sat_vb": fast_fee,
            "half_hour_fee_sat_vb": half_hour,
            "status": "ok"
        }
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(f"Schema mismatch in onchain adapter: {payload.get('schema_version')}")
        return payload
    except Exception as e:
        if isinstance(e, SchemaError):
            raise
        # return graceful default
        return {
            "schema_version": SCHEMA_VERSION,
            "fastest_fee_sat_vb": 20,
            "half_hour_fee_sat_vb": 15,
            "status": "fallback"
        }
