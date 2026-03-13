"""
auth.py — API Key authentication dependency.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from settings import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on server.",
        )
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return key