from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt  # PyJWT
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Keep auth settings centralized so other routers can import without circular deps.
SECRET_KEY = os.getenv("MGSA_SECRET_KEY", "mgsa-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("MGSA_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

security = HTTPBearer()

# NOTE: kept identical to server/server.py defaults
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "cveto-msga": {"password": "raspberry", "role": "client"},
}


def create_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

