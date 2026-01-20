import logging
from fastapi import APIRouter, HTTPException
from ..models import LoginRequest
from ..auth import create_token, users

logger = logging.getLogger("mgsa-server")
router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login")
async def login(req: LoginRequest):
    user = users.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(req.username, user["role"])
    logger.info(f"User logged in: {req.username}")
    return {"access_token": token, "token_type": "bearer", "username": req.username, "role": user["role"]}
