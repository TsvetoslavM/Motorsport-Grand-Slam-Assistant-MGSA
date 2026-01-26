from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.auth import verify_token  # reuse existing auth (no circular import)

try:
    from firmware.Optimal_Control.solver_api import (
        OptimizeOptions,
        optimize_trajectory_from_two_lines,
    )
except Exception as e:  # pragma: no cover
    # Server can still start; endpoint will return a clear error if called.
    OptimizeOptions = None  # type: ignore
    optimize_trajectory_from_two_lines = None  # type: ignore
    _IMPORT_ERR = str(e)
else:
    _IMPORT_ERR = ""


router = APIRouter(prefix="/api/trajectory", tags=["trajectory"])


class XYPoint(BaseModel):
    x: float
    y: float


class TwoLinesOptimizeRequest(BaseModel):
    left_line: List[XYPoint] = Field(..., description="Left boundary polyline (x,y) points")
    right_line: List[XYPoint] = Field(..., description="Right boundary polyline (x,y) points")

    n_points: int = Field(250, ge=30, le=5000)
    ipopt_max_iter: int = Field(2000, ge=50, le=20000)
    ipopt_print_level: int = Field(0, ge=0, le=12)
    ipopt_tol: float = Field(1e-4, gt=0)
    ipopt_acceptable_tol: float = Field(1e-3, gt=0)
    ipopt_linear_solver: str = Field("mumps")


@router.post("/optimize_from_two_lines")
async def optimize_from_two_lines(req: TwoLinesOptimizeRequest, token: dict = Depends(verify_token)):
    if optimize_trajectory_from_two_lines is None or OptimizeOptions is None:
        raise HTTPException(
            status_code=500,
            detail=f"Firmware optimizer unavailable (casadi/ipopt missing or import error): {_IMPORT_ERR}",
        )

    if len(req.left_line) < 3 or len(req.right_line) < 3:
        raise HTTPException(status_code=400, detail="Each line must contain at least 3 points")

    opts = OptimizeOptions(
        n_points=req.n_points,
        ipopt_max_iter=req.ipopt_max_iter,
        ipopt_print_level=req.ipopt_print_level,
        ipopt_tol=req.ipopt_tol,
        ipopt_acceptable_tol=req.ipopt_acceptable_tol,
        ipopt_linear_solver=req.ipopt_linear_solver,
    )

    try:
        result = optimize_trajectory_from_two_lines(
            [p.model_dump() for p in req.left_line],
            [p.model_dump() for p in req.right_line],
            options=opts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    return result

