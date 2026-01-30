## MGSA server

This package is the **laptop / desktop backend** for MGSA, built with FastAPI.
It exposes APIs for recording laps, managing tracks, uploading racing lines and running analysis.

### Main entrypoints

- `server.py` – FastAPI app used in the field (`uvicorn server.server:app`).
- `app.py` / `main.py` – legacy / helper entrypoints.
- `routes/` – per‑feature routers:
  - `auth_routes.py` – login / token.
  - `gps_routes.py` – ingest GPS points.
  - `lap_routes.py` – start/stop/list laps.
  - `track_routes.py` – boundaries, racing line upload/download.
  - `status_routes.py` – runtime status.
- `auto_pipeline.py`, `analysis_routes.py`, `trajectory_api.py` – higher‑level analysis and trajectory generation endpoints.

All persistent data lives under `./mgsa_data` by default (SQLite DB + CSV laps + track artifacts).

### Running the server

From project root:

```bash
pip install -r requirements.txt  # or ensure FastAPI + Uvicorn + Pydantic, etc.

uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Or, using the convenience `__main__` in `server.py`:

```bash
python -m server.server
```

### Configuration

- `MGSA_SECRET_KEY` – optional `SECRET_KEY` override (JWT signing).
- Data directory: `./mgsa_data` (created automatically on first start).

