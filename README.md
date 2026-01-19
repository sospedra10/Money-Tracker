# Money Tracker

Este repo contiene **dos apps**:

- **Legacy (Streamlit):** `moneytracker.py` (no se toca, sigue funcionando como antes).
- **Nueva versión (FastAPI + React/TS):** `backend/` + `frontend/`.

Los datos se siguen guardando en `financial_data.json` (mismo formato: snapshots por categoría con fecha + amount).

## Run (1-click)

1. Doble click en `MoneyTracker_NEW.bat`.
2. Se abre el navegador en `http://127.0.0.1:8000`.

La primera vez instalará dependencias (Python y npm) y hará el build del frontend.

## Run (dev)

1. Doble click en `MoneyTracker_DEV.bat`.
2. Frontend dev: `http://127.0.0.1:5173` (con proxy hacia el backend).

## API

- `GET /api/v1/health`
- `GET /api/v1/meta`
- `GET /api/v1/history`
- `POST /api/v1/history`
- `GET /api/v1/balances/latest`
- `POST /api/v1/analytics/dashboard`

