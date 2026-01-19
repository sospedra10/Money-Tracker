from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONEYTRACKER_", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[3]
    data_file: Path = project_root / "financial_data.json"

    api_prefix: str = "/api/v1"
    allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


settings = Settings()

