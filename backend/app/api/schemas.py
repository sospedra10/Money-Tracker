from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EntryCreate(BaseModel):
    category: str = Field(min_length=1)
    amount: float = Field(ge=0)
    date: datetime | None = None


class EntryOut(BaseModel):
    date: datetime
    category: str
    amount: float


class MetaOut(BaseModel):
    categories: list[str]
    investment_categories: list[str]
    non_investment_categories: list[str]
    risk_liquidity_profile: dict[str, dict[str, float]]
    volatility_assumptions: dict[str, float]


class LatestBalancesOut(BaseModel):
    as_of: datetime
    balances: dict[str, float]
    total: float


class HealthOut(BaseModel):
    status: str = "ok"
    details: dict[str, Any] = {}


class ScenarioSettings(BaseModel):
    enabled: bool = False
    savings_adjustment_pct: float = 20.0
    extra_monthly: float = Field(default=0.0, ge=0.0)
    interest_shift_pp: float = 1.0


class GoalSettings(BaseModel):
    target_amount: float = Field(default=100_000.0, ge=0.0)
    years: int = Field(default=5, ge=1, le=60)


class DashboardRequest(BaseModel):
    interest_rates: dict[str, float] | None = None
    months_to_analyze: int = Field(default=10, ge=1, le=60)
    projection_years: int = Field(default=5, ge=1, le=60)
    scenario: ScenarioSettings = ScenarioSettings()
    goal: GoalSettings = GoalSettings()
    mc_simulations: int = Field(default=300, ge=50, le=5000)
