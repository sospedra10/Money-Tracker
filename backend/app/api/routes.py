from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    DashboardRequest,
    EntryCreate,
    EntryOut,
    HealthOut,
    LatestBalancesOut,
    MetaOut,
)
from app.core.config import settings
from app.data.json_store import (
    DEFAULT_CATEGORIES,
    INVESTMENT_CATEGORIES,
    NON_INVESTMENT_CATEGORIES,
    RISK_LIQUIDITY_PROFILE,
    VOLATILITY_ASSUMPTIONS,
    HistoryEntry,
    JsonStore,
)
from app.analytics.dashboard import build_dashboard_payload


router = APIRouter()
store = JsonStore(settings.data_file, categories=DEFAULT_CATEGORIES, date_format="legacy")


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(details={"data_file": str(settings.data_file)})


@router.get("/meta", response_model=MetaOut)
def meta() -> MetaOut:
    return MetaOut(
        categories=DEFAULT_CATEGORIES,
        investment_categories=INVESTMENT_CATEGORIES,
        non_investment_categories=NON_INVESTMENT_CATEGORIES,
        risk_liquidity_profile=RISK_LIQUIDITY_PROFILE,
        volatility_assumptions=VOLATILITY_ASSUMPTIONS,
    )


@router.get("/history", response_model=list[EntryOut])
def history(
    category: str | None = None,
    limit: int = Query(default=5000, ge=1, le=100_000),
) -> list[EntryOut]:
    entries = store.load_history()
    if category is not None:
        entries = [e for e in entries if e.category == category]
    entries = sorted(entries, key=lambda e: e.date)
    return [EntryOut(date=e.date, category=e.category, amount=e.amount) for e in entries[-limit:]]


@router.post("/history", response_model=EntryOut)
def add_history_entry(payload: EntryCreate) -> EntryOut:
    entry = HistoryEntry(
        date=payload.date or datetime.now(),
        category=payload.category,
        amount=payload.amount,
    )
    try:
        store.append_entry(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EntryOut(date=entry.date, category=entry.category, amount=entry.amount)


@router.get("/balances/latest", response_model=LatestBalancesOut)
def latest_balances() -> LatestBalancesOut:
    entries = store.load_history()
    latest_by_cat: dict[str, HistoryEntry] = {}
    for entry in sorted(entries, key=lambda e: e.date):
        if entry.category not in DEFAULT_CATEGORIES:
            continue
        latest_by_cat[entry.category] = entry

    balances: dict[str, float] = {cat: 0.0 for cat in DEFAULT_CATEGORIES}
    as_of = datetime.fromtimestamp(0)
    for cat in DEFAULT_CATEGORIES:
        if cat not in latest_by_cat:
            continue
        balances[cat] = float(latest_by_cat[cat].amount)
        as_of = max(as_of, latest_by_cat[cat].date)

    total = float(sum(balances.values()))
    if as_of == datetime.fromtimestamp(0):
        as_of = datetime.now()

    return LatestBalancesOut(as_of=as_of, balances=balances, total=total)


@router.post("/analytics/dashboard")
def analytics_dashboard(payload: DashboardRequest) -> dict[str, Any]:
    entries = store.load_history()
    return build_dashboard_payload(
        entries,
        interest_rates=payload.interest_rates,
        months_to_analyze=payload.months_to_analyze,
        projection_years=payload.projection_years,
        scenario_enabled=payload.scenario.enabled,
        scenario_savings_adjustment_pct=payload.scenario.savings_adjustment_pct,
        scenario_extra_monthly=payload.scenario.extra_monthly,
        scenario_interest_shift_pp=payload.scenario.interest_shift_pp,
        goal_amount=payload.goal.target_amount,
        goal_years=payload.goal.years,
        mc_simulations=payload.mc_simulations,
    )
