from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.data.json_store import (
    DEFAULT_CATEGORIES,
    INVESTMENT_CATEGORIES,
    NON_INVESTMENT_CATEGORIES,
    RISK_LIQUIDITY_PROFILE,
    VOLATILITY_ASSUMPTIONS,
    HistoryEntry,
)


@dataclass(frozen=True)
class Streak:
    current: int
    best: int


def _as_dataframe(entries: list[HistoryEntry]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(columns=["date", "category", "amount"])
    return pd.DataFrame(
        [{"date": e.date, "category": e.category, "amount": e.amount} for e in entries]
    )


def prepare_chart_data(
    entries: list[HistoryEntry],
    *,
    categories: list[str] = DEFAULT_CATEGORIES,
) -> pd.DataFrame:
    df = _as_dataframe(entries)
    if df.empty:
        return pd.DataFrame()

    df = df[df["category"].isin(categories)]
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    aggregated = df.groupby(["date", "category"], as_index=False)["amount"].sum()
    pivoted = aggregated.pivot(index="date", columns="category", values="amount").sort_index()

    for category in categories:
        if category not in pivoted.columns:
            pivoted[category] = np.nan

    pivoted = pivoted[categories].ffill().fillna(0.0)
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted.index.name = "date"
    pivoted["Total"] = pivoted.sum(axis=1)
    return pivoted


def to_daily_last(chart_data: pd.DataFrame) -> pd.DataFrame:
    if chart_data.empty:
        return pd.DataFrame()
    daily = chart_data.copy().reset_index()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily.groupby("date", as_index=False).last()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def monthly_values(chart_data: pd.DataFrame) -> pd.DataFrame:
    if chart_data.empty:
        return pd.DataFrame()
    return chart_data.resample("M").last().ffill().fillna(0.0)


def monthly_changes(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return pd.DataFrame()
    return monthly_df.diff().fillna(0.0)


def monthly_changes_long(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame(columns=["month", "category", "change"])
    out = changes.copy()
    out.index = out.index.to_period("M").to_timestamp()
    out = out.reset_index().rename(columns={"date": "month"})
    if "month" not in out.columns:
        out = out.rename(columns={out.columns[0]: "month"})
    melted = out.melt(id_vars="month", var_name="category", value_name="change")
    melted["month"] = pd.to_datetime(melted["month"]).dt.strftime("%Y-%m")
    return melted


def avg_monthly_savings(
    changes_long: pd.DataFrame, *, months_to_analyze: int
) -> dict[str, float]:
    if changes_long.empty:
        return {cat: 0.0 for cat in DEFAULT_CATEGORIES} | {"Total": 0.0}

    months = changes_long["month"].drop_duplicates().tail(months_to_analyze).tolist()
    window = changes_long[changes_long["month"].isin(months)]
    out: dict[str, float] = {}
    for category in DEFAULT_CATEGORIES + ["Total"]:
        values = window.loc[window["category"] == category, "change"]
        out[category] = float(values.mean()) if not values.empty else 0.0
    return out


def calculate_streak(total_monthly_changes: pd.Series) -> Streak:
    series = total_monthly_changes.dropna().tolist()
    if not series:
        return Streak(current=0, best=0)

    best = 0
    run = 0
    for value in series:
        if value > 0:
            run += 1
            best = max(best, run)
        else:
            run = 0

    current = 0
    for value in reversed(series):
        if value > 0:
            current += 1
        else:
            break

    return Streak(current=current, best=best)


def herfindahl_index(weights: dict[str, float]) -> float:
    squares = [w * w for w in weights.values() if w > 0]
    return float(sum(squares)) if squares else 0.0


def max_drawdown(values: pd.Series) -> float:
    series = values.dropna()
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    drawdowns = (series / running_peak) - 1.0
    return float(drawdowns.min())


def cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0.0
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def _default_interest_rates() -> dict[str, float]:
    return {
        "Remuneration Account": 0.025,
        "Real Estate": 0.11,
        "ETFs and Stocks": 0.065,
        "Bank": 0.0,
        "Crypto": 0.01,
        "Reenlever": 0.11,
        "Staking": 0.03,
        "Others": 0.0,
    }


def normalize_interest_rates(rates: dict[str, float] | None) -> dict[str, float]:
    merged = _default_interest_rates()
    if rates:
        for key, value in rates.items():
            try:
                merged[key] = max(float(value), 0.0)
            except Exception:
                continue
    for category in DEFAULT_CATEGORIES:
        merged.setdefault(category, 0.0)
    return merged


def annual_passive_income(balances: dict[str, float], rates: dict[str, float]) -> float:
    total = 0.0
    for category, amount in balances.items():
        total += float(amount) * float(rates.get(category, 0.0))
    return float(total)


def risk_liquidity_snapshot(balances: dict[str, float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for category, profile in RISK_LIQUIDITY_PROFILE.items():
        balance = float(balances.get(category, 0.0))
        if balance <= 0:
            continue
        rows.append(
            {
                "category": category,
                "balance": balance,
                "risk": float(profile["risk"]),
                "liquidity": float(profile["liquidity"]),
                "expected_return": float(profile["expected_return"]),
            }
        )
    return rows


def correlation_matrix(monthly_df: pd.DataFrame) -> dict[str, Any]:
    if monthly_df.empty:
        return {"categories": [], "matrix": []}
    cats = [c for c in DEFAULT_CATEGORIES if c in monthly_df.columns]
    if not cats:
        return {"categories": [], "matrix": []}
    returns = monthly_df[cats].pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, how="all").dropna()
    if returns.empty:
        return {"categories": [], "matrix": []}
    corr = returns.corr().fillna(0.0)
    return {"categories": corr.columns.tolist(), "matrix": corr.values.round(4).tolist()}


def deterministic_projection_total(
    *,
    current_balances: dict[str, float],
    interest_rates: dict[str, float],
    monthly_contributions: dict[str, float],
    months: int,
) -> pd.DataFrame:
    if months <= 0:
        return pd.DataFrame(columns=["date", "value"])

    invest_cats = [c for c in INVESTMENT_CATEGORIES if c in DEFAULT_CATEGORIES]
    static_cats = [c for c in DEFAULT_CATEGORIES if c not in invest_cats]

    category_values = {cat: float(current_balances.get(cat, 0.0)) for cat in invest_cats}
    static_total = float(sum(float(current_balances.get(cat, 0.0)) for cat in static_cats))

    start_date = pd.to_datetime("today").normalize()
    dates = pd.date_range(start=start_date, periods=months + 1, freq="MS")

    totals: list[float] = [float(sum(category_values.values()) + static_total)]
    for _ in range(months):
        for cat in invest_cats:
            rate = float(interest_rates.get(cat, 0.0)) / 12.0
            contrib = float(monthly_contributions.get(cat, 0.0))
            value = category_values.get(cat, 0.0)
            value = max(value * (1 + rate) + contrib, 0.0)
            category_values[cat] = value
        totals.append(float(sum(category_values.values()) + static_total))

    return pd.DataFrame({"date": dates, "value": totals})


def run_monte_carlo_projection(
    *,
    current_balances: dict[str, float],
    interest_rates: dict[str, float],
    monthly_contributions: dict[str, float],
    months: int,
    simulations: int,
) -> pd.DataFrame:
    total_current = float(sum(current_balances.values()))
    if total_current <= 0 or months <= 0 or simulations <= 0:
        return pd.DataFrame(columns=["date", "p05", "p50", "p95"])

    invest_cats = [c for c in INVESTMENT_CATEGORIES if c in DEFAULT_CATEGORIES]
    static_cats = [c for c in DEFAULT_CATEGORIES if c not in invest_cats]
    static_total = float(sum(float(current_balances.get(cat, 0.0)) for cat in static_cats))

    rng = np.random.default_rng()
    projections = np.zeros((months + 1, simulations), dtype=float)
    projections[0, :] = total_current

    for sim in range(simulations):
        category_values = {cat: float(current_balances.get(cat, 0.0)) for cat in invest_cats}
        for month in range(1, months + 1):
            monthly_total = static_total
            for cat in invest_cats:
                value = category_values.get(cat, 0.0)
                mean = float(interest_rates.get(cat, 0.0)) / 12.0
                annual_vol = float(VOLATILITY_ASSUMPTIONS.get(cat, 0.2))
                monthly_vol = annual_vol / np.sqrt(12)
                simulated_return = float(rng.normal(mean, monthly_vol))
                contribution = float(monthly_contributions.get(cat, 0.0))
                value = max(value * (1 + simulated_return) + contribution, 0.0)
                category_values[cat] = value
                monthly_total += value
            projections[month, sim] = monthly_total

    percentiles = np.percentile(projections, [5, 50, 95], axis=1)
    date_range = pd.date_range(start=pd.to_datetime("today").normalize(), periods=months + 1, freq="MS")
    return pd.DataFrame(
        {
            "date": date_range,
            "p05": percentiles[0].astype(float),
            "p50": percentiles[1].astype(float),
            "p95": percentiles[2].astype(float),
        }
    )


def required_extra_monthly(
    *,
    current_total: float,
    goal_amount: float,
    goal_months: int,
    monthly_rate: float,
    current_monthly: float,
) -> float:
    if goal_amount <= 0 or goal_months <= 0:
        return 0.0

    growth_factor = (1 + monthly_rate) ** goal_months
    future_value_current = current_total * growth_factor

    if monthly_rate == 0:
        projected = future_value_current + current_monthly * goal_months
        shortfall = goal_amount - projected
        return max(shortfall / goal_months, 0.0)

    contribution_factor = ((1 + monthly_rate) ** goal_months - 1) / monthly_rate
    projected = future_value_current + current_monthly * contribution_factor
    shortfall = goal_amount - projected
    if shortfall <= 0:
        return 0.0
    return float(shortfall / contribution_factor)


def generate_insights(
    *,
    balances: dict[str, float],
    monthly_savings: dict[str, float],
    interest_rates: dict[str, float],
    streak: Streak,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []

    total = float(sum(balances.values()))
    if total <= 0:
        return insights

    leader = max(balances, key=balances.get)
    leader_share = (float(balances[leader]) / total) * 100.0 if total else 0.0
    insights.append(
        {
            "title": "Concentración principal",
            "text": f"{leader} es el {leader_share:,.1f}% del total (€{balances[leader]:,.0f}).",
        }
    )

    positive = {k: v for k, v in monthly_savings.items() if k != "Total" and v > 0}
    if positive:
        best = max(positive, key=positive.get)
        insights.append(
            {
                "title": "Mejor hábito de ahorro",
                "text": f"Estás aumentando {best} en ~€{positive[best]:,.0f}/mes (media reciente).",
            }
        )

    negative = {k: v for k, v in monthly_savings.items() if k != "Total" and v < 0}
    if negative:
        worst = min(negative, key=negative.get)
        insights.append(
            {
                "title": "Posible fuga",
                "text": f"{worst} baja ~€{abs(negative[worst]):,.0f}/mes. Revisa si es intencional.",
            }
        )

    passive = annual_passive_income(balances, interest_rates)
    if passive > 0:
        insights.append(
            {
                "title": "Ingreso pasivo estimado",
                "text": f"Con tus tasas actuales: ~€{passive:,.0f}/año (≈€{passive/12:,.0f}/mes).",
            }
        )

    if streak.current > 0:
        insights.append(
            {
                "title": "Racha positiva",
                "text": f"Llevas {streak.current} meses seguidos en positivo (mejor: {streak.best}).",
            }
        )

    return insights[:6]


def build_dashboard_payload(
    entries: list[HistoryEntry],
    *,
    interest_rates: dict[str, float] | None,
    months_to_analyze: int,
    projection_years: int,
    scenario_enabled: bool,
    scenario_savings_adjustment_pct: float,
    scenario_extra_monthly: float,
    scenario_interest_shift_pp: float,
    goal_amount: float,
    goal_years: int,
    mc_simulations: int,
) -> dict[str, Any]:
    chart = prepare_chart_data(entries)
    if chart.empty:
        return {
            "has_data": False,
            "message": "No hay datos todavía. Añade tu primer update en una categoría.",
        }

    daily = to_daily_last(chart)
    latest_row = chart.iloc[-1]
    balances = {cat: float(latest_row.get(cat, 0.0)) for cat in DEFAULT_CATEGORIES}
    total = float(latest_row.get("Total", 0.0))
    as_of = pd.to_datetime(chart.index.max()).to_pydatetime()

    rates = normalize_interest_rates(interest_rates)

    month_vals = monthly_values(chart)
    month_changes = monthly_changes(month_vals)
    changes_long = monthly_changes_long(month_changes)
    avg_savings = avg_monthly_savings(changes_long, months_to_analyze=months_to_analyze)

    total_monthly_changes = month_changes["Total"] if "Total" in month_changes.columns else pd.Series([])
    streak = calculate_streak(total_monthly_changes)

    allocation = {cat: (balances[cat] / total) if total else 0.0 for cat in DEFAULT_CATEGORIES}
    hhi = herfindahl_index(allocation)
    effective_positions = float(1.0 / hhi) if hhi > 0 else 0.0

    liquidity_total = float(
        balances.get("Bank", 0.0)
        + balances.get("Remuneration Account", 0.0)
        + balances.get("Reenlever", 0.0)
    )

    risk_rows = risk_liquidity_snapshot(balances)
    weighted_risk = 0.0
    weighted_liquidity = 0.0
    weighted_expected_return = 0.0
    if total > 0:
        for row in risk_rows:
            w = float(row["balance"]) / total
            weighted_risk += w * float(row["risk"])
            weighted_liquidity += w * float(row["liquidity"])
            weighted_expected_return += w * float(row["expected_return"])

    mdd = max_drawdown(daily["Total"]) if not daily.empty and "Total" in daily.columns else 0.0
    start_date = pd.to_datetime(daily["date"].min()) if not daily.empty else pd.to_datetime(as_of)
    years = max((pd.to_datetime(as_of) - start_date).days / 365.25, 0.0)
    cagr_total = cagr(float(daily["Total"].iloc[0]), float(daily["Total"].iloc[-1]), years) if not daily.empty else 0.0

    monthly_total = month_vals["Total"] if "Total" in month_vals.columns else pd.Series([])
    monthly_returns = monthly_total.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol_monthly = float(monthly_returns.std()) if not monthly_returns.empty else 0.0

    correlation = correlation_matrix(month_vals)

    base_monthly_contributions = {
        cat: float(avg_savings.get(cat, 0.0)) for cat in INVESTMENT_CATEGORIES
    }
    base_monthly_contributions = {k: max(v, 0.0) for k, v in base_monthly_contributions.items()}

    projection_months = max(int(projection_years) * 12, 0)
    baseline_proj = deterministic_projection_total(
        current_balances=balances,
        interest_rates=rates,
        monthly_contributions=base_monthly_contributions,
        months=projection_months,
    )

    scenario_proj: pd.DataFrame | None = None
    scenario_mc: pd.DataFrame | None = None
    scenario_rates = rates.copy()
    scenario_contrib = base_monthly_contributions.copy()
    if scenario_enabled:
        factor = 1 + float(scenario_savings_adjustment_pct) / 100.0
        per_cat_extra = (
            float(scenario_extra_monthly) / len(INVESTMENT_CATEGORIES) if INVESTMENT_CATEGORIES else 0.0
        )
        for cat in scenario_contrib:
            scenario_contrib[cat] = max(float(scenario_contrib.get(cat, 0.0)) * factor + per_cat_extra, 0.0)
        shift = float(scenario_interest_shift_pp) / 100.0
        for cat in scenario_rates:
            scenario_rates[cat] = max(float(scenario_rates.get(cat, 0.0)) + shift, 0.0)

        scenario_proj = deterministic_projection_total(
            current_balances=balances,
            interest_rates=scenario_rates,
            monthly_contributions=scenario_contrib,
            months=projection_months,
        )

    baseline_mc = run_monte_carlo_projection(
        current_balances=balances,
        interest_rates=rates,
        monthly_contributions=base_monthly_contributions,
        months=projection_months,
        simulations=mc_simulations,
    )
    if scenario_enabled:
        scenario_mc = run_monte_carlo_projection(
            current_balances=balances,
            interest_rates=scenario_rates,
            monthly_contributions=scenario_contrib,
            months=projection_months,
            simulations=mc_simulations,
        )

    goal_months = max(int(goal_years) * 12, 0)
    invest_total = float(sum(balances.get(cat, 0.0) for cat in INVESTMENT_CATEGORIES))
    weighted_rate = 0.0
    if invest_total > 0:
        weighted_rate = float(
            sum(balances.get(cat, 0.0) * rates.get(cat, 0.0) for cat in INVESTMENT_CATEGORIES) / invest_total
        )
    monthly_rate = weighted_rate / 12.0
    current_monthly = float(sum(base_monthly_contributions.values()))
    extra_needed = required_extra_monthly(
        current_total=total,
        goal_amount=float(goal_amount),
        goal_months=goal_months,
        monthly_rate=monthly_rate,
        current_monthly=current_monthly,
    )

    goal_proj = deterministic_projection_total(
        current_balances=balances,
        interest_rates=rates,
        monthly_contributions=base_monthly_contributions,
        months=goal_months,
    )
    goal_projection_total = float(goal_proj["value"].iloc[-1]) if not goal_proj.empty else total

    insights = generate_insights(
        balances=balances,
        monthly_savings=avg_savings,
        interest_rates=rates,
        streak=streak,
    )

    return {
        "has_data": True,
        "latest": {"as_of": as_of.isoformat(), "balances": balances, "total": total},
        "timeseries": {
            "daily": daily.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        },
        "monthly": {
            "values": month_vals.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
            "changes_long": changes_long.to_dict(orient="records"),
            "avg_savings": avg_savings,
            "streak": {"current": streak.current, "best": streak.best},
        },
        "kpis": {
            "liquidity_total": liquidity_total,
            "hhi": hhi,
            "effective_positions": effective_positions,
            "weighted_risk": weighted_risk,
            "weighted_liquidity": weighted_liquidity,
            "weighted_expected_return": weighted_expected_return,
            "max_drawdown": mdd,
            "cagr_total": cagr_total,
            "vol_monthly": vol_monthly,
            "annual_passive_income": annual_passive_income(balances, rates),
        },
        "allocation": allocation,
        "risk_liquidity": risk_rows,
        "correlation": correlation,
        "projection": {
            "years": int(projection_years),
            "baseline": baseline_proj.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
            "scenario": (
                scenario_proj.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
                if scenario_proj is not None and not scenario_proj.empty
                else None
            ),
            "monte_carlo": {
                "baseline": baseline_mc.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
                "scenario": (
                    scenario_mc.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
                    if scenario_mc is not None and not scenario_mc.empty
                    else None
                ),
            },
        },
        "goal": {
            "target_amount": float(goal_amount),
            "years": int(goal_years),
            "projection_total": goal_projection_total,
            "extra_monthly_needed": float(extra_needed),
            "on_track": goal_projection_total >= float(goal_amount) if float(goal_amount) > 0 else True,
        },
        "insights": insights,
    }
