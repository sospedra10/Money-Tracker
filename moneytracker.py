

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import streamlit as st


SAVINGS_MONTHS_TO_ANALYZE = 10
YEARS_PROJECTION = 5
MONTHS_PROJECTION = YEARS_PROJECTION * 12
MONTE_CARLO_SIMULATIONS = 300

INVESTMENT_CATEGORIES = [
    "Remuneration Account",
    "Real Estate",
    "ETFs and Stocks",
    "Bank",
    "Reenlever",
    "Staking",
]

NON_INVESTMENT_CATEGORIES = ["Crypto", "Others"]

RISK_LIQUIDITY_PROFILE = {
    "Bank": {"risk": 1, "liquidity": 5, "expected_return": 0.00},
    "Remuneration Account": {"risk": 1, "liquidity": 4, "expected_return": 0.02},
    "ETFs and Stocks": {"risk": 4, "liquidity": 4, "expected_return": 0.065},
    "Real Estate": {"risk": 3, "liquidity": 2, "expected_return": 0.11},
    "Crypto": {"risk": 5, "liquidity": 3, "expected_return": 0.15},
    "Reenlever": {"risk": 4, "liquidity": 4, "expected_return": 0.11},
    "Staking": {"risk": 3, "liquidity": 3, "expected_return": 0.03},
    "Others": {"risk": 2, "liquidity": 3, "expected_return": 0.02},
}

VOLATILITY_ASSUMPTIONS = {
    "Bank": 0.01,
    "Remuneration Account": 0.02,
    "ETFs and Stocks": 0.18,
    "Real Estate": 0.12,
    "Crypto": 0.60,
    "Reenlever": 0.18,
    "Staking": 0.25,
    "Others": 0.10,
}



class FinancialDataManager:
    """Handles data persistence and basic operations for financial tracking"""

    def __init__(self, data_file: str = "financial_data.json"):
        self.data_file = data_file
        self.categories = [
            "Bank",
            "Remuneration Account",
            "ETFs and Stocks",
            "Real Estate",
            "Crypto",
            "Reenlever",
            "Staking",
            "Others",
        ]
        self._initialize_data_file()

    def _initialize_data_file(self) -> None:
        """Create data file if it doesn't exist"""
        try:
            with open(self.data_file, "r") as file:
                json.load(file)
        except FileNotFoundError:
            initial_data = {"history": []}
            with open(self.data_file, "w") as file:
                json.dump(initial_data, file)

    def load_data(self) -> Dict:
        """Load financial data from JSON file"""
        with open(self.data_file, "r") as file:
            return json.load(file)

    def save_data(self, data: Dict) -> None:
        """Save financial data to JSON file"""
        with open(self.data_file, "w") as file:
            json.dump(data, file, indent=4)

    def add_category_entry(self, category: str, amount: float) -> None:
        """Add new entry for a specific category"""
        data = self.load_data()
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": category,
            "amount": amount,
        }
        data["history"].append(entry)
        self.save_data(data)

    def get_history_dataframe(self) -> pd.DataFrame:
        """Get transaction history as pandas DataFrame"""
        data = self.load_data()
        df = pd.DataFrame(data["history"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df



class FinancialAnalyzer:
    """Handles financial calculations and analysis"""

    def __init__(self, data_manager: FinancialDataManager):
        self.data_manager = data_manager
        self.categories = data_manager.categories
        self.investment_categories = INVESTMENT_CATEGORIES

    def calculate_latest_totals(self) -> pd.DataFrame:
        """Calculate the most recent total for each category"""
        history_df = self.data_manager.get_history_dataframe()
        if history_df.empty:
            return pd.DataFrame(
                {
                    "category": self.categories,
                    "amount": [0] * len(self.categories),
                    "date": [datetime.now()] * len(self.categories),
                }
            )

        history_df = history_df.sort_values("date")
        latest_totals = (
            history_df.groupby("category")
            .apply(lambda x: x.sort_values("date").iloc[-1])
            .reset_index(drop=True)
        )
        return latest_totals[["category", "amount", "date"]]

    def get_current_amounts_by_category(self) -> Dict[str, float]:
        """Get current amount for each category as dictionary"""
        latest_totals = self.calculate_latest_totals()
        amounts_dict: Dict[str, float] = {}

        for category in self.categories:
            category_data = latest_totals[latest_totals["category"] == category]
            if not category_data.empty:
                amounts_dict[category] = float(category_data.iloc[0]["amount"])
            else:
                amounts_dict[category] = 0.0

        return amounts_dict

    def prepare_chart_data(self) -> pd.DataFrame:
        """Prepare historical data for charting with proper aggregation"""
        history_df = self.data_manager.get_history_dataframe()
        if history_df.empty:
            return pd.DataFrame()

        aggregated = (
            history_df.groupby(["date", "category"])["amount"]
            .sum()
            .reset_index()
        )
        chart_data = (
            aggregated.pivot(index="date", columns="category", values="amount")
            .sort_index()
        )
        chart_data = chart_data.ffill().fillna(0.0)
        chart_data.index = pd.to_datetime(chart_data.index)
        chart_data.index.name = "date"
        chart_data["Total"] = chart_data.sum(axis=1)
        return chart_data


    def get_monthly_category_changes(self, chart_data: pd.DataFrame) -> pd.DataFrame:
        """Return monthly change per category for heatmaps and streaks"""
        if chart_data.empty:
            return pd.DataFrame(columns=["Month", "Category", "Savings"])

        monthly_series = chart_data.resample("M").last().ffill().fillna(0.0)
        monthly_changes = monthly_series.diff().fillna(0.0)
        monthly_changes.index = monthly_changes.index.to_period("M").to_timestamp()
        monthly_changes = monthly_changes.reset_index()
        month_column = monthly_changes.columns[0]
        monthly_changes = monthly_changes.rename(columns={month_column: "Month"})
        monthly_changes_long = monthly_changes.melt(
            id_vars="Month", var_name="Category", value_name="Savings"
        )
        monthly_changes_long["Month"] = monthly_changes_long["Month"].dt.strftime("%Y-%b")
        return monthly_changes_long

    def calculate_monthly_savings(
        self,
        chart_data: pd.DataFrame,
        category: str = "Total",
        months_to_analyze: int = SAVINGS_MONTHS_TO_ANALYZE,
    ) -> float:
        """Calculate average monthly savings for a specific category"""
        if chart_data.empty:
            return 0.0

        monthly_changes = self.get_monthly_category_changes(chart_data)
        category_changes = monthly_changes[monthly_changes["Category"] == category]
        if category_changes.empty:
            return 0.0

        recent = category_changes.tail(months_to_analyze)
        if recent.empty:
            return 0.0
        return float(recent["Savings"].mean())

    def calculate_compound_interest(
        self,
        principal: float,
        annual_rate_percent: float,
        months: int,
        monthly_contribution: float = 0.0,
    ) -> float:
        """Calculate compound interest with monthly contributions"""
        monthly_rate = (annual_rate_percent / 100.0) / 12.0

        future_value_principal = principal * (1 + monthly_rate) ** months

        if monthly_rate != 0:
            future_value_contributions = monthly_contribution * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            )
        else:
            future_value_contributions = monthly_contribution * months

        return future_value_principal + future_value_contributions

    def project_future_wealth(
        self,
        current_amounts: Dict[str, float],
        interest_rates: Dict[str, float],
        monthly_contributions: Dict[str, float],
        months: int,
    ) -> Dict[str, float]:
        """Project future wealth for each investment category"""
        future_amounts: Dict[str, float] = {}

        for category in self.investment_categories:
            current_amount = current_amounts.get(category, 0.0)
            annual_rate = interest_rates.get(category, 0.0) * 100.0
            monthly_contribution = monthly_contributions.get(category, 0.0)
            future_amount = self.calculate_compound_interest(
                current_amount,
                annual_rate,
                months,
                monthly_contribution,
            )
            future_amounts[category] = max(future_amount, 0.0)

        return future_amounts

    def calculate_annual_income(
        self, amounts: Dict[str, float], interest_rates: Dict[str, float]
    ) -> float:
        """Calculate annual passive income from investments"""
        annual_income = 0.0
        for category, amount in amounts.items():
            rate = interest_rates.get(category, 0.0)
            annual_income += amount * rate
        return annual_income


    def calculate_savings_streak(
        self, chart_data: pd.DataFrame, category: str = "Total"
    ) -> Tuple[int, int]:
        """Return current and best positive savings streak in months"""
        if chart_data.empty or category not in chart_data.columns:
            return 0, 0

        monthly_series = chart_data[category].resample("M").last().diff().dropna()
        if monthly_series.empty:
            return 0, 0

        best_streak = 0
        current_streak = 0
        temp_streak = 0

        for value in monthly_series:
            if value > 0:
                temp_streak += 1
                best_streak = max(best_streak, temp_streak)
            else:
                temp_streak = 0

        for value in reversed(monthly_series.tolist()):
            if value > 0:
                current_streak += 1
            else:
                break

        return current_streak, best_streak

    def generate_insights(
        self,
        current_amounts: Dict[str, float],
        monthly_savings: Dict[str, float],
        interest_rates: Dict[str, float],
        chart_data: pd.DataFrame,
    ) -> List[Dict[str, str]]:
        """Generate narrative insights for the dashboard"""
        insights: List[Dict[str, str]] = []
        total = sum(current_amounts.values())

        if total > 0:
            leader = max(current_amounts, key=current_amounts.get)
            leader_share = (current_amounts[leader] / total) * 100.0
            insights.append(
                {
                    "title": "Portfolio Leader",
                    "text": (
                        f"{leader} represents {leader_share:,.1f}% of your assets "
                        f"with €{current_amounts[leader]:,.0f}. Consider whether that level "
                        f"of concentration matches your comfort with risk."
                    ),
                }
            )

        positive_savers = {
            k: v for k, v in monthly_savings.items() if v > 0 and k != "Total"
        }
        if positive_savers:
            best_saver = max(positive_savers, key=positive_savers.get)
            insights.append(
                {
                    "title": "Top Saver",
                    "text": (
                        f"You're adding about €{positive_savers[best_saver]:,.0f} per month to {best_saver}. "
                        "Doubling down on that habit compounds quickly."
                    ),
                }
            )

        negative_savers = {
            k: v for k, v in monthly_savings.items() if v < 0 and k != "Total"
        }
        if negative_savers:
            toughest = min(negative_savers, key=negative_savers.get)
            insights.append(
                {
                    "title": "Potential Leak",
                    "text": (
                        f"{toughest} has been shrinking by about €{abs(negative_savers[toughest]):,.0f} "
                        "each month. Reviewing that allocation could free up cash for faster growers."
                    ),
                }
            )

        passive_income = {
            cat: current_amounts.get(cat, 0.0) * interest_rates.get(cat, 0.0)
            for cat in self.investment_categories
        }
        if any(value > 0 for value in passive_income.values()):
            top_income_cat = max(passive_income, key=passive_income.get)
            insights.append(
                {
                    "title": "Passive Income Engine",
                    "text": (
                        f"{top_income_cat} is driving about €{passive_income[top_income_cat]:,.0f} "
                        "in annual passive income. Small rate bumps or contributions here will have "
                        "outsized impact."
                    ),
                }
            )

        current_streak, best_streak = self.calculate_savings_streak(chart_data)
        if current_streak:
            insights.append(
                {
                    "title": "Savings Streak",
                    "text": (
                        f"You're on a {current_streak}-month positive savings streak "
                        f"(best so far: {best_streak} months). Keep it alive!"
                    ),
                }
            )

        return insights[:4]

    def calculate_required_extra_monthly(
        self,
        current_total: float,
        goal_amount: float,
        goal_months: int,
        monthly_rate: float,
        current_monthly: float,
    ) -> float:
        """Compute additional monthly amount needed to hit a target"""
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
        return shortfall / contribution_factor

    def run_monte_carlo_projection(
        self,
        current_amounts: Dict[str, float],
        interest_rates: Dict[str, float],
        monthly_contributions: Dict[str, float],
        months: int,
        simulations: int = MONTE_CARLO_SIMULATIONS,
    ) -> pd.DataFrame:
        """Monte Carlo simulation for portfolio growth"""
        total_current = sum(current_amounts.values())
        if total_current <= 0 or months <= 0 or simulations <= 0:
            return pd.DataFrame()

        rng = np.random.default_rng()
        projections = np.zeros((months + 1, simulations))
        projections[0, :] = total_current

        static_categories = [
            cat for cat in current_amounts if cat not in self.investment_categories
        ]
        static_total = sum(current_amounts.get(cat, 0.0) for cat in static_categories)

        for sim in range(simulations):
            category_values = {
                cat: current_amounts.get(cat, 0.0) for cat in self.investment_categories
            }
            for month in range(1, months + 1):
                monthly_total = static_total
                for cat in self.investment_categories:
                    value = category_values.get(cat, 0.0)
                    mean = interest_rates.get(cat, 0.0) / 12.0
                    annual_vol = VOLATILITY_ASSUMPTIONS.get(cat, 0.2)
                    monthly_vol = annual_vol / np.sqrt(12)
                    simulated_return = rng.normal(mean, monthly_vol)
                    contribution = monthly_contributions.get(cat, 0.0)
                    value = max(value * (1 + simulated_return) + contribution, 0.0)
                    category_values[cat] = value
                    monthly_total += value
                projections[month, sim] = monthly_total

        percentiles = np.percentile(projections, [5, 50, 95], axis=1)
        date_range = pd.date_range(
            start=pd.to_datetime("today").normalize(), periods=months + 1, freq="MS"
        )

        return pd.DataFrame(
            {
                "Date": date_range,
                "p05": percentiles[0],
                "p50": percentiles[1],
                "p95": percentiles[2],
            }
        )


class FinancialVisualizer:
    """Handles all chart and visualization creation"""

    def __init__(self, analyzer: FinancialAnalyzer):
        self.analyzer = analyzer

    def create_category_bar_chart(self, latest_totals: pd.DataFrame) -> go.Figure:
        """Create bar chart showing current totals by category"""
        fig = px.bar(
            latest_totals,
            x="Category",
            y="Total Amount",
            title="Current Totals by Category",
            text="Total Amount",
        )
        fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
        fig.update_layout(height=400, template="plotly_white")
        return fig

    def create_time_series_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create time series chart showing amounts over time"""
        if chart_data.empty:
            return go.Figure()

        chart_data_reset = chart_data.reset_index()
        chart_data_reset["date"] = chart_data_reset["date"].astype(str).str[:10]
        chart_data_daily = chart_data_reset.groupby("date").last().reset_index()
        chart_data_daily["date"] = pd.to_datetime(chart_data_daily["date"])

        fig = px.line(
            chart_data_daily,
            x="date",
            y=chart_data_daily.columns.drop("date"),
            title="?? Portfolio Value Over Time",
            markers=True,
        )

        fig.update_layout(
            height=500,
            template="plotly_white",
            title_font_size=22,
            title_x=0.05,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title="Date",
            yaxis_title="Amount (€)",
            legend_title_text="Category",
        )

        fig.update_traces(line=dict(width=2), marker=dict(size=5))
        return fig

    def create_trend_prediction_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create chart with trend line and future predictions"""
        if chart_data.empty:
            return go.Figure()

        amount_over_time = chart_data.copy().reset_index()
        amount_over_time["date"] = amount_over_time["date"].astype(str).str[:10]
        amount_over_time = amount_over_time.groupby("date").last().reset_index()

        X = np.arange(len(amount_over_time)).reshape(-1, 1)
        y = amount_over_time["Total"].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        n_future_days = 7
        X_future = np.arange(len(amount_over_time), len(amount_over_time) + n_future_days).reshape(
            -1, 1
        )
        y_future_pred = reg.predict(X_future)
        last_date = pd.to_datetime(amount_over_time["date"].iloc[-1])
        future_dates = [
            (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, n_future_days + 1)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=amount_over_time["date"],
                y=amount_over_time["Total"],
                mode="lines+markers",
                name="Actual Total",
                line=dict(color="royalblue", width=3),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=amount_over_time["date"],
                y=y_pred.flatten(),
                mode="lines",
                name="Trend Line",
                line=dict(color="orange", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=y_future_pred.flatten(),
                mode="lines+markers",
                name="Prediction",
                line=dict(color="green", width=2, dash="dot"),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title="?? Portfolio Value with Trend Analysis",
            height=500,
            template="plotly_white",
            title_font_size=24,
            title_x=0.05,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title="Date",
            yaxis_title="Amount (€)",
        )
        return fig


    def create_future_projection_chart(self, future_values: List[float]) -> go.Figure:
        """Create chart showing future wealth projection"""
        months = list(range(len(future_values)))
        fig = px.line(
            x=months,
            y=future_values,
            title="Future Wealth Projection",
            labels={"x": "Months from Now", "y": "Projected Wealth (€)"},
            markers=True,
        )
        fig.update_layout(height=400, template="plotly_white", title_font_size=20)
        return fig

    def create_monthly_savings_bar_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing savings for each month."""
        if chart_data.empty or "Total" not in chart_data.columns:
            return go.Figure()

        monthly_totals = chart_data["Total"].resample("M").last().ffill()
        if monthly_totals.empty:
            return go.Figure()

        monthly_savings = monthly_totals.diff().dropna().reset_index()
        monthly_savings.columns = ["Month", "Savings"]
        monthly_savings["Month"] = monthly_savings["Month"].dt.strftime("%Y-%b")

        fig = px.bar(
            monthly_savings,
            x="Month",
            y="Savings",
            title="?? Monthly Savings Breakdown",
            text="Savings",
        )
        fig.update_traces(
            texttemplate="€%{text:,.0f}",
            textposition="outside",
            marker_color=[
                "green" if value >= 0 else "red" for value in monthly_savings["Savings"]
            ],
        )
        fig.update_layout(
            height=500,
            template="plotly_white",
            title_font_size=22,
            yaxis_title="Savings Amount (€)",
        )
        return fig

    def create_risk_liquidity_scatter(
        self, current_amounts: Dict[str, float]
    ) -> go.Figure:
        """Visualize allocation across risk/liquidity quadrants"""
        data_rows = []
        for category, metrics in RISK_LIQUIDITY_PROFILE.items():
            balance = current_amounts.get(category, 0.0)
            if balance <= 0:
                continue
            data_rows.append(
                {
                    "Category": category,
                    "Risk": metrics["risk"],
                    "Liquidity": metrics["liquidity"],
                    "Balance": balance,
                    "Expected Return": metrics["expected_return"] * 100.0,
                }
            )

        if not data_rows:
            return go.Figure()

        df = pd.DataFrame(data_rows)
        fig = px.scatter(
            df,
            x="Liquidity",
            y="Risk",
            size="Balance",
            color="Expected Return",
            hover_name="Category",
            size_max=50,
            color_continuous_scale="Viridis",
            title="Risk vs. Liquidity Positioning",
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            xaxis=dict(dtick=1, title="Liquidity (1=Low, 5=High)"),
            yaxis=dict(dtick=1, title="Risk (1=Low, 5=High)"),
        )
        return fig


    def create_diversification_heatmap(self, chart_data: pd.DataFrame) -> go.Figure:
        """Show correlations between category returns"""
        if chart_data.empty:
            return go.Figure()

        categories = [cat for cat in self.analyzer.categories if cat in chart_data.columns]
        returns = chart_data[categories].pct_change().replace([np.inf, -np.inf], np.nan).dropna(
            how="all"
        )
        returns = returns.dropna(axis=1, how="all").dropna()
        if returns.empty:
            return go.Figure()

        corr_matrix = returns.corr()
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale="RdBu",
            range_color=(-1, 1),
            title="Diversification Heatmap (Correlation of Monthly Changes)",
        )
        fig.update_layout(
            height=500,
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def create_savings_heatmap(self, monthly_changes: pd.DataFrame) -> go.Figure:
        """Heatmap of monthly savings by category"""
        if monthly_changes.empty:
            return go.Figure()

        recent_months = (
            monthly_changes["Month"].drop_duplicates().tail(12).tolist()
        )
        heatmap_data = monthly_changes[monthly_changes["Month"].isin(recent_months)]
        pivot = heatmap_data.pivot(index="Category", columns="Month", values="Savings")
        desired_order = self.analyzer.categories.copy()
        if "Total" in pivot.index and "Total" not in desired_order:
            desired_order.append("Total")
        pivot = pivot.reindex(desired_order)
        pivot = pivot.fillna(0.0)

        fig = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(color="Monthly Change (€)"),
            title="Savings Consistency Heatmap (Last 12 Months)",
        )
        fig.update_layout(
            height=500,
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def create_monte_carlo_fan_chart(
        self,
        baseline_df: pd.DataFrame,
        scenario_df: Optional[pd.DataFrame] = None,
    ) -> go.Figure:
        """Render Monte Carlo percentile bands"""
        fig = go.Figure()
        if baseline_df.empty:
            return fig

        fig.add_trace(
            go.Scatter(
                x=baseline_df["Date"],
                y=baseline_df["p95"],
                line=dict(color="rgba(65, 105, 225, 0.2)", width=0),
                name="Baseline 95th",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Date"],
                y=baseline_df["p05"],
                line=dict(color="rgba(65, 105, 225, 0.2)", width=0),
                fill="tonexty",
                fillcolor="rgba(65, 105, 225, 0.2)",
                name="Baseline 5th-95th",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline_df["Date"],
                y=baseline_df["p50"],
                line=dict(color="royalblue", width=3),
                name="Baseline Median",
            )
        )

        if scenario_df is not None and not scenario_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=scenario_df["Date"],
                    y=scenario_df["p95"],
                    line=dict(color="rgba(220, 53, 69, 0.2)", width=0),
                    name="Scenario 95th",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scenario_df["Date"],
                    y=scenario_df["p05"],
                    line=dict(color="rgba(220, 53, 69, 0.2)", width=0),
                    fill="tonexty",
                    fillcolor="rgba(220, 53, 69, 0.2)",
                    name="Scenario 5th-95th",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scenario_df["Date"],
                    y=scenario_df["p50"],
                    line=dict(color="crimson", width=3, dash="dash"),
                    name="Scenario Median",
                )
            )

        fig.update_layout(
            title="Monte Carlo Forecast Bands",
            template="plotly_white",
            height=520,
            xaxis_title="Date",
            yaxis_title="Portfolio Value (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig



class FinancialTrackerApp:
    """Main application class that orchestrates the Streamlit interface"""

    def __init__(self):
        self.data_manager = FinancialDataManager()
        self.analyzer = FinancialAnalyzer(self.data_manager)
        self.visualizer = FinancialVisualizer(self.analyzer)

        self.default_rates = {
            "Remuneration Account": 0.025,
            "Real Estate": 0.11,
            "ETFs and Stocks": 0.065,
            "Bank": 0.0,
            "Crypto": 0.01,
            "Reenlever": 0.11,
            "Staking": 0.03,
            "Others": 0.0,
        }

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(page_title="?? Money Tracker", layout="wide")
        st.title("?? Financial Portfolio Tracker")
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                font-size: 25px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def create_sidebar(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Create sidebar with input controls and return configurations"""
        st.sidebar.header("?? Update Portfolio")

        selected_category = st.sidebar.selectbox("Category", self.data_manager.categories)
        amount = st.sidebar.number_input(
            "Total Amount",
            min_value=0.0,
            max_value=1e6,
            step=0.01,
            format="%.2f",
        )

        if st.sidebar.button("Update Category"):
            self.data_manager.add_category_entry(selected_category, amount)
            st.sidebar.success(f"? Updated {selected_category} to €{amount:,.2f}")
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.header("?? Interest Rates (%)")

        interest_rates: Dict[str, float] = {}
        rate_inputs = [
            ("Remuneration Account", 2.5),
            ("Real Estate", 11.0),
            ("ETFs and Stocks", 6.5),
            ("Reenlever", 11.0),
            ("Staking", 3.0),
        ]

        for category, default_value in rate_inputs:
            rate_percent = st.sidebar.number_input(
                f"{category}",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                value=default_value,
                format="%.2f",
            )
            interest_rates[category] = rate_percent / 100.0

        for category in ["Bank", "Crypto", "Others"]:
            interest_rates[category] = 0.0

        st.sidebar.markdown("---")
        st.sidebar.header("?? What-if Scenario")
        scenario_enabled = st.sidebar.checkbox("Enable scenario comparison", value=False)
        savings_adjustment = st.sidebar.slider(
            "Monthly savings adjustment (%)", min_value=-50, max_value=200, value=20, step=5
        )
        extra_monthly = st.sidebar.number_input(
            "Extra monthly contribution (€)", min_value=0.0, step=50.0, value=0.0
        )
        interest_shift = st.sidebar.slider(
            "Interest rate shift (percentage points)",
            min_value=-5.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )

        scenario_settings = {
            "enabled": scenario_enabled,
            "savings_adjustment": savings_adjustment,
            "extra_monthly": extra_monthly,
            "interest_shift": interest_shift,
        }

        st.sidebar.markdown("---")
        st.sidebar.header("?? Goal Tracker")
        goal_amount = st.sidebar.number_input(
            "Target Amount (€)", min_value=0.0, step=1000.0, value=100000.0
        )
        goal_years = st.sidebar.slider(
            "Years to reach goal", min_value=1, max_value=30, value=5, step=1
        )
        goal_settings = {"amount": goal_amount, "years": goal_years}

        st.sidebar.markdown("---")
        st.sidebar.caption("?? Track your financial growth over time!")

        return interest_rates, scenario_settings, goal_settings


    def display_main_metrics(
        self,
        current_amounts: Dict[str, float],
        interest_rates: Dict[str, float],
        monthly_savings: Dict[str, float],
    ):
        """Display main portfolio metrics"""
        total_amount = sum(current_amounts.values())
        annual_income = self.analyzer.calculate_annual_income(current_amounts, interest_rates)
        monthly_income = annual_income / 12.0

        st.metric(
            label="?? Total Portfolio Value",
            value=f"€{total_amount:,.0f}",
            delta=f"€{monthly_income:,.0f}/month (€{annual_income:,.0f}/year)",
            border=True,
        )
        # add a metric for Liquidity Total: 
        liquidity_total = sum(current_amounts[cat] for cat in ["Bank", "Remuneration Account", "Reenlever"])
        st.metric(
            label="?? Liquidity Total (Bank, Remuneration Account, Reenlever)",
            value=f"€{liquidity_total:,.0f}",
        )


        st.subheader("?? Future Projections")
        projection_years = [1, 2, 5, 10, 25]
        cols = st.columns(len(projection_years))

        for i, years in enumerate(projection_years):
            months = years * 12
            future_amounts = self.analyzer.project_future_wealth(
                current_amounts,
                interest_rates,
                monthly_savings,
                months,
            )
            total_future = sum(future_amounts.values()) + sum(
                current_amounts.get(cat, 0.0) for cat in NON_INVESTMENT_CATEGORIES
            )
            future_annual_income = self.analyzer.calculate_annual_income(
                future_amounts, interest_rates
            )
            future_monthly_income = future_annual_income / 12.0
            gained = total_future - total_amount

            cols[i].metric(
                label=f"?? {years} Year{'s' if years > 1 else ''}",
                value=f"€{total_future:,.0f}",
                delta=f"+€{gained/1000:,.0f}K | €{future_monthly_income:,.0f}/mo",
            )

    def display_category_metrics(
        self,
        current_amounts: Dict[str, float],
        interest_rates: Dict[str, float],
    ):
        """Display individual category metrics"""
        st.subheader("?? By Category")
        cols = st.columns(len(self.data_manager.categories))

        for i, category in enumerate(self.data_manager.categories):
            amount = current_amounts.get(category, 0.0)
            annual_income = amount * interest_rates.get(category, 0.0)
            monthly_income = annual_income / 12.0
            cols[i].metric(
                label=category,
                value=f"€{amount:,.0f}",
                delta=f"€{monthly_income:,.0f}/mo" if monthly_income > 0 else None,
                border=True,
            )

    def display_savings_metrics(self, monthly_savings: Dict[str, float]):
        """Display monthly savings by category"""
        st.subheader("?? Monthly Savings")
        total_monthly_savings = sum(monthly_savings.get(cat, 0.0) for cat in monthly_savings)

        cols = st.columns(len(self.data_manager.categories) + 1)
        cols[0].metric(label="Total Monthly", value=f"€{total_monthly_savings:,.0f}", border=True)

        for i, category in enumerate(self.data_manager.categories):
            cols[i + 1].metric(
                label=category,
                value=f"€{monthly_savings.get(category, 0):,.0f}",
            )


    def run(self):
        """Main application entry point"""
        self.setup_page_config()

        interest_rates, scenario_settings, goal_settings = self.create_sidebar()
        history_df = self.data_manager.get_history_dataframe()

        if history_df.empty:
            st.info("?? No data available yet. Use the sidebar to add your first category update!")
            return

        current_amounts = self.analyzer.get_current_amounts_by_category()
        chart_data = self.analyzer.prepare_chart_data()

        monthly_savings: Dict[str, float] = {}
        for category in self.data_manager.categories:
            monthly_savings[category] = self.analyzer.calculate_monthly_savings(
                chart_data, category, months_to_analyze=SAVINGS_MONTHS_TO_ANALYZE
            )
        monthly_savings["Total"] = self.analyzer.calculate_monthly_savings(
            chart_data, "Total", months_to_analyze=SAVINGS_MONTHS_TO_ANALYZE
        )

        self.display_main_metrics(current_amounts, interest_rates, monthly_savings)

        

        self.display_category_metrics(current_amounts, interest_rates)
        self.display_savings_metrics(monthly_savings)

        st.subheader("Monthly Savings History")
        monthly_savings_chart = self.visualizer.create_monthly_savings_bar_chart(chart_data)
        st.plotly_chart(monthly_savings_chart, use_container_width=True)

        monthly_changes_df = self.analyzer.get_monthly_category_changes(chart_data)
        current_streak, best_streak = self.analyzer.calculate_savings_streak(chart_data)

        st.subheader("Savings Consistency")
        st.caption(
            f"Current positive savings streak: {current_streak} month{'s' if current_streak != 1 else ''} "
            f"(best streak: {best_streak} months)."
        )
        savings_heatmap = self.visualizer.create_savings_heatmap(monthly_changes_df)
        st.plotly_chart(savings_heatmap, use_container_width=True)

        insights = self.analyzer.generate_insights(
            current_amounts, monthly_savings, interest_rates, chart_data
        )
        if insights:
            st.subheader("?? Portfolio Insights")
            for insight in insights:
                st.markdown(f"**{insight['title']}**  \n{insight['text']}") 

        st.subheader("?? Future Wealth Projection")
        col1, col_space, col2, col3 = st.columns([2, 0.2, 3, 3])
        input_years_projection = col1.number_input(
            "Set Projected Years", min_value=1, max_value=50, value=YEARS_PROJECTION, step=1
        )
        months_projection = input_years_projection * 12

        start_date = pd.to_datetime("today").normalize()
        date_range = pd.date_range(start=start_date, periods=months_projection + 1, freq="MS")

        total_current = sum(current_amounts.values())
        future_projections = [total_current]

        base_monthly_contributions = {
            cat: monthly_savings.get(cat, 0.0) for cat in self.analyzer.investment_categories
        }
        scenario_monthly_contributions = base_monthly_contributions.copy()
        scenario_interest_rates = interest_rates.copy()

        if scenario_settings["enabled"]:
            adjustment_factor = 1 + scenario_settings["savings_adjustment"] / 100.0
            extra_monthly = scenario_settings["extra_monthly"]
            per_category_extra = (
                extra_monthly / len(self.analyzer.investment_categories)
                if self.analyzer.investment_categories
                else 0.0
            )
            for cat in scenario_monthly_contributions:
                scenario_monthly_contributions[cat] = max(
                    scenario_monthly_contributions[cat] * adjustment_factor + per_category_extra,
                    0.0,
                )
            shift = scenario_settings["interest_shift"] / 100.0
            for cat in scenario_interest_rates:
                scenario_interest_rates[cat] = max(scenario_interest_rates.get(cat, 0.0) + shift, 0.0)

        scenario_projections: List[float] = [total_current]

        for month in range(1, months_projection + 1):
            future_amounts = self.analyzer.project_future_wealth(
                current_amounts, interest_rates, base_monthly_contributions, month
            )
            total_future = sum(future_amounts.values()) + sum(
                current_amounts.get(cat, 0.0) for cat in NON_INVESTMENT_CATEGORIES
            )
            future_projections.append(total_future)

            if scenario_settings["enabled"]:
                scenario_future_amounts = self.analyzer.project_future_wealth(
                    current_amounts, scenario_interest_rates, scenario_monthly_contributions, month
                )
                scenario_total = sum(scenario_future_amounts.values()) + sum(
                    current_amounts.get(cat, 0.0) for cat in NON_INVESTMENT_CATEGORIES
                )
                scenario_projections.append(scenario_total)

        df_projection = pd.DataFrame(
            {
                "Date": date_range,
                "Projected Wealth (€)": future_projections,
            }
        )

        col2.metric("?? Current Total Wealth", f"{future_projections[0]:,.0f} €")
        col3.metric(
            f"?? Projected in {input_years_projection} Years ({months_projection} Months)",
            f"{future_projections[-1]:,.0f} €",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_projection["Date"],
                y=df_projection["Projected Wealth (€)"],
                mode="lines+markers",
                line=dict(color="royalblue", width=3),
                marker=dict(size=4),
                name="Wealth Over Time",
            )
        )

        straight_line = [
            future_projections[0] + sum(base_monthly_contributions.values()) * i
            for i in range(len(df_projection))
        ]
        fig.add_trace(
            go.Scatter(
                x=df_projection["Date"],
                y=straight_line,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Monthly Savings (No Growth)",
            )
        )

        if scenario_settings["enabled"] and len(scenario_projections) == len(future_projections):
            fig.add_trace(
                go.Scatter(
                    x=df_projection["Date"],
                    y=scenario_projections,
                    mode="lines",
                    line=dict(color="crimson", width=3, dash="dot"),
                    name="Scenario Projection",
                )
            )

        annotation_indices = [1, 3, 6, 12]
        annotation_indices += list(range(24, months_projection + 1, 12))
        annotation_indices = [i for i in annotation_indices if i < len(df_projection)]

        for idx in annotation_indices:
            fig.add_annotation(
                x=df_projection["Date"][idx],
                y=df_projection["Projected Wealth (€)"][idx],
                text=(
                    f"{int(df_projection['Projected Wealth (€)'][idx]):,} € "
                    f"({df_projection['Date'][idx].strftime('%b %Y')})"
                ),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )

        fig.update_layout(
            title="Wealth Growth Projection Over Time",
            xaxis_title="Date",
            yaxis_title="Total Wealth (€)",
            template="plotly_white",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        milestone_dates = df_projection[
            df_projection["Date"].dt.month.isin([1, 6, 12])
            | df_projection.index.isin([0, 12, 24, 36, 48, 54])
        ]
        milestone_dates["Projected Wealth (€)"] = milestone_dates["Projected Wealth (€)"].apply(
            lambda x: f"{int(x):,} €"
        )
        with st.expander("?? Show Key Month Projections"):
            st.dataframe(milestone_dates.set_index("Date"), use_container_width=True)


        baseline_mc = self.analyzer.run_monte_carlo_projection(
            current_amounts, interest_rates, base_monthly_contributions, months_projection
        )
        scenario_mc = None
        if scenario_settings["enabled"]:
            scenario_mc = self.analyzer.run_monte_carlo_projection(
                current_amounts,
                scenario_interest_rates,
                scenario_monthly_contributions,
                months_projection,
            )
        mc_fig = self.visualizer.create_monte_carlo_fan_chart(baseline_mc, scenario_mc)
        st.plotly_chart(mc_fig, use_container_width=True)

        goal_amount = goal_settings["amount"]
        goal_months = goal_settings["years"] * 12
        invest_balances = sum(current_amounts.get(cat, 0.0) for cat in self.analyzer.investment_categories)
        weighted_interest_rate = 0.0
        if invest_balances > 0:
            weighted_interest_rate = sum(
                current_amounts.get(cat, 0.0) * interest_rates.get(cat, 0.0)
                for cat in self.analyzer.investment_categories
            ) / invest_balances
        monthly_rate = weighted_interest_rate / 12.0
        current_monthly = sum(base_monthly_contributions.values())
        additional_needed = self.analyzer.calculate_required_extra_monthly(
            total_current, goal_amount, goal_months, monthly_rate, current_monthly
        )
        goal_projection_amounts = self.analyzer.project_future_wealth(
            current_amounts, interest_rates, base_monthly_contributions, goal_months
        )
        goal_projection_total = sum(goal_projection_amounts.values()) + sum(
            current_amounts.get(cat, 0.0) for cat in NON_INVESTMENT_CATEGORIES
        )

        st.subheader("?? Goal Tracker")
        goal_cols = st.columns([2, 1, 1])
        goal_cols[0].metric("Target Amount", f"€{goal_amount:,.0f}")
        goal_cols[1].metric("On-Track Projection", f"€{goal_projection_total:,.0f}")
        goal_cols[2].metric("Extra Monthly Needed", f"€{additional_needed:,.0f}")

        progress_ratio = 0.0 if goal_amount <= 0 else min(sum(current_amounts.values()) / goal_amount, 1.0)
        st.progress(progress_ratio)
        if goal_projection_total >= goal_amount:
            st.success("You're on track to reach your target within the selected timeframe!")
        else:
            shortfall = goal_amount - goal_projection_total
            st.warning(
                f"Projected shortfall of €{shortfall:,.0f}. Consider the suggested extra monthly contribution."
            )

        st.header("?? Portfolio Analysis")
        latest_totals = self.analyzer.calculate_latest_totals()
        st.subheader("Current Portfolio Breakdown")
        latest_df = pd.DataFrame(latest_totals)
        latest_df.columns = ["Category", "Total Amount", "Last Updated"]
        bar_chart = self.visualizer.create_category_bar_chart(latest_df)
        st.plotly_chart(bar_chart, use_container_width=True)

        risk_scatter = self.visualizer.create_risk_liquidity_scatter(current_amounts)
        st.plotly_chart(risk_scatter, use_container_width=True)

        st.subheader("Historical Performance")
        time_series_chart = self.visualizer.create_time_series_chart(chart_data)
        st.plotly_chart(time_series_chart, use_container_width=True)

        diversification_heatmap = self.visualizer.create_diversification_heatmap(chart_data)
        st.plotly_chart(diversification_heatmap, use_container_width=True)

        trend_chart = self.visualizer.create_trend_prediction_chart(chart_data)
        st.plotly_chart(trend_chart, use_container_width=True)


if __name__ == "__main__":
    app = FinancialTrackerApp()
    app.run()


