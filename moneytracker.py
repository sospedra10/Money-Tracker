import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np


class FinancialDataManager:
    """Handles data persistence and basic operations for financial tracking"""
    
    def __init__(self, data_file: str = "financial_data.json"):
        self.data_file = data_file
        self.categories = ["Bank", "Remuneration Account", "ETFs and Stocks", 
                          "Real Estate", "Crypto", "Others"]
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
            "amount": amount
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
    
    def calculate_latest_totals(self) -> pd.DataFrame:
        """Calculate the most recent total for each category"""
        history_df = self.data_manager.get_history_dataframe()
        if history_df.empty:
            return pd.DataFrame({
                "category": self.categories,
                "amount": [0] * len(self.categories),
                "date": [datetime.now()] * len(self.categories)
            })
        
        latest_totals = history_df.groupby("category").apply(
            lambda x: x.sort_values("date").iloc[-1]
        ).reset_index(drop=True)
        
        return latest_totals[["category", "amount", "date"]]
    
    def get_current_amounts_by_category(self) -> Dict[str, float]:
        """Get current amount for each category as dictionary"""
        latest_totals = self.calculate_latest_totals()
        amounts_dict = {}
        
        for category in self.categories:
            category_data = latest_totals[latest_totals["category"] == category]
            if not category_data.empty:
                amounts_dict[category] = category_data.iloc[0]["amount"]
            else:
                amounts_dict[category] = 0.0
        
        return amounts_dict
    
    def prepare_chart_data(self) -> pd.DataFrame:
        """Prepare historical data for charting with proper aggregation"""
        history_df = self.data_manager.get_history_dataframe()
        if history_df.empty:
            return pd.DataFrame()
        
        # Group by category and date, sum amounts
        aggregated_data = history_df.groupby(["category", "date"])["amount"].sum().reset_index()
        aggregated_data = aggregated_data.sort_values("date")
        
        # Pivot data to have categories as columns
        chart_data = pd.DataFrame()
        for category in aggregated_data['category'].unique():
            category_data = aggregated_data[
                aggregated_data['category'] == category
            ].set_index('date')['amount']
            category_data = category_data.rename(category)
            chart_data = pd.merge(chart_data, category_data, 
                                left_index=True, right_index=True, 
                                how='outer', suffixes=('', f'_{category}'))
        
        # Forward fill missing values and calculate total
        chart_data = chart_data.fillna(method='ffill').fillna(0)
        chart_data['Total'] = chart_data.sum(axis=1)
        
        return chart_data
    
    def calculate_monthly_savings(self, chart_data: pd.DataFrame, 
                                category: str = "Total", 
                                months_to_analyze: int = 1) -> float:
        """Calculate average monthly savings for a specific category"""
        if chart_data.empty or category not in chart_data.columns:
            return 0.0
        
        # Calculate differences (savings/expenses)
        savings_data = chart_data[category].diff().dropna().reset_index()
        
        # Group by month (first 7 characters of date: YYYY-MM)
        savings_data['date'] = savings_data['date'].astype(str).str[:7]
        monthly_savings = savings_data.groupby('date')[category].sum().reset_index()
        monthly_savings['date'] = pd.to_datetime(monthly_savings['date'])
        
        # Get last N months
        recent_savings = monthly_savings.sort_values('date').tail(months_to_analyze)
        
        if recent_savings.empty:
            return 0.0
        
        total_savings = recent_savings[category].sum()
        return total_savings / months_to_analyze
    
    def calculate_compound_interest(self, principal: float, annual_rate_percent: float,
                                  months: int, monthly_contribution: float = 0.0) -> float:
        """Calculate compound interest with monthly contributions"""
        monthly_rate = (annual_rate_percent / 100) / 12
        
        # Future value of initial principal
        future_value_principal = principal * (1 + monthly_rate) ** months
        
        # Future value of monthly contributions
        if monthly_rate != 0:
            future_value_contributions = monthly_contribution * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            )
        else:
            future_value_contributions = monthly_contribution * months
        
        return future_value_principal + future_value_contributions
    
    def project_future_wealth(self, current_amounts: Dict[str, float],
                            interest_rates: Dict[str, float],
                            monthly_contributions: Dict[str, float],
                            months: int) -> Dict[str, float]:
        """Project future wealth for each investment category"""
        future_amounts = {}
        
        investment_categories = ["Remuneration Account", "Real Estate", "ETFs and Stocks", "Bank"]
        
        for category in investment_categories:
            current_amount = current_amounts.get(category, 0)
            annual_rate = interest_rates.get(category, 0) * 100
            monthly_contribution = monthly_contributions.get(category, 0)
            
            future_amount = self.calculate_compound_interest(
                current_amount, annual_rate, months, monthly_contribution
            )
            future_amounts[category] = max(future_amount, 0)  # Ensure non-negative
        
        return future_amounts
    
    def calculate_annual_income(self, amounts: Dict[str, float], 
                              interest_rates: Dict[str, float]) -> float:
        """Calculate annual passive income from investments"""
        annual_income = 0
        for category, amount in amounts.items():
            rate = interest_rates.get(category, 0)
            annual_income += amount * rate
        return annual_income


class FinancialVisualizer:
    """Handles all chart and visualization creation"""
    
    def __init__(self, analyzer: FinancialAnalyzer):
        self.analyzer = analyzer
    
    def create_category_bar_chart(self, latest_totals: pd.DataFrame) -> go.Figure:
        """Create bar chart showing current totals by category"""
        fig = px.bar(
            latest_totals, 
            x='Category', 
            y='Total Amount', 
            title='Current Totals by Category',
            text='Total Amount'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400, template='plotly_white')
        return fig
    
    def create_time_series_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create time series chart showing amounts over time"""
        if chart_data.empty:
            return go.Figure()
        
        chart_data_reset = chart_data.reset_index()
        chart_data_reset['date'] = chart_data_reset['date'].astype(str).str[:10]
        chart_data_daily = chart_data_reset.groupby('date').last().reset_index()
        chart_data_daily['date'] = pd.to_datetime(chart_data_daily['date'])
        
        fig = px.line(
            chart_data_daily,
            x='date',
            y=chart_data_daily.columns.drop('date'),
            title='📊 Portfolio Value Over Time',
            markers=True
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            title_font_size=22,
            title_x=0.05,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            legend_title_text='Category'
        )
        
        fig.update_traces(line=dict(width=2), marker=dict(size=5))
        return fig
    
    def create_trend_prediction_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create chart with trend line and future predictions"""
        if chart_data.empty:
            return go.Figure()
        
        # Prepare data
        amount_over_time = chart_data.copy().reset_index()
        amount_over_time['date'] = amount_over_time['date'].astype(str).str[:10]
        amount_over_time = amount_over_time.groupby('date').last().reset_index()
        
        # Linear regression for trend
        X = np.arange(len(amount_over_time)).reshape(-1, 1)
        y = amount_over_time['Total'].values.reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        # Future prediction
        n_future_days = 7
        X_future = np.arange(len(amount_over_time), 
                           len(amount_over_time) + n_future_days).reshape(-1, 1)
        y_future_pred = reg.predict(X_future)
        
        # Generate future dates
        last_date = pd.to_datetime(amount_over_time['date'].iloc[-1])
        future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') 
                       for i in range(1, n_future_days + 1)]
        
        # Create figure
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=amount_over_time['date'],
            y=amount_over_time['Total'],
            mode='lines+markers',
            name='Actual Total',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6)
        ))
        
        # Trend line
        fig.add_trace(go.Scatter(
            x=amount_over_time['date'],
            y=y_pred.flatten(),
            mode='lines',
            name='Trend Line',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # Future prediction
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_future_pred.flatten(),
            mode='lines+markers',
            name='Prediction',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='📈 Portfolio Value with Trend Analysis',
            height=500,
            template='plotly_white',
            title_font_size=24,
            title_x=0.05,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title='Date',
            yaxis_title='Amount ($)'
        )
        
        return fig
    
    def create_future_projection_chart(self, future_values: List[float]) -> go.Figure:
        """Create chart showing future wealth projection"""
        months = list(range(len(future_values)))
        
        fig = px.line(
            x=months,
            y=future_values,
            title='Future Wealth Projection',
            labels={'x': 'Months from Now', 'y': 'Projected Wealth ($)'},
            markers=True
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            title_font_size=20
        )
        
        return fig


    def create_monthly_savings_bar_chart(self, chart_data: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing savings for each month."""
        if chart_data.empty or 'Total' not in chart_data.columns:
            return go.Figure()

        # Calculate daily change and then sum by month
        monthly_savings = chart_data['Total'].diff().resample('M').sum().reset_index()
        monthly_savings.columns = ['Month', 'Savings']
        last_month = chart_data['Total'].resample('M').last().reset_index()
        # st.write(chart_data)
        # st.write('mine')
        # st.write(last_month)
        # st.write(last_month.diff())
        # st.write('new')
        # st.write(monthly_savings)
        # Format month for better display
        monthly_savings['Month'] = monthly_savings['Month'].dt.strftime('%Y-%b')

        fig = px.bar(
            monthly_savings,
            x='Month',
            y='Savings',
            title='💰 Monthly Savings Breakdown',
            text='Savings'
        )
        
        fig.update_traces(
            texttemplate='$%{text:,.0f}', 
            textposition='outside',
            marker_color=['green' if s >= 0 else 'red' for s in monthly_savings['Savings']]
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            title_font_size=22,
            yaxis_title='Savings Amount ($)'
        )
        
        return fig


class FinancialTrackerApp:
    """Main application class that orchestrates the Streamlit interface"""
    
    def __init__(self):
        self.data_manager = FinancialDataManager()
        self.analyzer = FinancialAnalyzer(self.data_manager)
        self.visualizer = FinancialVisualizer(self.analyzer)
        
        # Default interest rates (as decimals)
        self.default_rates = {
            "Remuneration Account": 0.025,
            "Real Estate": 0.10,
            "ETFs and Stocks": 0.065,
            "Bank": 0.0,
            "Crypto": 0.01,
            "Others": 0.0
        }
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(page_title="💰 Money Tracker", layout="wide")
        st.title("💰 Financial Portfolio Tracker")
        
        # Custom CSS for better metric display
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 25px !important;
            }  
            </style>
        """, unsafe_allow_html=True)
    
    def create_sidebar(self) -> Dict[str, float]:
        """Create sidebar with input controls and return interest rates"""
        st.sidebar.header("💼 Update Portfolio")
        
        # Category update section
        selected_category = st.sidebar.selectbox("Category", self.data_manager.categories)
        amount = st.sidebar.number_input(
            "Total Amount", 
            min_value=0.0, 
            max_value=1e6, 
            step=0.01, 
            format="%.2f"
        )
        
        if st.sidebar.button("Update Category"):
            self.data_manager.add_category_entry(selected_category, amount)
            st.sidebar.success(f"✅ Updated {selected_category} to ${amount:,.2f}")
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.header("📈 Interest Rates (%)")
        
        # Interest rate inputs
        interest_rates = {}
        rate_inputs = [
            ("Remuneration Account", 2.5),
            ("Real Estate", 10.0),
            ("ETFs and Stocks", 6.5)
        ]
        
        for category, default_value in rate_inputs:
            rate_percent = st.sidebar.number_input(
                f"{category}", 
                min_value=0.0, 
                max_value=100.0, 
                step=0.1, 
                value=default_value, 
                format="%.2f"
            )
            interest_rates[category] = rate_percent / 100.0
        
        # Add zero rates for non-investment categories
        for category in ["Bank", "Crypto", "Others"]:
            interest_rates[category] = 0.0
        
        st.sidebar.markdown("---")
        st.sidebar.caption("📊 Track your financial growth over time!")
        
        return interest_rates
    
    def display_main_metrics(self, current_amounts: Dict[str, float], 
                           interest_rates: Dict[str, float], 
                           monthly_savings: Dict[str, float]):
        """Display main portfolio metrics"""
        total_amount = sum(current_amounts.values())
        annual_income = self.analyzer.calculate_annual_income(current_amounts, interest_rates)
        monthly_income = annual_income / 12
        
        # Main metric
        st.metric(
            label="💰 Total Portfolio Value", 
            value=f"${total_amount:,.0f}", 
            delta=f"${monthly_income:,.0f}/month (${annual_income:,.0f}/year)",
            border=True
        )
        
        # Future projections
        st.subheader("🔮 Future Projections")
        projection_years = [1, 2, 5, 10, 25]
        cols = st.columns(len(projection_years))
        
        for i, years in enumerate(projection_years):
            months = years * 12
            future_amounts = self.analyzer.project_future_wealth(
                current_amounts, interest_rates, monthly_savings, months
            )
            
            # Calculate total future value including non-investment categories
            total_future = (sum(future_amounts.values()) + 
                          current_amounts["Crypto"] + current_amounts["Others"])
            
            future_annual_income = self.analyzer.calculate_annual_income(future_amounts, interest_rates)
            future_monthly_income = future_annual_income / 12
            
            gained = total_future - total_amount
            
            cols[i].metric(
                label=f"📅 {years} Year{'s' if years > 1 else ''}",
                value=f"${total_future:,.0f}",
                delta=f"+${gained/1000:,.0f}K | ${future_monthly_income:,.0f}/mo"
            )
    
    def display_category_metrics(self, current_amounts: Dict[str, float], 
                                interest_rates: Dict[str, float]):
        """Display individual category metrics"""
        st.subheader("📊 By Category")
        cols = st.columns(len(self.data_manager.categories))
        
        for i, category in enumerate(self.data_manager.categories):
            amount = current_amounts.get(category, 0)
            annual_income = amount * interest_rates.get(category, 0)
            monthly_income = annual_income / 12
            
            cols[i].metric(
                label=category,
                value=f"${amount:,.0f}",
                delta=f"${monthly_income:,.0f}/mo" if monthly_income > 0 else None,
                border=True
            )


    
    def display_savings_metrics(self, monthly_savings: Dict[str, float]):
        """Display monthly savings by category"""
        st.subheader("💵 Monthly Savings")
        total_monthly_savings = sum(monthly_savings.values())
        
        cols = st.columns(len(self.data_manager.categories) + 1)
        cols[0].metric(
            label="Total Monthly",
            value=f"${total_monthly_savings:,.0f}",
            border=True
        )
        
        for i, category in enumerate(self.data_manager.categories):
            cols[i + 1].metric(
                label=category,
                value=f"${monthly_savings.get(category, 0):,.0f}"
            )

    
    def run(self):
        """Main application entry point"""
        self.setup_page_config()
        
        # Create sidebar and get interest rates
        interest_rates = self.create_sidebar()
        
        # Get current data
        history_df = self.data_manager.get_history_dataframe()
        
        if history_df.empty:
            st.info("📝 No data available yet. Use the sidebar to add your first category update!")
            return
        
        # Calculate current amounts and savings
        current_amounts = self.analyzer.get_current_amounts_by_category()
        chart_data = self.analyzer.prepare_chart_data()
        
        # Calculate monthly savings for each category
        monthly_savings = {}
        for category in self.data_manager.categories:
            monthly_savings[category] = self.analyzer.calculate_monthly_savings(chart_data, category, months_to_analyze=1)
        
        # Display metrics
        self.display_main_metrics(current_amounts, interest_rates, monthly_savings)
        self.display_category_metrics(current_amounts, interest_rates)
        self.display_savings_metrics(monthly_savings)
        
        st.subheader("Monthly Savings History")
        monthly_savings_chart = self.visualizer.create_monthly_savings_bar_chart(chart_data)
        st.plotly_chart(monthly_savings_chart, use_container_width=True)

        # Parámetros
        start_date = pd.to_datetime("today").normalize()  # Hoy como fecha inicial
        max_months = 54
        date_range = pd.date_range(start=start_date, periods=max_months+1, freq='MS')  # Monthly Start

        # Simulación (usa tu lógica real aquí)
        future_projections = []
        total_current = sum(current_amounts.values())
        future_projections.append(total_current)

        for month in range(1, max_months + 1):
            future_amounts = self.analyzer.project_future_wealth(current_amounts, interest_rates, monthly_savings, month)
            total_future = (sum(future_amounts.values()) + current_amounts["Crypto"] + current_amounts["Others"])
            future_projections.append(total_future)

        # Crear DataFrame con fechas reales
        df = pd.DataFrame({
            "Date": date_range,
            "Projected Wealth (€)": future_projections
        })

        # Sección de visualización
        st.subheader("📈 Future Wealth Projection")

        # Métricas principales
        col1, col2 = st.columns(2)
        col1.metric("💰 Current Total Wealth", f"{future_projections[0]:,.0f} €")
        col2.metric("🚀 Projected in 54 Months", f"{future_projections[-1]:,.0f} €")

        # Plot interactivo con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Projected Wealth (€)"],
            mode='lines+markers',
            line=dict(color='royalblue', width=3),
            marker=dict(size=4),
            name="Wealth Over Time"
        ))
        # add another trace with just a line without the compound interest we use in the other one just summing the monthly savings
        # Just monthly savings
        X = np.arange(len(df)).reshape(-1, 1)
        y = [future_projections[0]] + [future_projections[0] + sum(monthly_savings.values()) * i for i in range(1, max_months + 1)]
        fig.add_trace(go.Scatter(
            x=df["Date"], y=y,
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name="Monthly Savings"
        ))

        # Anotaciones clave
        for i in [1, 3, 6, 12, 24, 36, 54]:
            fig.add_annotation(
                x=df["Date"][i],
                y=df["Projected Wealth (€)"][i],
                text=f"{int(df['Projected Wealth (€)'][i]):,} € ({df['Date'][i].strftime('%B')[:3]}.)",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )

        fig.update_layout(
            title="Wealth Growth Projection Over Time",
            xaxis_title="Date",
            yaxis_title="Total Wealth (€)",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabla resumen
        milestone_dates = df[df["Date"].dt.month.isin([1, 6, 12]) | df.index.isin([0, 12, 24, 36, 48, 54])]
        milestone_dates["Projected Wealth (€)"] = milestone_dates["Projected Wealth (€)"].apply(lambda x: f"{int(x):,} €")

        with st.expander("📊 Show Key Month Projections"):
            st.dataframe(milestone_dates.set_index("Date"), use_container_width=True)
        
        # Display charts
        st.header("📊 Portfolio Analysis")
        
        # Latest totals
        latest_totals = self.analyzer.calculate_latest_totals()
        st.subheader("Current Portfolio Breakdown")
        
        latest_df = pd.DataFrame(latest_totals)
        latest_df.columns = ["Category", "Total Amount", "Last Updated"]
        
        bar_chart = self.visualizer.create_category_bar_chart(latest_df)
        st.plotly_chart(bar_chart, use_container_width=True)
        
        # Time series analysis
        st.subheader("Historical Performance")
        
        # Portfolio over time
        time_series_chart = self.visualizer.create_time_series_chart(chart_data)
        st.plotly_chart(time_series_chart, use_container_width=True)
        
        # Trend analysis with predictions
        trend_chart = self.visualizer.create_trend_prediction_chart(chart_data)
        st.plotly_chart(trend_chart, use_container_width=True)


# Application entry point
if __name__ == "__main__":
    app = FinancialTrackerApp()
    app.run()