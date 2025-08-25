import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px

# File to store financial data
data_file = "financial_data.json"

# Categories for money tracking
categories = ["Bank", "Remuneration Account", "ETFs and Stocks", "Real Estate", "Crypto", "Others"]

# Initialize data file if it doesn't exist
def initialize_data():
    try:
        with open(data_file, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {"history": []}
        with open(data_file, "w") as file:
            json.dump(data, file)
    return data

# Load data from file
def load_data():
    with open(data_file, "r") as file:
        return json.load(file)

# Save data to file
def save_data(data):
    with open(data_file, "w") as file:
        json.dump(data, file, indent=4)

# Add or update a category entry
def add_or_update_category(category, amount):
    data = load_data()
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "amount": amount
    }
    data["history"].append(entry)
    save_data(data)

# Calculate latest totals by category
def calculate_latest_totals():
    data = load_data()
    df = pd.DataFrame(data["history"])
    if df.empty:
        return {category: 0 for category in categories}
    latest_totals = df.groupby("category").apply(lambda x: x.sort_values("date").iloc[-1])
    return latest_totals.reset_index(drop=True)[["category", "amount", "date"]]


def plot_historical_data(historical_df, categories):
    st.subheader("📊 Historical Data")

    for category in categories:
        st.subheader(f"📂 {category} History")
        category_data = history_df[history_df["category"] == category]

        if not category_data.empty:
            # Asegurar que 'date' esté en formato string limpio
            category_data['date'] = category_data['date'].astype(str).str[:10]

            fig = px.line(
                category_data,
                x='date',
                y='amount',
                title=f"{category} Over Time",
                labels={'date': 'Date', 'amount': 'Amount'},
                markers=True
            )

            fig.update_layout(
                height=400,
                template='plotly_white',
                title_font_size=20,
                title_x=0.05,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title='',
                yaxis_title='',
            )

            fig.update_traces(line=dict(width=2.5), marker=dict(size=5))

            st.plotly_chart(fig, use_container_width=True)

def get_chart_data():
    # Get sum per category and date
    historical_amount_sum_each_last_category = history_df.groupby(["category", "date"])["amount"].sum().reset_index()
    historical_amount_sum_each_last_category = historical_amount_sum_each_last_category.sort_values("date")
    

    # Create a DataFrame with both category-wise and total amounts
    chart_data = pd.DataFrame()
    for category in historical_amount_sum_each_last_category['category'].unique():
        category_data = historical_amount_sum_each_last_category[
            historical_amount_sum_each_last_category['category'] == category
        ].set_index('date')['amount']
        category_data = category_data.rename(f'{category}')
        chart_data = pd.merge(chart_data, category_data, left_index=True, right_index=True, how='outer', suffixes=('', f'_{category}'))
        

    # Calculate total sum across all categories per date
    total_by_date = history_df.groupby("date").sum().reset_index().sort_values("date")
    
    # chart_data = chart_data.fillna(0)
    chart_data = chart_data.fillna(method='ffill').fillna(0)

    chart_data['Total'] = chart_data.sum(axis=1)
    return chart_data


def get_mean_savings(chart_data, by="Total"):
    # Get mean savings per month
    savings_by_month = chart_data[by].diff().dropna().reset_index()
    n = 7 # monthly
    savings_by_month['date'] = savings_by_month['date'].astype(str).apply(lambda x: x[:n])
    savings_by_month = savings_by_month.groupby('date')[by].sum().reset_index()
    savings_by_month['date'] = pd.to_datetime(savings_by_month['date'])
    # only get the last N_LAST_MONTHS months
    N_LAST_MONTHS = 1
    savings_by_month = savings_by_month.sort_values('date').tail(N_LAST_MONTHS)
    start_date = savings_by_month['date'].min()
    end_date = datetime.now()
    n_months = (end_date - start_date).days // 30 + 1
    total_savings = savings_by_month[by].sum()
    return total_savings / n_months



def compound_interest(principal: float,
                      annual_rate_pct: float,
                      months: int = 12,
                      monthly_contribution: float = 0.0) -> float:
    """
    Calcula el valor futuro de una inversión con interés compuesto mensual
    y aportaciones periódicas mensuales.

    :param principal: importe inicial
    :param annual_rate_pct: rentabilidad anual en porcentaje (p.ej. 5 para 5%)
    :param months: horizonte en meses
    :param monthly_contribution: cantidad aportada al final de cada mes
    :return: valor acumulado tras los meses indicados
    """
    # tasa mensual en decimales
    monthly_rate = (annual_rate_pct / 100) / 12
    # número total de meses
    n = months

    # valor futuro de la aportación inicial
    fv_principal = principal * (1 + monthly_rate) ** n

    # valor futuro de las aportaciones periódicas
    if monthly_rate != 0:
        fv_contributions = monthly_contribution * (((1 + monthly_rate) ** n - 1) / monthly_rate)
    else:
        fv_contributions = monthly_contribution * n

    return fv_principal + fv_contributions

def future_money(years=None, months=12):
    # remuneration_account_monthly_contribution = max(get_mean_savings(chart_data, by="Remuneration Account"), 0)
    # real_state_monthly_contribution = get_mean_savings(chart_data, by="Real Estate")
    # etf_monthly_contribution = get_mean_savings(chart_data, by="ETFs and Stocks")

    if months is None:
        if years is not None:
            months = years * 12
        else:
            raise ValueError("Either years or months must be specified")
        
    remuneration_account_monthly_contribution = all_monthly_savings["Remuneration Account"]
    real_state_monthly_contribution = all_monthly_savings["Real Estate"]
    etf_monthly_contribution = all_monthly_savings["ETFs and Stocks"]
    bank_monthly_contribution = all_monthly_savings["Bank"]
    
    renum_account_rents = max(compound_interest(actual_amounts_dic["Remuneration Account"], pcts["Remuneration Account"]*100, months, remuneration_account_monthly_contribution), 0)
    real_state_rents = compound_interest(actual_amounts_dic["Real Estate"], pcts["Real Estate"]*100, months, real_state_monthly_contribution)
    etf_rents = compound_interest(actual_amounts_dic["ETFs and Stocks"], pcts["ETFs and Stocks"]*100, months, etf_monthly_contribution)
    bank_rents = compound_interest(actual_amounts_dic["Bank"], pcts["Bank"]*100, months, bank_monthly_contribution)
    return renum_account_rents, real_state_rents, etf_rents, bank_rents


def predict_future_money(max_months=24):
    # Income per month
    months = range(1, max_months+1)
    money = []
    ## income in future
    for i, n_months in enumerate(months):
        renum_account_rents, real_state_rents, etf_rents, bank_rents = future_money(months=n_months) 
        # st.write('year money:', n_years, renum_account_rents, real_state_rents, etf_rents, bank_rents)
        invested_rents = renum_account_rents + real_state_rents + etf_rents + bank_rents

        general_amount = actual_amounts_dic["Crypto"] + actual_amounts_dic["Others"]
        future_amount = general_amount + invested_rents 
        money.append(future_amount)
    return money

# App layout
st.set_page_config(page_title="💰 Money Tracker", layout="wide")
st.title("💰 Money Tracker App")

# Initialize data
initialize_data()

# Sidebar: Add or update category totals
st.sidebar.header("Update Category Total")
selected_category = st.sidebar.selectbox("Category", categories)
amount = st.sidebar.number_input("Total Amount", min_value=0.0, max_value=1e6, step=0.01, format="%.2f")
if st.sidebar.button("Update"):
    add_or_update_category(selected_category, amount)
    st.sidebar.success(f"Updated {selected_category} total to {amount:.2f}")
    st.rerun()

st.sidebar.markdown("---")
remuneration_account_pct = st.sidebar.number_input("Remuneration Account Percentage", min_value=0.0, max_value=100.0, step=0.1, value=2.5, format="%.2f") / 100.0
real_state_pct = st.sidebar.number_input("Real State Percentage", min_value=0.0, max_value=100.0, step=0.1, value=13.0, format="%.2f") / 100.0
etf_pct = st.sidebar.number_input("ETF Percentage", min_value=0.0, max_value=100.0, step=0.1, value=6.5, format="%.2f") / 100.0
pcts = {"Remuneration Account": remuneration_account_pct, "Real Estate": real_state_pct, "ETFs and Stocks": etf_pct, "Bank": 0.0, "Crypto": 0.0, "Others": 0.0}

# Main Metrics Section
data = load_data()
history_df = pd.DataFrame(data["history"])
if not history_df.empty:
    history_df["date"] = pd.to_datetime(history_df["date"])
    latest_totals = calculate_latest_totals()
    total_amount = latest_totals["amount"].sum()

    chart_data = get_chart_data()
    st.subheader("Money by Timeframe")  

    all_monthly_savings = {category: get_mean_savings(chart_data, by=category) for category in categories}
         
    monthly_savings = sum(all_monthly_savings.values())

    
    actual_amounts_dic = {category: latest_totals[latest_totals["category"] == category].iloc[0]["amount"] for category in categories}
    remun_account_income = actual_amounts_dic["Remuneration Account"] * pcts["Remuneration Account"]
    real_state_income = actual_amounts_dic["Real Estate"] * pcts["Real Estate"]
    etf_income = actual_amounts_dic["ETFs and Stocks"] * pcts["ETFs and Stocks"]
    income = remun_account_income + real_state_income + etf_income
    st.metric(label="Total Amount", value=f"${total_amount:,.0f}", delta=f"${income/12:,.0f} (${income:,.0f})", border=True)

    

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
    
    
    
    # Income per month
    years = [1, 2, 5, 10, 25]
    
    cols = st.columns(len(years))

    ## income in future
    for i, n_years in enumerate(years):
        renum_account_rents, real_state_rents, etf_rents, bank_rents = future_money(years=n_years) 
        # st.write('year money:', n_years, renum_account_rents, real_state_rents, etf_rents, bank_rents)
        invested_rents = renum_account_rents + real_state_rents + etf_rents + bank_rents

        remun_account_income = renum_account_rents * pcts["Remuneration Account"]
        real_state_income = real_state_rents  * pcts["Real Estate"]
        etf_income = etf_rents * pcts["ETFs and Stocks"]
        new_rents = remun_account_income + real_state_income + etf_income

        general_amount = actual_amounts_dic["Crypto"] + actual_amounts_dic["Others"]
        # already_invested = actual_amounts_dic["Remuneration Account"] + actual_amounts_dic["Real Estate"] + actual_amounts_dic["ETFs and Stocks"]
        future_amount = general_amount + invested_rents #- already_invested #+ total_savings_by_month * 12 * n_years
        earned = future_amount - total_amount
        cols[i].metric(label=f"{n_years} Years Amount", value=f"${future_amount:,.0f} (+{earned/1000:,.0f}K)", delta=f"${new_rents/12:,.0f} (${new_rents:,.0f})")


    
    # Individual category metrics
    cols = st.columns(len(categories))
    for i, category in enumerate(categories):
        category_data = latest_totals[latest_totals["category"] == category]
        if not category_data.empty:
            amount = category_data.iloc[0]["amount"]
            income = amount * pcts[category]
            cols[i].metric(label=category, value=f"${amount:,.0f}", delta=f"${income/12:,.0f} (${income:,.0f})", border=True)


    st.subheader("Savings by Month")
    cols = st.columns(len(categories) + 1)
    cols[0].metric(label="Savings by Month", value=f"${monthly_savings:,.0f}", border=True)
    for i, category in enumerate(categories):
        cols[i+1].metric(label=category, value=f"${all_monthly_savings[category]:,.0f}")

    future_money = predict_future_money(max_months=54)
    # Add the first element as the actual money right now
    future_money = [total_amount] + future_money
    st.subheader("Future Money")
    st.line_chart(future_money)

    
            
else:
    st.metric(label="Total Amount", value="$0.00")
    cols = st.columns(len(categories))
    for i, category in enumerate(categories):
        cols[i].metric(label=category, value="$0.0")




# Main Dashboard
st.header("Dashboard")
if not history_df.empty:
    # st.subheader("Transaction History")
    # st.dataframe(history_df)

    st.subheader("Latest Category Totals")
    latest_totals_df = pd.DataFrame(latest_totals, columns=["category", "amount", "date"])
    latest_totals_df.columns = ["Category", "Total Amount", "Last Updated"]
    # st.dataframe(latest_totals_df)

    fig = px.bar(latest_totals_df, x='Category', y='Total Amount', title='Totals', text='Total Amount')
    st.plotly_chart(fig, use_container_width=True)

    # Show expenses by day in boxplots
    st.subheader("Money by Timeframe")  
    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
    with col1:
        transaction_type = st.selectbox("Transaction Type", ["All", "Expense", "Income"])
    with col2:
        timeframe = st.selectbox("Timeframe", ["Day", "Month"])  
    with col3:
        observe = st.selectbox("Observe", ["Total", "Remuneration Account", "Bank"])
    expenses_by_day = chart_data[observe].diff().dropna().reset_index()

    if transaction_type == "Expense":
        expenses_by_day = expenses_by_day[expenses_by_day[observe] < 0]
    elif transaction_type == "Income":
        expenses_by_day = expenses_by_day[expenses_by_day[observe] > 0]
    else:
        expenses_by_day = expenses_by_day

    # group by date by day
    n = 10 if timeframe == "Day" else 7
    expenses_by_day['date'] = expenses_by_day['date'].astype(str).apply(lambda x: x[:n])
    expenses_by_day = expenses_by_day.groupby('date')[observe].sum().reset_index()
    expenses_by_day['date'] = pd.to_datetime(expenses_by_day['date'])

    # Create bar plot of daily expenses
    fig = px.bar(expenses_by_day, x='date', y=observe, title=f'{'Daily' if timeframe == "Day" else 'Monthly'} Expenses', text=observe)
    st.plotly_chart(fig, use_container_width=True)
    
    # deepcopy of chart_data
    amount_over_time_data = chart_data.copy()
    amount_over_time_data = amount_over_time_data.reset_index()#
    # Use dates of each day but use the last value of the day
    amount_over_time_data['date'] = amount_over_time_data['date'].astype(str).apply(lambda x: x[:10])
    amount_over_time_data = amount_over_time_data.groupby('date').last().reset_index()
    

    

    # New
    
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Codificar fechas como números para regresión
    X = np.arange(len(amount_over_time_data)).reshape(-1, 1)
    y = amount_over_time_data['Total'].values.reshape(-1, 1)

    # Ajustar regresión lineal
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Proyección futura
    n_future = 7  # días a predecir
    X_future = np.arange(len(amount_over_time_data), len(amount_over_time_data) + n_future).reshape(-1, 1)
    y_future_pred = reg.predict(X_future)

    # Fechas futuras como texto
    last_date = pd.to_datetime(amount_over_time_data['date'].iloc[-1])
    future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, n_future + 1)]

    # Crear figura
    fig = go.Figure()

    # Línea real
    fig.add_trace(go.Scatter(
        x=amount_over_time_data['date'],
        y=amount_over_time_data['Total'],
        mode='lines+markers',
        name='Total Amount',
        line=dict(color='royalblue', width=3),
        marker=dict(size=6)
    ))

    # Línea de regresión (ajustada al histórico)
    fig.add_trace(go.Scatter(
        x=amount_over_time_data['date'],
        y=y_pred.flatten(),
        mode='lines',
        name='Trend (Linear Regression)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Proyección futura
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_future_pred.flatten(),
        mode='lines+markers',
        name='Prediction (Next Days)',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    # Layout
    fig.update_layout(
        title='📈 Amount Over Time with Trend and Prediction',
        width=950,
        height=500,
        template='plotly_white',
        title_font_size=24,
        title_x=0.05,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title='',
        yaxis_title='',
    )

    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
   # old
    # Plot con Plotly
    fig = px.line(
        amount_over_time_data,
        x='date',
        y='Total',
        title='📈 Amount Over Time',
        labels={'date': 'Date', 'Total': 'Total Amount'},
        markers=True
    )

    # Estética mejorada
    fig.update_layout(
        width=900,
        height=500,
        template='plotly_white',
        title_font_size=24,
        title_x=0.05,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title='',
        yaxis_title='',
    )

    fig.update_traces(line=dict(color='royalblue', width=3), marker=dict(size=6))
    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)





    # Create a line chart showing both category-wise and total amounts
    amount_over_time_data['date'] = pd.to_datetime(amount_over_time_data['date'])

    # Crear gráfico de líneas mejorado
    fig = px.line(
        amount_over_time_data,
        x='date',
        y=amount_over_time_data.columns.drop('date'),
        title='📊 Category-wise and Total Amounts Over Time',
        markers=True
    )

    # Estética y formato
    fig.update_layout(
        height=500,
        template='plotly_white',
        title_font_size=22,
        title_x=0.05,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title='',
        yaxis_title='',
        legend_title_text='Category'
    )

    fig.update_traces(line=dict(width=2), marker=dict(size=5))

    st.plotly_chart(fig, use_container_width=True)
    

    # Plot historical data
    plot_historical_data(history_df, categories)
        
else:
    st.info("No data available yet. Update a category to get started.")




st.sidebar.markdown("---")
st.sidebar.caption("Track your financial growth over time!")
