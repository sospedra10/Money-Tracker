import streamlit as st
import pandas as pd
import json
from datetime import datetime

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

# App layout
st.set_page_config(page_title="💰 Money Tracker", layout="wide")
st.title("💰 Money Tracker App")

# Initialize data
initialize_data()

# Main Metrics Section
data = load_data()
history_df = pd.DataFrame(data["history"])
if not history_df.empty:
    history_df["date"] = pd.to_datetime(history_df["date"])
    latest_totals = calculate_latest_totals()
    total_amount = latest_totals["amount"].sum()
    st.metric(label="Total Amount", value=f"${total_amount:,.0f}")

    # Individual category metrics
    cols = st.columns(len(categories))
    for i, category in enumerate(categories):
        category_data = latest_totals[latest_totals["category"] == category]
        if not category_data.empty:
            amount = category_data.iloc[0]["amount"]
            cols[i].metric(label=category, value=f"${amount:,.0f}")
else:
    st.metric(label="Total Amount", value="$0.00")
    cols = st.columns(len(categories))
    for i, category in enumerate(categories):
        cols[i].metric(label=category, value="$0.0")

# Sidebar: Add or update category totals
st.sidebar.header("Update Category Total")
selected_category = st.sidebar.selectbox("Category", categories)
amount = st.sidebar.number_input("Total Amount", min_value=0.0, max_value=1e6, step=0.01, format="%.2f")
if st.sidebar.button("Update"):
    add_or_update_category(selected_category, amount)
    st.sidebar.success(f"Updated {selected_category} total to {amount:.2f}")
    st.rerun()

# Main Dashboard
st.header("Dashboard")
if not history_df.empty:
    st.subheader("Transaction History")
    st.dataframe(history_df)

    st.subheader("Latest Category Totals")
    latest_totals_df = pd.DataFrame(latest_totals, columns=["category", "amount", "date"])
    latest_totals_df.columns = ["Category", "Total Amount", "Last Updated"]
    st.dataframe(latest_totals_df)

    st.bar_chart(latest_totals_df.set_index("Category")["Total Amount"])

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

    # Create a line chart showing both category-wise and total amounts
    st.line_chart(chart_data, width=600, height=300)

    st.subheader("Historical Data")
    for category in categories:
        st.subheader(f"{category} History")
        category_data = history_df[history_df["category"] == category]
        if not category_data.empty:
            st.line_chart(category_data.set_index("date")["amount"], width=600, height=300)
else:
    st.info("No data available yet. Update a category to get started.")

st.sidebar.markdown("---")
st.sidebar.caption("Track your financial growth over time!")
