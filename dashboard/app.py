# Fraud Detection Dashboard - Flask & Dash

from flask import Flask, jsonify
import pandas as pd
import plotly.express as px
from dash import dcc, html
import dash

# 📌 Initialize Flask App
server = Flask(__name__)

# 📌 Initialize Dash App
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")
app.title = "Fraud Detection Dashboard"
data_path = "../data/processed/processed_fraud_data.csv"
fraud_data = pd.read_csv(data_path)

total_transactions = len(fraud_data)
fraud_cases = fraud_data[fraud_data["class"] == 1].shape[0]
fraud_rate = round((fraud_cases / total_transactions) * 100, 2)


browser_counts = fraud_data[fraud_data["class"] == 1]["browser"].value_counts().reset_index()
browser_counts.columns = ["browser", "count"]

country_counts = fraud_data[fraud_data["class"] == 1]["country"].value_counts().reset_index()
country_counts.columns = ["country", "count"]

# 📌 Create Dashboard Layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

    # Summary Statistics
    html.Div([
        html.H3(f"Total Transactions: {total_transactions}"),
        html.H3(f"Total Fraud Cases: {fraud_cases}"),
        html.H3(f"Fraud Rate: {fraud_rate}%")
    ], style={"textAlign": "center"}),

    # Fraud Transactions by Browser
    dcc.Graph(
        id="fraud-by-browser",
        figure=px.bar(
            browser_counts,
            x="browser",
            y="count",
            title="Fraud Transactions by Browser"
        )
    ),

    # Fraud Transactions by Country
    dcc.Graph(
        id="fraud-by-country",
        figure=px.bar(
            country_counts,
            x="country",
            y="count",
            title="Fraud Transactions by Country"
        )
    ),

    # Fraud Transactions by Hour of Day
    dcc.Graph(
        id="fraud-by-hour",
        figure=px.histogram(
            fraud_data[fraud_data["class"] == 1], 
            x="hour_of_day", 
            title="Fraud Transactions by Hour of Day"
        )
    ),
])


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
