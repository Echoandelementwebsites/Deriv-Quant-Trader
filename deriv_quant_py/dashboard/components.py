from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def create_sidebar():
    return html.Div(
        [
            html.H2("Deriv Quant", className="display-5"),
            html.Hr(),
            html.P("Settings", className="lead"),
            dbc.Input(id="api-token", placeholder="API Token", type="password", className="mb-2"),
            dbc.Checklist(
                options=[{"label": "Master Switch", "value": "active"}],
                value=[],
                id="master-switch",
                switch=True,
                className="mb-3"
            ),
            html.Label("Risk Percentage"),
            dbc.Input(id="risk-perc", type="number", value=1.0, className="mb-2"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("Dashboard", href="/", active="exact"),
                    dbc.NavLink("Backtest", href="/backtest", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "background-color": "#1e1e1e",
            "color": "white"
        },
    )

def create_top_bar():
    return dbc.Row(
        [
            dbc.Col(html.H4("Status: Disconnected", id="connection-status", className="text-danger"), width=4),
            dbc.Col(html.H4("Balance: $0.00", id="balance-display", className="text-success"), width=4),
        ],
        className="mb-4 mt-2"
    )

def create_market_grid():
    return dbc.Card(
        [
            dbc.CardHeader("Market Scanner"),
            dbc.CardBody(
                html.Div(id="scanner-content", children="Scanning...")
            )
        ],
        color="dark", inverse=True
    )

def create_chart_area():
    return dbc.Card(
        [
            dbc.CardHeader("Live Chart", id="live-chart-header"),
            dbc.CardBody(
                dcc.Graph(id="live-chart")
            )
        ],
        color="dark", inverse=True,
        id="live-chart-card"
    )

def create_logs_area():
    return dbc.Card(
        [
            dbc.CardHeader("Signal & Trade Logs"),
            dbc.CardBody(
                html.Div(id="logs-content", style={"maxHeight": "200px", "overflowY": "scroll"})
            )
        ],
        color="dark", inverse=True
    )
