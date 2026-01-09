from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
from deriv_quant_py.dashboard.components import create_sidebar, create_top_bar, create_market_grid, create_chart_area, create_logs_area
from deriv_quant_py.database import init_db, SignalLog, Trade
from deriv_quant_py.config import Config
from deriv_quant_py.shared_state import state
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy.orm import Session

# Initialize DB connection for the frontend
SessionLocal = init_db(Config.DB_PATH)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Interval(id="interval-fast", interval=2000, n_intervals=0), # 2s updates for scanner/logs
    dcc.Interval(id="interval-slow", interval=10000, n_intervals=0), # 10s updates for balance
    dcc.Store(id='selected-symbol', data='R_100'), # Default

    create_sidebar(),

    html.Div(
        [
            create_top_bar(),
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')
        ],
        style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem"}
    )
])

# Page Layouts
def dashboard_layout():
    return html.Div([
        dbc.Row(
            [
                dbc.Col(create_market_grid(), width=4),
                dbc.Col(create_chart_area(), width=8),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(create_logs_area(), width=12),
            ],
            className="mt-3"
        )
    ])

def backtest_layout():
    return html.Div([
        html.H3("Backtesting Module"),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="bt-symbol", placeholder="Symbol (e.g. R_100)", value="R_100"),
            ], width=3),
            dbc.Col([
                 dbc.Button("Run Grid Search", id="bt-run-btn", color="primary")
            ], width=3)
        ], className="mb-3"),
        dcc.Loading(
            dcc.Graph(id="bt-heatmap")
        )
    ])

# Callbacks for Backtest
@app.callback(
    Output("bt-heatmap", "figure"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-symbol", "value")
)
def run_backtest_callback(n, symbol):
    if not n or not symbol:
        return go.Figure()

    # 1. Send Request to Backend
    state.set_backtest_request(symbol)

    # 2. Wait for result (This is bad practice in Dash callbacks to block,
    # but for simplicity we poll or return placeholder and use interval update?
    # Better: trigger interval to check result.)

    # Let's assume we return a "Loading" placeholder?
    # Actually, we can't block here easily.
    # We will trigger the request, but we need an Interval to pick up the result.
    # So we'll update the Graph via an Interval, not this button click directly?

    # Simplification: This callback just sets the request.
    # Another callback (Interval) checks for result and updates graph.
    return go.Figure(layout={'title': 'Request Sent... Waiting for result'})

@app.callback(
    Output("bt-heatmap", "figure", allow_duplicate=True),
    Input("interval-fast", "n_intervals"),
    prevent_initial_call=True
)
def check_backtest_result(n):
    res = state.get_backtest_result()
    if res:
        # Build Heatmap
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame(res)
        pivot = df.pivot(index='EMA', columns='RSI', values='WinRate')
        fig = px.imshow(pivot,
                        labels=dict(x="RSI Period", y="EMA Period", color="Win Rate (%)"),
                        title="Backtest Grid Search Results")
        return fig
    # Return nothing or existing figure?
    # returning no_update is better
    from dash import no_update
    return no_update

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/backtest':
        return backtest_layout()
    else:
        return dashboard_layout()

# Callbacks
@app.callback(
    Output("logs-content", "children"),
    Input("interval-fast", "n_intervals")
)
def update_logs(n):
    session = SessionLocal()
    try:
        logs = session.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(20).all()
        return [html.P(f"{log.timestamp.strftime('%H:%M:%S')} - {log.symbol} - {log.signal} - {log.reason}") for log in logs]
    finally:
        session.close()

@app.callback(
    Output("scanner-content", "children"),
    Input("interval-fast", "n_intervals")
)
def update_scanner(n):
    data = state.get_scanner_data()
    if not data:
        return "Scanning..."

    # Flatten or categorize
    items = []
    for cat, assets in data.items():
        if assets:
            items.append(html.H6(cat))
            items.extend([html.Div(f"{a['symbol']} - {a['name']}") for a in assets[:5]]) # Limit 5

    return html.Div(items)

@app.callback(
    Output("live-chart", "figure"),
    Input("interval-fast", "n_intervals"),
    State("selected-symbol", "data")
)
def update_chart(n, symbol):
    # Fetch history from shared state
    history = state.get_history(symbol)

    fig = go.Figure()

    if history:
        df = pd.DataFrame(history)
        df['epoch'] = pd.to_datetime(df['epoch'], unit='s')

        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df['epoch'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))

        # Calculate EMA overlay (if enough data)
        # Note: We can import ta here or trust backend passed it?
        # Backend doesn't pass indicators in history yet, just candles.
        # We calc locally for viz.
        import pandas_ta as ta
        if len(df) > Config.EMA_PERIOD:
            ema = ta.ema(df['close'], length=Config.EMA_PERIOD)
            fig.add_trace(go.Scatter(
                x=df['epoch'],
                y=ema,
                line=dict(color='orange', width=1),
                name=f'EMA {Config.EMA_PERIOD}'
            ))

        last_price = df['close'].iloc[-1]
        title = f"Live Chart: {symbol} | Last: {last_price}"
    else:
        title = f"Live Chart: {symbol} (Waiting for data...)"

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    return fig

@app.callback(
    Output("connection-status", "children"),
    Input("master-switch", "value"),
    Input("risk-mult", "value")
)
def update_settings(switch_value, risk_val):
    is_active = "active" in switch_value
    state.set_trading_active(is_active)

    # Also update risk
    # Note: Triggered by interval usually, but here by input
    # To avoid cyclic deps, we just set it.

    status_text = "Trading Active" if is_active else "Trading Stopped"
    return status_text # We are reusing the ID connection-status for simplicity, but cleaner to separate

# Run server is done in main.py
