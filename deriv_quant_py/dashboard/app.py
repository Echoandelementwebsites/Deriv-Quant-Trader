from dash import Dash, html, dcc, Output, Input, State, no_update
import dash
import dash_bootstrap_components as dbc
from deriv_quant_py.dashboard.components import create_sidebar, create_top_bar, create_market_grid, create_chart_area, create_logs_area
from deriv_quant_py.database import init_db, SignalLog, Trade
from deriv_quant_py.config import Config
from deriv_quant_py.shared_state import state
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy.orm import Session
import json

# Initialize DB connection for the frontend
SessionLocal = init_db(Config.DB_PATH)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Interval(id="interval-fast", interval=3000, n_intervals=0), # 3s updates for robustness
    dcc.Interval(id="interval-slow", interval=10000, n_intervals=0), # 10s updates for balance
    dcc.Store(id='selected-symbol', data='R_100'), # Default
    dcc.Store(id='view-mode', data='single'), # 'single' or 'grid'
    dcc.Store(id='grid-page', data=0),
    dcc.Store(id='ui-visible-symbols', data=['R_100']), # Init

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
                dbc.Col(create_market_grid(), width=3),
                dbc.Col(
                    [
                        # Toolbar
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("Single View", id="btn-view-single", color="primary", outline=True, active=True),
                                            dbc.Button("Grid View", id="btn-view-grid", color="primary", outline=True),
                                        ]
                                    ),
                                    width="auto"
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Button("Prev", id="btn-grid-prev", color="secondary", size="sm", className="me-2"),
                                            dbc.Button("Next", id="btn-grid-next", color="secondary", size="sm"),
                                        ],
                                        id="grid-nav-controls",
                                        style={"display": "none"} # Hidden by default
                                    ),
                                    width="auto"
                                ),
                            ],
                            className="mb-2 justify-content-between"
                        ),
                        html.Div(id="charts-container")
                    ],
                    width=9
                ),
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
        if not logs:
             return html.P("No logs available yet.")
        return [html.P(f"{log.timestamp.strftime('%H:%M:%S')} - {log.symbol} - {log.signal} - {log.reason}") for log in logs]
    except Exception as e:
        return html.P(f"Error fetching logs: {e}")
    finally:
        session.close()

@app.callback(
    Output("scanner-content", "children"),
    Input("interval-fast", "n_intervals")
)
def update_scanner(n):
    data = state.get_scanner_data()
    active_trades = state.get_active_trades()

    if not data:
        return "Scanning..."

    accordion_items = []
    for cat, assets in data.items():
        if not assets:
            continue

        list_items = []
        for asset in assets:
            symbol = asset['symbol']
            name = asset.get('name', symbol)

            # Label Content
            label_children = [html.Span(f"{symbol} - {name}")]

            # Red Dot if active trade
            if symbol in active_trades:
                label_children.append(
                    dbc.Badge("🔴", color="danger", pill=True, className="ms-2", style={"fontSize": "0.6em"})
                )

            item = dbc.ListGroupItem(
                label_children,
                id={'type': 'asset-item', 'index': symbol},
                action=True,
                style={"cursor": "pointer"}
            )
            list_items.append(item)

        accordion_items.append(
            dbc.AccordionItem(
                dbc.ListGroup(list_items, flush=True),
                title=cat
            )
        )

    return html.Div(
        dbc.Accordion(accordion_items, start_collapsed=False, flush=True),
        style={"maxHeight": "80vh", "overflowY": "scroll"}
    )

@app.callback(
    Output("connection-status", "children"),
    Input("master-switch", "value"),
    Input("risk-mult", "value")
)
def update_settings(switch_value, risk_val):
    is_active = "active" in switch_value
    state.set_trading_active(is_active)
    status_text = "Trading Active" if is_active else "Trading Stopped"
    return status_text

# --- Callbacks for View Mode & Grid ---

@app.callback(
    Output("view-mode", "data"),
    Output("btn-view-single", "active"),
    Output("btn-view-grid", "active"),
    Output("grid-nav-controls", "style"),
    Input("btn-view-single", "n_clicks"),
    Input("btn-view-grid", "n_clicks"),
    Input({'type': 'asset-item', 'index': dash.ALL}, 'n_clicks'),
    State("view-mode", "data")
)
def toggle_view_mode(n_single, n_grid, asset_clicks, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_mode, current_mode == 'single', current_mode == 'grid', {"display": "none"} if current_mode == 'single' else {"display": "block"}

    trig_id = ctx.triggered[0]['prop_id']

    mode = current_mode
    if "btn-view-single" in trig_id:
        mode = "single"
    elif "btn-view-grid" in trig_id:
        mode = "grid"
    elif "asset-item" in trig_id:
        # If any asset clicked, switch to single
        if any(asset_clicks):
             mode = "single"

    is_single = (mode == "single")
    nav_style = {"display": "none"} if is_single else {"display": "block"}

    return mode, is_single, not is_single, nav_style

@app.callback(
    Output("grid-page", "data"),
    Input("btn-grid-prev", "n_clicks"),
    Input("btn-grid-next", "n_clicks"),
    State("grid-page", "data")
)
def update_grid_page(prev_n, next_n, current_page):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_page

    trig_id = ctx.triggered[0]['prop_id']
    if "btn-grid-prev" in trig_id:
        return max(0, current_page - 1)
    elif "btn-grid-next" in trig_id:
        return current_page + 1
    return current_page

@app.callback(
    Output("selected-symbol", "data"),
    Input({'type': 'asset-item', 'index': dash.ALL}, 'n_clicks'),
    State("selected-symbol", "data")
)
def update_selected_symbol(n_clicks, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current

    # Check which was clicked
    trig_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        trig_obj = json.loads(trig_id_str)
        return trig_obj['index']
    except:
        return current

@app.callback(
    Output("ui-visible-symbols", "data"),
    Input("charts-container", "children"),
    State("view-mode", "data")
)
def dummy_sub_update(n, mode):
    return no_update


@app.callback(
    Output("charts-container", "children"),
    Output("ui-visible-symbols", "data", allow_duplicate=True),
    Input("interval-fast", "n_intervals"),
    Input("view-mode", "data"),
    Input("grid-page", "data"),
    Input("selected-symbol", "data"),
    prevent_initial_call=True
)
def update_charts_container(n, mode, page, selected_symbol):
    if mode == "single":
        fig = generate_chart(selected_symbol)
        content = dbc.Card(
            [dbc.CardHeader(f"Live Chart: {selected_symbol}"), dbc.CardBody(dcc.Graph(figure=fig))],
            color="dark", inverse=True
        )
        # Notify backend
        state.set_ui_visible_symbols([selected_symbol])
        return content, [selected_symbol]

    else: # Grid
        # Get all symbols
        scanner = state.get_scanner_data()
        all_symbols = []
        for cat, assets in scanner.items():
            for a in assets:
                all_symbols.append(a['symbol'])

        # Pagination
        start = page * 4
        end = start + 4
        visible_subset = all_symbols[start:end]

        if not visible_subset:
            return html.Div("No assets found on this page."), []

        graphs = []
        rows = []

        # 2x2 Grid using Bootstrap Rows/Cols

        # Row 1
        row1_cols = []
        for sym in visible_subset[:2]:
            fig = generate_chart(sym, height=350)
            row1_cols.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        rows.append(dbc.Row(row1_cols, className="mb-2"))

        # Row 2
        row2_cols = []
        for sym in visible_subset[2:4]:
            fig = generate_chart(sym, height=350)
            row2_cols.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        rows.append(dbc.Row(row2_cols))

        # Notify backend
        state.set_ui_visible_symbols(visible_subset)
        return html.Div(rows), visible_subset

def generate_chart(symbol, height=600):
    history = state.get_history(symbol)
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title=f"{symbol} (Loading...)",
            template="plotly_dark",
            height=height
        )
        return fig

    df = pd.DataFrame(history)
    if df.empty:
        return go.Figure(layout={'title': f'{symbol} (No Data)', 'template': 'plotly_dark', 'height': height})

    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')

    # Imports
    import pandas_ta as ta
    from plotly.subplots import make_subplots

    # Create Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f"{symbol} Price", "RSI"),
                        row_heights=[0.7, 0.3])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df['epoch'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price'
    ), row=1, col=1)

    # 2. EMA
    if len(df) > Config.EMA_PERIOD:
        ema = ta.ema(df['close'], length=Config.EMA_PERIOD)
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=ema, line=dict(color='orange', width=1), name=f'EMA {Config.EMA_PERIOD}'
        ), row=1, col=1)

    # 3. RSI
    if len(df) > Config.RSI_PERIOD:
        rsi = ta.rsi(df['close'], length=Config.RSI_PERIOD)
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=rsi, line=dict(color='purple', width=1), name='RSI'
        ), row=2, col=1)

        # RSI Levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=height,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    return fig
