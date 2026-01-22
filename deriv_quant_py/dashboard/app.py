from dash import Dash, html, dcc, Output, Input, State, no_update, dash_table
import dash
import dash_bootstrap_components as dbc
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
from deriv_quant_py.dashboard.components import create_sidebar, create_top_bar, create_market_grid, create_chart_area, create_logs_area
from deriv_quant_py.database import init_db, SignalLog, Trade, StrategyParams
from deriv_quant_py.config import Config
from deriv_quant_py.shared_state import state
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

# Initialize DB connection for the frontend
SessionLocal = init_db(Config.DB_PATH)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Interval(id="interval-fast", interval=3000, n_intervals=0), # 3s updates for robustness
    dcc.Interval(id="interval-slow", interval=10000, n_intervals=0), # 10s updates for balance
    dcc.Store(id='selected-symbol', data='R_100'), # Default
    dcc.Store(id='view-mode', data='single'), # 'single' or 'grid'
    dcc.Store(id='grid-page', data=0),
    dcc.Store(id='scanner-state'), # Persist scanner accordion state

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
                        html.Div([
                            create_chart_area(),
                            html.Div(id="grid-view-container", style={"display": "none"})
                        ])
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
                dcc.Dropdown(id="bt-symbol", placeholder="Select Symbol", style={'color': 'black'}),
            ], width=3),
            dbc.Col([
                 dbc.Button("Run Grid Search", id="bt-run-btn", color="primary", className="me-2"),
                 dbc.Button("Run Full System Scan", id="bt-scan-btn", color="danger", className="me-2"),
                 dbc.Button("Resume Scan", id="bt-resume-btn", color="warning", className="me-2"),
                 dbc.Button("Run AI Research", id="btn-ai-research", color="info", className="me-2"),
                 dbc.Button("Resume AI Research", id="btn-ai-resume", color="success", className="me-2"),
            ], width=9)
        ], className="mb-3"),

        # Scan Progress Bar
        dbc.Progress(id="scan-progress", value=0, label="", striped=True, animated=True, style={"height": "20px", "display": "none"}),
        html.Div(id="scan-status-text", className="mb-3 text-warning"),

        dcc.Loading(
            html.Div([
                # Obsolete chart removed
                html.Hr(),
                html.H4("Global Portfolio Overview"),

                # Portfolio Table Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Previous", id="pt-prev", color="secondary", size="sm", className="me-2"),
                        dbc.Button("Next", id="pt-next", color="secondary", size="sm"),
                        html.Span(" Page 1", id="pt-page-label", className="ms-2")
                    ], width=6, className="mb-2")
                ]),

                dash_table.DataTable(
                    id='portfolio-table',
                    columns=[
                        {'name': 'Asset', 'id': 'symbol'},
                        {'name': 'Strategy', 'id': 'strategy_type'},
                        {'name': 'Win Rate', 'id': 'win_rate', 'type': 'numeric', 'format': FormatTemplate.percentage(1)},
                        {'name': 'EV', 'id': 'expectancy', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},

                        # --- ADD THESE NEW COLUMNS ---
                        {'name': 'Kelly %', 'id': 'kelly', 'type': 'numeric', 'format': FormatTemplate.percentage(1)},
                        {'name': 'Max DD', 'id': 'max_drawdown', 'type': 'numeric', 'format': FormatTemplate.percentage(1)},
                        # -----------------------------

                        {'name': 'Last Optimized', 'id': 'timestamp'}
                    ],
                    data=[],
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white'
                    },
                    style_data_conditional=[
                        # Kelly Color Coding
                        {
                            'if': {'filter_query': '{kelly} >= 0.05', 'column_id': 'kelly'},
                            'backgroundColor': '#d4edda', 'color': 'green' # Green for > 5%
                        },
                        {
                            'if': {'filter_query': '{kelly} > 0 && {kelly} < 0.05', 'column_id': 'kelly'},
                            'backgroundColor': '#fff3cd', 'color': '#856404' # Yellow for 1-5%
                        },
                        # Max Drawdown Warning
                        {
                            'if': {'filter_query': '{max_drawdown} < -0.2', 'column_id': 'max_drawdown'},
                            'backgroundColor': '#f8d7da', 'color': '#721c24' # Red for High Drawdown
                        }
                    ],
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    page_action='custom',
                    page_current=0,
                    page_size=20
                )
            ])
        )
    ])

# Callbacks for Backtest

@app.callback(
    Output("bt-symbol", "options"),
    Input("interval-slow", "n_intervals")
)
def update_bt_symbol_options(n):
    data = state.get_scanner_data()
    options = []
    for cat, assets in data.items():
        for a in assets:
            options.append({'label': f"{a['symbol']} - {a.get('name')}", 'value': a['symbol']})
    return options

@app.callback(
    Output("scan-progress", "style", allow_duplicate=True),
    Output("scan-status-text", "children", allow_duplicate=True),
    Input("bt-run-btn", "n_clicks"),
    Input("bt-scan-btn", "n_clicks"),
    Input("bt-resume-btn", "n_clicks"),
    Input("btn-ai-research", "n_clicks"),
    Input("btn-ai-resume", "n_clicks"),
    State("bt-symbol", "value"),
    prevent_initial_call=True
)
def run_backtest_actions(n_grid, n_scan, n_resume, n_ai, n_ai_resume, symbol):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    trig_id = ctx.triggered[0]['prop_id']
    load_style = {"display": "flex", "height": "20px"}

    if "bt-run-btn" in trig_id:
        if not symbol:
             return no_update, no_update
        # Send Request to Backend
        state.set_backtest_request(symbol)
        return {"display": "none"}, "Request Sent..."

    elif "bt-scan-btn" in trig_id:
        # Trigger Full Scan
        state.set_backtest_request("FULL_SCAN")
        return load_style, "Starting Full Scan..."

    elif "bt-resume-btn" in trig_id:
        # Trigger Resume Scan
        state.set_backtest_request("FULL_SCAN_RESUME")
        return load_style, "Resuming Scan..."

    elif "btn-ai-research" in trig_id:
        # Trigger AI Research
        state.set_backtest_request("AI_RESEARCH_START")
        return load_style, "Starting AI Research..."

    elif "btn-ai-resume" in trig_id:
        # Trigger Resume AI Research
        state.set_backtest_request("AI_RESEARCH_RESUME")
        return load_style, "Resuming AI Research..."

    return no_update, no_update

@app.callback(
    Output("scan-progress", "value"),
    Output("scan-progress", "label"),
    Output("scan-status-text", "children"),
    Output("scan-progress", "style"),
    Input("interval-fast", "n_intervals"),
    prevent_initial_call=True
)
def update_scan_progress_bar(n):
    progress = state.get_scan_progress()
    if progress['status'] == 'idle':
        return 0, "", "", {"display": "none"}

    total = progress['total']
    current = progress['current']
    symbol = progress['current_symbol']
    status = progress['status']

    percent = (current / total * 100) if total > 0 else 0

    if status == 'complete':
        return 100, "100%", "Scan Complete!", {"display": "flex", "height": "20px"}

    label = f"{int(percent)}%"
    text = f"Scanning {symbol}... ({current}/{total})"
    return percent, label, text, {"display": "flex", "height": "20px"}


# Pagination Callback
@app.callback(
    Output('portfolio-table', 'data'),
    Output('pt-page-label', 'children'),
    Input('portfolio-table', 'page_current'),
    Input('portfolio-table', 'page_size'),
    Input('interval-slow', 'n_intervals'),
    Input('pt-prev', 'n_clicks'),
    Input('pt-next', 'n_clicks'),
    State('portfolio-table', 'page_current')
)
def update_portfolio_table(page_current, page_size, n, prev_clicks, next_clicks, current_page_state):
    if page_current is None: page_current = 0
    if page_size is None: page_size = 20

    try:
        with SessionLocal() as session:
            # Ensure the query grabs the new fields
            # Aliasing last_updated as timestamp to match the column definition
            stmt = text("SELECT symbol, strategy_type, win_rate, expectancy, kelly, max_drawdown, last_updated AS timestamp FROM strategy_params ORDER BY last_updated DESC")

            df = pd.read_sql(stmt, session.bind)

            # CRITICAL: Handle NULL values (New columns might be NULL for old rows)
            df['kelly'] = df['kelly'].fillna(0.0)
            df['max_drawdown'] = df['max_drawdown'].fillna(0.0)
            df['expectancy'] = df['expectancy'].fillna(0.0)

            # Pagination
            start = page_current * page_size
            end = start + page_size
            dff = df.iloc[start:end]

            return dff.to_dict('records'), f" Page {page_current + 1}"

    except Exception as e:
        print(f"Error fetching portfolio: {e}")
        return [], " Error"

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
    try:
        with SessionLocal() as session:
            logs = session.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(20).all()
            if not logs:
                 return html.P("No logs available yet.")
            return [html.P(f"{log.timestamp.strftime('%H:%M:%S')} - {log.symbol} - {log.signal} - {log.reason}") for log in logs]
    except Exception as e:
        return html.P(f"Error fetching logs: {e}")

@app.callback(
    Output("scanner-state", "data"),
    Input("market-scanner-accordion", "active_item"),
    prevent_initial_call=True
)
def save_scanner_state(active_item):
    return active_item

@app.callback(
    Output("scanner-content", "children"),
    Input("interval-fast", "n_intervals"),
    State("scanner-state", "data")
)
def update_scanner(n, saved_active_item):
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
                title=cat,
                item_id=cat  # Explicit ID for persistence
            )
        )

    return html.Div(
        dbc.Accordion(
            accordion_items,
            id="market-scanner-accordion",
            start_collapsed=False,
            flush=True,
            active_item=saved_active_item
        ),
        style={"maxHeight": "80vh", "overflowY": "scroll"}
    )

@app.callback(
    Output("connection-status", "children"),
    Output("balance-display", "children"),
    Input("master-switch", "value"),
    Input("risk-perc", "value"),
    Input("interval-fast", "n_intervals")
)
def update_settings(switch_value, risk_val, n):
    is_active = "active" in switch_value
    state.set_trading_active(is_active)

    # Update risk settings
    if risk_val is not None:
        state.set_risk_settings(risk_val, 50.0) # Maintaining default daily loss for now

    status_text = "Trading Active" if is_active else "Trading Stopped"

    # Update Balance
    balance = state.get_balance()
    balance_text = f"Balance: ${balance:,.2f}"

    return status_text, balance_text

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
    trig = ctx.triggered[0]
    # Ignore triggers with falsy values (e.g. initialization/reset to None)
    if not trig['value']:
        return current

    trig_id_str = trig['prop_id'].split('.')[0]
    try:
        trig_obj = json.loads(trig_id_str)
        return trig_obj['index']
    except:
        return current

@app.callback(
    Output("live-chart", "figure"),
    Output("live-chart-header", "children"),
    Output("live-chart-card", "style"),
    Input("interval-fast", "n_intervals"),
    Input("selected-symbol", "data"),
    Input("view-mode", "data"),
    prevent_initial_call=True
)
def update_single_chart_view(n, selected_symbol, mode):
    if mode != "single":
        return no_update, no_update, {"display": "none"}

    fig = generate_chart(selected_symbol)
    state.set_ui_visible_symbols([selected_symbol])
    return fig, f"Live Chart: {selected_symbol}", {"display": "block"}


@app.callback(
    Output("grid-view-container", "children"),
    Output("grid-view-container", "style"),
    Input("interval-fast", "n_intervals"),
    Input("grid-page", "data"),
    Input("view-mode", "data"),
    prevent_initial_call=True
)
def update_grid_view_content(n, page, mode):
    if mode != "grid":
        return no_update, {"display": "none"}

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
        return html.Div("No assets found on this page."), {"display": "block"}

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
    return html.Div(rows), {"display": "block"}

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

    # Ensure numeric types
    cols = ['open', 'high', 'low', 'close']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

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
        margin=dict(l=50, r=50, t=30, b=30),
        uirevision='constant' # Preserve zoom/pan
    )
    return fig
