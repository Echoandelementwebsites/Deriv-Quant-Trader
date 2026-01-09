# Deriv Quant Python Terminal

## Executive Summary

This project is a high-performance, asynchronous trading terminal designed for Deriv.com, migrated from a legacy Node-RED architecture to a production-ready **Python** system. It combines a low-latency **AsyncIO** backend for real-time market data processing and trade execution with a responsive **Dash** frontend for "Bloomberg-style" visualization and control.

## Core Features

*   **Triple Confluence Strategy**: Implements a robust "Classic Reversal" strategy combining:
    *   **Trend Filter**: 200 EMA (Price > EMA for Bull, Price < EMA for Bear).
    *   **Momentum**: RSI (14) Reversal Logic (< 30 Buy, > 70 Sell).
    *   **Pattern Recognition**: Engulfing, Hammer, Three Black Crows.
    *   **ADX Filter**: Ensures market is trending (ADX > 25) before trading.
*   **Risk Management Layer**:
    *   **Spread Filter**: Prevents trading when spreads are excessive.
    *   **Martingale Sizing**: Intelligent stake sizing based on previous trade outcomes.
    *   **Daily Loss Limit**: Hard stop if daily loss exceeds a configured threshold.
    *   **Master Switch**: Instant kill-switch via the UI.
*   **Real-Time Dashboard**:
    *   Dark-themed interface built with Dash & Plotly.
    *   Live Candlestick Chart with EMA overlay.
    *   Active Market Scanner.
    *   Live Trade & Signal Logs.
*   **Backtesting Module**: Integrated grid search to optimize RSI and EMA parameters.

## Architecture

The system uses a Dual-Thread Architecture:
1.  **Backend Thread (AsyncIO)**: Handles WebSocket connections to Deriv API, processes ticks/candles, executes strategy logic, and manages database writes.
2.  **Frontend Thread (Dash/Flask)**: Serves the web UI and reads from a thread-safe `SharedState` singleton to display real-time data without blocking the trading logic.

## Installation & Usage

### Prerequisites
*   Python 3.10+
*   Deriv API Token

### Local Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd deriv-quant-py
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r deriv_quant_py/requirements.txt
    ```

3.  **Configure Environment**:
    Copy the template and edit your API token:
    ```bash
    cp deriv_quant_py/.env deriv_quant_py/.env.local
    # Edit .env.local with your DERIV_TOKEN
    ```

4.  **Run the Application**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python deriv_quant_py/main.py
    ```
    Access the Dashboard at `http://127.0.0.1:8050`.

### Docker Deployment

1.  **Build the Image**:
    ```bash
    docker build -t deriv-quant .
    ```

2.  **Run the Container**:
    Pass your environment variables directly or via a file.
    ```bash
    docker run -p 8050:8050 -e DERIV_TOKEN=your_token_here deriv-quant
    ```

## Configuration

All settings are managed via `deriv_quant_py/config.py` and environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DERIV_TOKEN` | Required | Your Deriv API Token |
| `TRADING_ACTIVE` | `False` | Master switch (can be toggled in UI) |
| `RISK_MULTIPLIER`| `2.1` | Martingale multiplier on loss |
| `DAILY_LOSS_LIMIT`| `50.0` | Max daily loss in USD |
| `EMA_PERIOD` | `200` | Trend filter period |
| `RSI_PERIOD` | `14` | Momentum period |

## File Structure

```text
deriv_quant_py/
├── main.py                 # Application Entry Point
├── config.py               # Configuration
├── shared_state.py         # Thread-safe Bridge
├── core/
│   ├── connection.py       # Async WebSocket Client
│   ├── engine.py           # Trading Logic Orchestrator
│   ├── executor.py         # Trade Execution & Monitoring
│   ├── scanner.py          # Market Scanner
│   └── backtester.py       # Grid Search Module
├── strategies/
│   └── triple_confluence.py # Signal Logic
├── dashboard/
│   ├── app.py              # Dash App Layout & Callbacks
│   └── components.py       # UI Components
└── utils/                  # Helpers
```
