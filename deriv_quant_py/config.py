import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DERIV_TOKEN = os.getenv("DERIV_TOKEN")
    DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
    WEBSOCKET_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

    # Trading
    TRADING_ACTIVE = os.getenv("TRADING_ACTIVE", "False").lower() == "true"
    RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "50.0"))

    # Strategy
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
    EMA_PERIOD = int(os.getenv("EMA_PERIOD", 200))
    RSI_OB = int(os.getenv("RSI_OB", 70))
    RSI_OS = int(os.getenv("RSI_OS", 30))
    ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", 25))

    DB_PATH = "sqlite:///deriv_quant.db"
