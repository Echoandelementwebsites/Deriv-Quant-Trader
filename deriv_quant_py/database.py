from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    signal_type = Column(String)  # CALL/PUT
    contract_id = Column(String, nullable=True)
    stake = Column(Float)
    status = Column(String)  # OPEN, WON, LOST
    profit = Column(Float, default=0.0)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    details = Column(Text, nullable=True) # JSON blob for extra info

class SignalLog(Base):
    __tablename__ = 'signal_logs'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    signal = Column(String) # CALL/PUT
    reason = Column(String)
    price = Column(Float)
    indicators = Column(Text) # JSON blob {rsi: 50, ema: 100}

class StrategyParams(Base):
    __tablename__ = 'strategy_params'
    symbol = Column(String, primary_key=True)
    rsi_period = Column(Integer)
    ema_period = Column(Integer)
    optimal_duration = Column(Integer, default=3)
    rsi_vol_window = Column(Integer, default=100)
    win_rate = Column(Float)
    signal_count = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)
    details = Column(Text, nullable=True) # JSON blob for meta info
    strategy_type = Column(String, nullable=True)  # REVERSAL, TREND, BREAKOUT
    config_json = Column(Text, nullable=True)  # Strategy specific params

    # Advanced Metrics
    expectancy = Column(Float, default=0.0)
    kelly = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

def init_db(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    # Auto-migration for schema updates (SQLite)
    # Check if new columns exist, if not, add them.
    from sqlalchemy import inspect, text
    inspector = inspect(engine)

    if 'strategy_params' in inspector.get_table_names():
        columns = [c['name'] for c in inspector.get_columns('strategy_params')]

        with engine.connect() as conn:
            if 'optimal_duration' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN optimal_duration INTEGER DEFAULT 3"))

            if 'rsi_vol_window' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN rsi_vol_window INTEGER DEFAULT 100"))

            if 'details' not in columns:
                 conn.execute(text("ALTER TABLE strategy_params ADD COLUMN details TEXT"))

            if 'strategy_type' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN strategy_type VARCHAR"))

            if 'config_json' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN config_json TEXT"))

            if 'expectancy' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN expectancy FLOAT DEFAULT 0.0"))

            if 'kelly' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN kelly FLOAT DEFAULT 0.0"))

            if 'max_drawdown' not in columns:
                conn.execute(text("ALTER TABLE strategy_params ADD COLUMN max_drawdown FLOAT DEFAULT 0.0"))

    return sessionmaker(bind=engine)
