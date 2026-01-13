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
    win_rate = Column(Float)
    signal_count = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)
    details = Column(Text, nullable=True) # JSON blob for meta info

def init_db(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
