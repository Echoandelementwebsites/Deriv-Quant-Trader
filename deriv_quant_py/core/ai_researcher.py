import openai
import os
import logging
import re
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback
import asyncio
import json
from deriv_quant_py.core.market_physics import MarketPhysics
from deriv_quant_py.shared_state import state
from deriv_quant_py.core.scanner import MarketScanner
from deriv_quant_py.core.metrics import MetricsEngine
from deriv_quant_py.core.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)


class AIResearcher:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.validator = StrategyValidator()

    def generate_strategy(self, symbol, dna_profile, df, max_retries=5):
        # 1. SYSTEM PROMPT: JSON Schema Enforcement
        system_prompt = """
        You are a Senior Quantitative Researcher.
        Task: Output a VALID JSON configuration for a trading strategy.

        ### SCHEMA
        {
            "regime_type": "TREND" | "RANGE",
            "indicators": [
                {"kind": "rsi", "length": 14},
                {"kind": "bbands", "length": 20, "std": 2.0},
                {"kind": "ema", "length": 50},
                {"kind": "adx", "length": 14}
            ],
            "entry_logic": {
                "call": "(close > BBU_20_2.0) & (RSI_14 > 70) & (ADX_14 > 25)",
                "put": "(close < BBL_20_2.0) & (RSI_14 < 30) & (ADX_14 > 25)"
            }
        }

        ### CRITICAL RULES
        1. **OUTPUT JSON ONLY**. No markdown, no comments outside the JSON.
        2. **Column Names**:
           - RSI: `RSI_{length}` (e.g., RSI_14)
           - EMA: `EMA_{length}` (e.g., EMA_50)
           - BBands: `BBL_{len}_{std}`, `BBM...`, `BBU_{len}_{std}` (e.g., BBU_20_2.0)
           - ADX: `ADX_{length}` (e.g., ADX_14)
        3. **Logic**: Must be valid `pandas.eval` syntax. Use `&` for AND, `|` for OR.
        4. **Profitability**:
           - If `regime_type` is "TREND", you MUST filter for `ADX_14 > 25`.
           - If `regime_type` is "RANGE", you MUST filter for `ADX_14 < 25`.
        """

        user_prompt = f"""
        ### MARKET GENOME: {symbol}
        - Cycle Length: {dna_profile['dominant_cycle']}
        - Trend Strength: {dna_profile['trend_strength']:.1f}
        - Regime: {dna_profile['regime']}
        - Structure: {dna_profile['market_structure']['structure_tag']}

        Design a strategy that fits this regime.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use recent history for testing (approx 3.5 days)
        test_df = df.iloc[-5000:].copy() if len(df) > 5000 else df.copy()

        for attempt in range(max_retries):
            try:
                logger.info(f"AI Strategy Gen: {symbol} (Attempt {attempt + 1})")

                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    temperature=0.1
                )

                raw_content = response.choices[0].message.content
                json_config = self._extract_json(raw_content)

                if not json_config:
                    logger.warning(f"Failed to parse JSON for {symbol}")
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({"role": "user", "content": "Error: Invalid JSON. Output ONLY valid JSON."})
                    continue

                # 2. Validation Loop
                validation_result = self.validator.validate_strategy(test_df, json_config)

                if validation_result.passed:
                    logger.info(f"SUCCESS: {symbol} Passed Validation. EV={validation_result.metrics['ev']:.2f}")

                    # Convert to Python and Save
                    py_code = self._generate_python_code(json_config)
                    self._save_to_file(symbol, py_code)
                    return True
                else:
                    # Feedback
                    reason = validation_result.reason
                    logger.warning(f"Strategy Failed Validation ({symbol}): {reason}")

                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": f"Strategy FAILED. Reason: {reason}. \nAdjust parameters or logic to fix this failure."
                    })

            except Exception as e:
                logger.error(f"AI Gen Error: {e}")
                traceback.print_exc()

        return False

    def _extract_json(self, text):
        try:
            # Find JSON block
            if "```json" in text:
                pattern = r"```json(.*?)```"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return json.loads(match.group(1).strip())
            elif "```" in text:
                pattern = r"```(.*?)```"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return json.loads(match.group(1).strip())

            return json.loads(text.strip())
        except:
            return None

    def _generate_python_code(self, config):
        indicators = config.get('indicators', [])
        entry_logic = config.get('entry_logic', {})

        # We define a robust strategy file that rebuilds the exact same logic
        code = f"""import pandas as pd
import pandas_ta as ta
import numpy as np

def strategy_logic(df):
    df = df.copy()

    # 1. Indicators
    indicators = {json.dumps(indicators)}

    # Construct Strategy
    try:
        if indicators:
            strat = ta.Strategy(name="AI_Generated", ta=indicators)
            df.ta.strategy(strat)
    except Exception as e:
        # Fallback empty
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    # 2. Logic
    try:
        call_signal = df.eval({repr(entry_logic.get('call', 'False'))})
        put_signal = df.eval({repr(entry_logic.get('put', 'False'))})

        # Ensure Series
        if isinstance(call_signal, bool): call_signal = pd.Series(call_signal, index=df.index)
        if isinstance(put_signal, bool): put_signal = pd.Series(put_signal, index=df.index)

    except Exception as e:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    return call_signal.fillna(False), put_signal.fillna(False)
"""
        return code

    def _save_to_file(self, symbol, code):
        path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(code)

async def run_research_session(client, backtester, researcher, resume=False):
    """
    Centralized logic for running the AI research loop.
    """
    logger.info(f"Starting AI Research Session (Resume={resume})...")

    # 1. Determine Target Assets
    scanner = MarketScanner(client)
    market_map = await scanner.scan_market()

    target_symbols = []
    if market_map:
        for cat, assets in market_map.items():
            for a in assets:
                target_symbols.append(a['symbol'])
        target_symbols = list(set(target_symbols))

    # Fallback
    if not target_symbols:
        logger.warning("Market scan empty. Using fallback.")
        target_symbols = [
            "R_100", "R_75", "R_50", "R_25", "1HZ100V", "1HZ50V",
            "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "CRASH_1000", "BOOM_1000"
        ]

    # 2. Resume Logic
    if resume:
        strategy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../strategies/generated'))
        if os.path.exists(strategy_dir):
            existing = set()
            for f in os.listdir(strategy_dir):
                if f.endswith("_ai.py"):
                    existing.add(f.replace("_ai.py", ""))
            target_symbols = [s for s in target_symbols if s not in existing]

    if not target_symbols:
        logger.info("No assets to process.")
        state.update_scan_progress(100, 100, "All Done", status="complete")
        return

    # 3. Execution Loop
    total = len(target_symbols)
    for i, symbol in enumerate(target_symbols):
        state.update_scan_progress(total, i + 1, symbol, status="running")
        logger.info(f"AI Researching {symbol} ({i+1}/{total})...")

        try:
            df = await backtester.fetch_history_paginated(symbol, months=1)
            if df.empty or len(df) < 500:
                continue

            dna = MarketPhysics.perform_deep_analysis(df)
            researcher.generate_strategy(symbol, dna, df)

            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error researching {symbol}: {e}")

    state.update_scan_progress(total, total, "Complete", status="complete")
    logger.info("AI Research Session Completed.")
