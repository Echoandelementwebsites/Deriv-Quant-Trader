import openai
import os
import logging
import re
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback
import asyncio
from deriv_quant_py.core.market_physics import MarketPhysics
from deriv_quant_py.shared_state import state
from deriv_quant_py.core.scanner import MarketScanner

logger = logging.getLogger(__name__)


class AIResearcher:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_strategy(self, symbol, dna_profile, max_retries=3):
        # 1. SYSTEM PROMPT: Enforce Imports and Functional Logic
        system_prompt = """
        You are a Senior Quantitative Python Developer.
        Task: Write a complete Python module for a trading strategy.

        ### REQUIREMENTS
        1. **IMPORTS:** You MUST include standard imports at the top:
           `import pandas as pd`
           `import pandas_ta as ta`
           `import numpy as np`
        2. **FUNCTION SIGNATURE:** `def strategy_logic(df):`
        3. **OUTPUT:** Return a tuple of two boolean Series: `(call_signal, put_signal)`.
           - The 1st Series is the **CALL** (Long) signal.
           - The 2nd Series is the **PUT** (Short) signal.

        ### CRITICAL SYNTAX RULES
        1. **NO APPEND:** Do NOT use `append=True` in pandas_ta.
        2. **FUNCTIONAL STYLE:** Assign results to variables. 
           - Correct: `rsi = ta.rsi(df['close'], length=14)`
        3. **SAFE COLUMN ACCESS:** - For multi-column indicators (BBANDS, MACD), use `.iloc[:, index]`.
           - **NEVER** guess column names like 'BBL_20_2.0'.
           - Example: `upper_band = ta.bbands(df['close'], length=20).iloc[:, 2]`

        ### TEMPLATE
        ```python
        import pandas as pd
        import pandas_ta as ta
        import numpy as np

        def strategy_logic(df):
            # 1. Indicators
            close = df['close']
            rsi = ta.rsi(close, length=14)
            bb = ta.bbands(close, length=20, std=2.0)

            # Safe Extract
            lower = bb.iloc[:, 0]
            upper = bb.iloc[:, 2]

            # 2. Logic (Fill NaNs!)
            rsi = rsi.fillna(50)

            # 3. Signals (Booleans)
            call_signal = (close < lower) & (rsi < 30)
            put_signal = (close > upper) & (rsi > 70)

            # 4. Return
            return call_signal, put_signal
        ```
        """

        user_prompt = f"""
        ### DEEP MARKET VISION: {symbol}

        1. **The Natural Heartbeat (Dominant Cycle):** {dna_profile['dominant_cycle']} Bars
           - *CRITICAL INSTRUCTION:* Do NOT use hardcoded lengths like 14.
           - Use `length={dna_profile['dominant_cycle']}` for RSI, ADX, ATR, etc.
           - This aligns your strategy with the asset's actual frequency.

        2. **Raw Price Action (The Tape):**
           (Last 15 candles, normalized to basis points. 'Doji' means indecision.)
           ---------------------------------------------------
           {dna_profile['price_action_tape']}
           ---------------------------------------------------
           - *Task:* "Look" at this tape. Is momentum accelerating? Are candles getting smaller (compression) or larger (expansion)?

        3. **Market Context:**
           - Trend Strength (ADX): {dna_profile['trend_strength']:.1f}
           - Regime: {dna_profile['regime']} (Hurst={dna_profile['hurst']:.2f})
           - Noise: {dna_profile['noise_index']:.2f}

        ### STRATEGY GENERATION TASK
        Write the `strategy_logic(df)` function based on the **Raw Tape** and **Dominant Cycle**.

        1. **Dynamic Parameters:** Use variables for lengths.
           `cycle = {dna_profile['dominant_cycle']}`
           `rsi = ta.rsi(df['close'], length=cycle)`

        2. **Pattern Logic:** If you see "Doji" or compression in the tape, code a breakout logic (e.g., using BB Width). If you see strong "Green" expansion, code a trend follower.

        3. **Output:** Raw Python Code only.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            try:
                logger.info(f"AI Strategy Gen: {symbol} (Attempt {attempt + 1})")

                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    temperature=0.1
                )

                raw_code = response.choices[0].message.content
                clean_code = self._sanitize_code(raw_code)

                # Validate
                is_valid, error_msg = self._validate_code(clean_code)

                if is_valid:
                    self._save_to_file(symbol, clean_code)
                    logger.info(f"SUCCESS: Strategy for {symbol} created.")
                    return True
                else:
                    logger.warning(f"Validation Failed ({symbol}): {error_msg}")
                    # Feedback loop
                    messages.append({"role": "assistant", "content": raw_code})
                    messages.append({
                        "role": "user",
                        "content": f"FIX CODE ERROR: {error_msg}\n"
                                   f"Ensure you imported pandas_ta as ta. "
                                   f"Ensure you return (call, put) series."
                    })

            except Exception as e:
                logger.error(f"AI Gen Error: {e}")
                return False

        return False

    def _validate_code(self, code_str):
        try:
            # FIX: Use a SINGLE dictionary for scope so functions see imports
            scope = {}

            # 1. Pre-load imports (just in case AI missed them, though prompt enforces it)
            exec("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np", scope)

            # 2. Exec the AI code in the SAME scope
            exec(code_str, scope)

            if 'strategy_logic' not in scope:
                return False, "Function 'strategy_logic' not found."

            # 3. Test Run
            # Create dummy OHLC
            size = 100
            dates = pd.date_range(start='2024-01-01', periods=size, freq='1min')
            close = np.random.normal(100, 1, size).cumsum() + 1000
            df = pd.DataFrame({
                'open': close + np.random.normal(0, 0.1, size),
                'high': close + 1,
                'low': close - 1,
                'close': close,
                'volume': np.random.randint(100, 1000, size)
            }, index=dates)

            # RUN
            call, put = scope['strategy_logic'](df.copy())

            # 4. Checks
            if not isinstance(call, pd.Series) or not isinstance(put, pd.Series):
                return False, f"Return must be (pd.Series, pd.Series). Got {type(call)}, {type(put)}"

            if len(call) != len(df):
                return False, "Output length mismatch."

            return True, "Passed"

        except Exception as e:
            # Return simple error to AI, log full trace
            logger.debug(traceback.format_exc())
            return False, f"Runtime Error: {str(e)}"

    def _sanitize_code(self, code):
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.replace("```python", "").replace("```", "").strip()

    def _save_to_file(self, symbol, code):
        # Saves the valid code to a file
        path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(code)

async def run_research_session(client, backtester, researcher, resume=False):
    """
    Centralized logic for running the AI research loop.
    Used by both main.py (Dashboard) and scripts/run_ai_research.py (CLI).
    """
    logger.info(f"Starting AI Research Session (Resume={resume})...")

    # 1. Determine Target Assets
    # Try dynamic scan first
    scanner = MarketScanner(client)
    market_map = await scanner.scan_market()

    target_symbols = []
    if market_map:
        for cat, assets in market_map.items():
            for a in assets:
                target_symbols.append(a['symbol'])
        # De-duplicate
        target_symbols = list(set(target_symbols))

    # Fallback if scan empty
    if not target_symbols:
        logger.warning("Market scan returned empty. Using fallback list.")
        target_symbols = [
            "R_100", "R_75", "R_50", "R_25", "R_10",
            "1HZ100V", "1HZ75V", "1HZ50V", "1HZ25V", "1HZ10V",
            "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
            "CRASH_1000", "BOOM_1000", "CRASH_500", "BOOM_500"
        ]

    # 2. Resume Logic (Filter)
    if resume:
        # Resolve path relative to THIS file
        strategy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../strategies/generated'))
        if os.path.exists(strategy_dir):
            existing = set()
            for f in os.listdir(strategy_dir):
                if f.endswith("_ai.py"):
                    existing.add(f.replace("_ai.py", ""))

            original_count = len(target_symbols)
            target_symbols = [s for s in target_symbols if s not in existing]
            logger.info(f"RESUME: Filtered {original_count} -> {len(target_symbols)} assets.")
        else:
            logger.warning(f"Strategy dir {strategy_dir} not found. Processing all.")

    if not target_symbols:
        logger.info("No assets to process. Task Complete.")
        state.update_scan_progress(100, 100, "All Done", status="complete")
        return

    # 3. Execution Loop
    total = len(target_symbols)
    for i, symbol in enumerate(target_symbols):
        # Update Dashboard Progress
        state.update_scan_progress(total, i + 1, symbol, status="running")
        logger.info(f"AI Researching {symbol} ({i+1}/{total})...")

        try:
            # A. Fetch History
            df = await backtester.fetch_history_paginated(symbol, months=1)
            if df.empty or len(df) < 500:
                logger.warning(f"Insufficient data for {symbol}. Skipping.")
                continue

            # B. DNA Analysis (Deep Dive)
            # Use perform_deep_analysis instead of get_asset_dna to get the cycle and tape
            dna = MarketPhysics.perform_deep_analysis(df)

            # C. Generate Strategy
            success = researcher.generate_strategy(symbol, dna)
            if success:
                logger.info(f"SUCCESS: Generated strategy for {symbol}")

            # Rate Limit
            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error researching {symbol}: {e}")

    state.update_scan_progress(total, total, "Complete", status="complete")
    logger.info("AI Research Session Completed.")
