import openai
import os
import logging
import re
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback

logger = logging.getLogger(__name__)


class AIResearcher:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_strategy(self, symbol, dna_profile, max_retries=3):
        # ---------------------------------------------------------
        # 1. ARCHITECTURE PROMPT (The "Team" Logic)
        # ---------------------------------------------------------
        system_prompt = """
        You are a Senior Quantitative Architect (Python). 
        Your task is to write a robust, error-free strategy function `strategy_logic(df)` for a high-frequency trading bot.

        ### CRITICAL ARCHITECTURE RULES (DO NOT IGNORE)
        1. **NO APPEND:** DO NOT use `append=True`. DO NOT modify the input `df`.
        2. **FUNCTIONAL STYLE:** Assign indicator results to variables.
           - WRONG: `df.ta.rsi(append=True)` then accessing `df['RSI_14']`
           - CORRECT: `rsi_series = ta.rsi(df['close'], length=14)`
        3. **SAFE COLUMN ACCESS:** `pandas_ta` returns DataFrames for multi-line indicators (like BBands). 
           - **NEVER guess column names** like 'BBL_20_2.0'. They are dynamic and cause crashes.
           - **USE .iloc[]:** Access columns by index.
             - BBANDS: [0]=Lower, [1]=Mid, [2]=Upper
             - MACD:   [0]=MACD, [1]=Histogram, [2]=Signal
             - STOCH:  [0]=K, [1]=D

        ### DATA STRUCTURE
        - Input: `df` (pandas DataFrame) with columns: 'open', 'high', 'low', 'close'.
        - Output: Tuple `(call_signal, put_signal)` -> both `pd.Series` of booleans.

        ### TEMPLATE (FOLLOW STYLES EXACTLY)
        def strategy_logic(df):
            # 1. Extract Series for readability
            close = df['close']
            high = df['high']
            low = df['low']

            # 2. Indicators (Functional Assignment)
            # RSI (Returns Series)
            rsi = ta.rsi(close, length=14)

            # BBANDS (Returns DataFrame: Lower, Mid, Upper)
            bb_df = ta.bbands(close, length=20, std=2.0)
            # SAFE EXTRACT: Use iloc to avoid Name Errors
            bb_lower = bb_df.iloc[:, 0] 
            bb_upper = bb_df.iloc[:, 2]

            # EMA (Returns Series)
            ema_trend = ta.ema(close, length=200)

            # 3. Logic & Cleaning
            # Important: Fill NaNs to prevent false signals at start of data
            rsi = rsi.fillna(50)
            ema_trend = ema_trend.fillna(close) 

            # 4. Conditions
            # Example: Reversion Strategy
            long_cond = (close < bb_lower) & (rsi < 30) & (close > ema_trend)
            short_cond = (close > bb_upper) & (rsi > 70) & (close < ema_trend)

            # 5. Return (Series, Series)
            return long_cond, short_cond
        """

        # ---------------------------------------------------------
        # 2. STRATEGY CONTEXT (The "Quant" Logic)
        # ---------------------------------------------------------
        user_prompt = f"""
        ### ASSET REPORT: {symbol}
        - **Market Regime:** {dna_profile['regime']} (Hurst={dna_profile['hurst']:.2f})
        - **Noise Level:** {dna_profile['noise_index']:.2f} (0=Clean, 1=Noisy)
        - **Volatility:** {dna_profile['volatility']:.5f}

        ### STRATEGY OBJECTIVES
        1. **Regime Alignment:**
           - If TRENDING (Hurst > 0.5): Use Trend Following (EMA, MACD, Supertrend).
           - If MEAN_REVERSION (Hurst < 0.5): Use Oscillators (RSI, Stoch, BB).
        2. **Noise Handling:**
           - The Noise Index is {dna_profile['noise_index']:.2f}. 
           - {"Use longer lookbacks/smoothing (EMA) to filter noise." if dna_profile['noise_index'] > 0.6 else "Standard settings are fine."}

        ### OUTPUT REQUIREMENT
        - Output ONLY raw Python code wrapped in ```python ... ``` blocks.
        - No markdown text outside the code block.
        - The function MUST be named `strategy_logic`.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ---------------------------------------------------------
        # 3. GENERATION & REPAIR LOOP
        # ---------------------------------------------------------
        for attempt in range(max_retries):
            try:
                logger.info(f"AI Strategy Gen: {symbol} (Attempt {attempt + 1})")

                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",  # Leverages DeepSeek's CoT
                    messages=messages,
                    temperature=0.1  # Low temp for code precision
                )

                raw_code = response.choices[0].message.content
                clean_code = self._sanitize_code(raw_code)

                # Validate
                is_valid, error_msg = self._validate_code(clean_code)

                if is_valid:
                    self._save_to_file(symbol, clean_code)
                    logger.info(f"SUCCESS: Strategy for {symbol} compiled and verified.")
                    return True
                else:
                    logger.warning(f"VALIDATION FAILED ({symbol}): {error_msg}")
                    # FEEDBACK LOOP: Show the AI exactly what went wrong
                    messages.append({"role": "assistant", "content": raw_code})
                    messages.append({
                        "role": "user",
                        "content": f"CRITICAL ERROR in your code: {error_msg}\n"
                                   f"REMINDER: Do NOT access columns by name (like ['BBL...']). "
                                   f"Use .iloc[:, index] to access indicator outputs safely. Rewrite the code."
                    })

            except Exception as e:
                logger.error(f"DeepSeek API Error: {e}")
                return False

        logger.error(f"FAILED to generate strategy for {symbol} after retries.")
        return False

    def _validate_code(self, code_str):
        """
        Executes the code in a sandbox with realistic random data to catch runtime errors.
        """
        try:
            local_scope = {}
            # Pre-import standard libs
            exec("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np", {}, local_scope)
            exec(code_str, {}, local_scope)

            if 'strategy_logic' not in local_scope:
                return False, "Function 'strategy_logic' not found."

            # Generate realistic test data (OHLC)
            size = 200
            np.random.seed(42)
            close = np.random.normal(100, 2, size).cumsum() + 1000
            high = close + np.random.random(size)
            low = close - np.random.random(size)
            open_ = close + np.random.random(size) - 0.5

            df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close})

            # RUN THE STRATEGY
            call, put = local_scope['strategy_logic'](df.copy())

            # Type Checks
            if not isinstance(call, pd.Series) or not isinstance(put, pd.Series):
                return False, f"Must return (pd.Series, pd.Series). Got ({type(call)}, {type(put)})"

            # Length Checks
            if len(call) != size:
                return False, f"Output length mismatch. Input={size}, Output={len(call)}"

            # Value Checks (Ensure no NaNs or Infs leaked)
            if call.isnull().any() or put.isnull().any():
                return False, "Output contains NaNs. Did you use .fillna(False)?"

            return True, "Passed"

        except Exception as e:
            # Capture the full traceback for the logs, return the simplified error for the AI
            err_details = traceback.format_exc()
            logger.debug(err_details)
            return False, f"Runtime Error: {str(e)}"

    def _sanitize_code(self, code):
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: remove markdown backticks if pattern fails
        return code.replace("```python", "").replace("```", "").strip()

    def _save_to_file(self, symbol, code):
        path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np\n\n")
            f.write(code)