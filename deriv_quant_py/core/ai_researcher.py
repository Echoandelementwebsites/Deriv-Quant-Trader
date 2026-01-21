import openai
import os
import logging
import re
import pandas as pd
import numpy as np
import pandas_ta as ta  # Required for Validation Scope

logger = logging.getLogger(__name__)


class AIResearcher:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_strategy(self, symbol, dna_profile, max_retries=3):
        # 1. THE ENHANCED SYSTEM PROMPT (With Docs & One-Shot Example)
        system_prompt = """
        You are a Senior Quantitative Developer. Write a Python function `strategy_logic(df)`.

        ### REFERENCE DOCUMENTATION
        - Pandas TA Lib: https://github.com/twopirllc/pandas-ta
        - Pandas Docs: https://pandas.pydata.org/docs/

        ### LIBRARY CONTEXT
        - Library: `pandas_ta` (standard naming convention).
        - Input: `df` (columns: open, high, low, close).
        - Output: Tuple `(call_signal, put_signal)` where both are `pd.Series` of booleans.

        ### CRITICAL SYNTAX RULES
        1. **APPEND IS MANDATORY:** Always use `append=True` for indicators.
           - Correct: `df.ta.bbands(length=20, std=2, append=True)`
           - Wrong: `bb = df.ta.bbands(...)`
        2. **COLUMN NAMES:** Pandas TA generates specific names. You must use them.
           - BBANDS(20, 2) -> `BBL_20_2.0`, `BBM_20_2.0`, `BBU_20_2.0`
           - RSI(14) -> `RSI_14`
           - SUPERTREND(7, 3) -> `SUPERT_7_3.0`
        3. **SAFETY:** Use `.fillna(False)` on final signals. Avoid `inf`.

        ### GOLDEN EXAMPLE (FOLLOW THIS STRUCTURE)
        def strategy_logic(df):
            # 1. Calculate Indicators (append=True)
            df.ta.ema(length=50, append=True)  # Creates 'EMA_50'
            df.ta.rsi(length=14, append=True)  # Creates 'RSI_14'

            # 2. Define Logic
            # Trend Condition: Close > EMA
            trend_up = df['close'] > df['EMA_50']

            # Entry Condition: RSI < 30 (Oversold)
            entry_long = df['RSI_14'] < 30

            # 3. Generate Signals
            call_signal = trend_up & entry_long
            put_signal = (~trend_up) & (df['RSI_14'] > 70)

            # 4. Return safely
            return call_signal.fillna(False), put_signal.fillna(False)
        """

        # 2. DETAILED CONTEXT PROMPT
        user_prompt = f"""
        ### ASSET DNA REPORT: {symbol}
        1. **Regime:** {dna_profile['regime']} (Hurst={dna_profile['hurst']:.2f})
           - If TRENDING (Hurst > 0.5): Prioritize breakouts (Donchian, Supertrend).
           - If REVERSION (Hurst < 0.5): Prioritize oscillators (RSI, Stoch, BB).

        2. **Noise Level:** {dna_profile['noise_index']:.2f} (0.0=Smooth, 1.0=Chaotic)
           - Current Level: {"HIGH" if dna_profile['noise_index'] > 0.6 else "LOW"}
           - Instruction: If High Noise, use longer lookback periods or smoothing (EMA/SMA) to filter fakeouts.

        3. **Volatility:** {dna_profile['volatility']:.5f}
           - Instruction: Use this to tune thresholds. Higher volatility requires wider stop-losses or bands.

        ### TASK
        Write the `strategy_logic` function for this asset. 
        Focus on exploiting the identified Regime while mitigating the Noise.
        OUTPUT ONLY RAW PYTHON CODE. NO MARKDOWN.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            try:
                logger.info(f"AI Generation Attempt {attempt + 1}/{max_retries} for {symbol}...")

                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",  # DeepSeek-V3.2 Thinking Mode
                    messages=messages,
                    temperature=0.0  # Strict determinism for coding
                )

                raw_code = response.choices[0].message.content
                clean_code = self._sanitize_code(raw_code)

                # 3. VALIDATION LAYER
                is_valid, error_msg = self._validate_code(clean_code)

                if is_valid:
                    self._save_to_file(symbol, clean_code)
                    logger.info(f"Generated & Validated strategy for {symbol}")
                    return True
                else:
                    logger.warning(f"Attempt {attempt + 1} Failed: {error_msg}")
                    # Feed specific error back to AI
                    messages.append({"role": "assistant", "content": raw_code})
                    messages.append({
                        "role": "user",
                        "content": f"VALIDATION ERROR: {error_msg}\n"
                                   f"Review the 'CRITICAL SYNTAX RULES'. Did you use the correct column name? "
                                   f"Did you use `append=True`? Fix the code."
                    })

            except Exception as e:
                logger.error(f"AI Research Error {symbol}: {e}")
                return False

        logger.error(f"Failed to generate valid strategy for {symbol} after {max_retries} attempts.")
        return False

    def _validate_code(self, code_str):
        """
        Sandboxes the code to ensure it runs without crashing and returns the correct format.
        """
        try:
            # 1. Create a safe local execution scope
            local_scope = {}
            exec("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np", {}, local_scope)

            # 2. Compile the AI code
            exec(code_str, {}, local_scope)

            if 'strategy_logic' not in local_scope:
                return False, "Function 'strategy_logic' not defined."

            # 3. Create Dummy Data (Random Walk)
            np.random.seed(42)
            prices = np.random.normal(100, 1, 100).cumsum()
            df = pd.DataFrame({
                'open': prices,
                'high': prices + 1,
                'low': prices - 1,
                'close': prices
            })

            # 4. Dry Run
            call, put = local_scope['strategy_logic'](df.copy())

            # 5. Check Outputs
            if not isinstance(call, pd.Series) or not isinstance(put, pd.Series):
                return False, f"Return types invalid. Got {type(call)}, {type(put)}"

            if len(call) != len(df):
                return False, "Output Series length mismatch."

            return True, "Passed"

        except Exception as e:
            return False, f"Runtime Error: {str(e)}"

    def _sanitize_code(self, code):
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.replace("```", "").strip()

    def _save_to_file(self, symbol, code):
        # Enforce the naming convention: frxEURUSD_ai.py
        path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np\n\n")
            f.write(code)