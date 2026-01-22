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
        ### ASSET DNA: {symbol}
        - Regime: {dna_profile['regime']} (Hurst={dna_profile['hurst']:.2f})
        - Noise: {dna_profile['noise_index']:.2f}
        - Volatility: {dna_profile['volatility']:.5f}

        Write the strategy module. OUTPUT ONLY RAW PYTHON CODE.
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