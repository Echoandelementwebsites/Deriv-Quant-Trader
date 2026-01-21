import openai
import os
import logging
import re
import pandas as pd
import numpy as np
import pandas_ta as ta # Required for validation check

logger = logging.getLogger(__name__)

class AIResearcher:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_strategy(self, symbol, dna_profile):
        # 1. THE STRICT PROMPT
        system_prompt = """
        You are a Quantitative Researcher. Write a Python function `strategy_logic(df)`.

        RULES:
        1. Input: `df` (pandas DataFrame with columns: open, high, low, close).
        2. Output: Tuple `(call_signal, put_signal)` where both are `pd.Series` of booleans.
        3. LIB: Use `pandas_ta` extension.

        CRITICAL SYNTAX REQUIREMENTS:
        - ALWAYS use `append=True` when calculating indicators.
          Correct: `df.ta.rsi(length=14, append=True)`
          Incorrect: `rsi = df.ta.rsi(...)`
        - Access columns using their generated names (e.g., 'RSI_14').
        - Handle NaN values using `.fillna(False)` on the final result.
        - Do NOT import libraries inside the function. Assume `pd`, `ta`, `np` are available.
        - Return strictly: `return call_signal, put_signal`
        """

        user_prompt = f"""
        Asset: {symbol}
        Physics Profile:
        - Regime: {dna_profile['regime']}
        - Noise: {dna_profile['noise_index']:.2f} (0=Clean, 1=Noisy)
        - Volatility: {dna_profile['volatility']:.5f}

        Task: Write the `strategy_logic` function.
        - If Regime is MEAN_REVERSION: Use RSI or Bollinger Bands.
        - If Regime is TRENDING: Use Supertrend or EMA Cross.
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner", # Uses R1 Chain-of-Thought
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5 # Lower temp for more deterministic code
            )

            raw_code = response.choices[0].message.content
            clean_code = self._sanitize_code(raw_code)

            # 2. THE VALIDATION LAYER
            is_valid, error_msg = self._validate_code(clean_code)

            if is_valid:
                self._save_to_file(symbol, clean_code)
                logger.info(f"Generated & Validated strategy for {symbol}")
                return True
            else:
                logger.error(f"Strategy Validation Failed for {symbol}: {error_msg}")
                # Optional: Retry logic could go here
                return False

        except Exception as e:
            logger.error(f"AI Research Error {symbol}: {e}")
            return False

    def _validate_code(self, code_str):
        """
        Sandboxes the code to ensure it runs without crashing and returns the correct format.
        """
        try:
            # 1. Create a safe local execution scope
            # We use a single dictionary for globals/locals to ensure functions
            # defined in the code can access the imported libraries.
            scope = {}

            # Pre-import standard libs into the scope
            exec("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np", scope)

            # 2. Compile the AI code
            exec(code_str, scope)

            if 'strategy_logic' not in scope:
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
            # We must pass the df into the function found in the scope
            call, put = scope['strategy_logic'](df.copy())

            # 5. Check Outputs
            if not isinstance(call, pd.Series) or not isinstance(put, pd.Series):
                return False, f"Return types invalid. Got {type(call)}, {type(put)}"

            if len(call) != len(df):
                return False, "Output Series length mismatch."

            return True, "Passed"

        except Exception as e:
            return False, f"Runtime Error: {str(e)}"

    def _sanitize_code(self, code):
        # Remove Markdown wrappers
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.replace("```", "").strip()

    def _save_to_file(self, symbol, code):
        # Enforce the naming convention
        path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np\n\n")
            f.write(code)
