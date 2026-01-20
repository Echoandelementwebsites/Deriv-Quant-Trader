import openai
import os
import logging
import re
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AIResearcher:
    def __init__(self, api_key):
        # STRICT: Use DeepSeek Base URL
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_strategy(self, symbol, dna_profile):
        system_prompt = """
        You are a Quant Researcher specializing in Binary Options.
        Task: Write a Python function `strategy_logic(df)` using `pandas_ta`.

        CRITICAL OUTPUT RULES:
        1. Output ONLY valid Python code. No Markdown. No Explanations.
        2. Function signature: `def strategy_logic(df): -> tuple(pd.Series, pd.Series)` returning (call_signal, put_signal).
        3. Do not assume 'df' has a datetime index. Use `iloc`.
        """

        user_prompt = f"""
        Asset: {symbol}
        DeepSeek DNA Profile:
        - Hurst Exponent: {dna_profile['hurst']:.3f} (0.5=Random, >0.5=Trend, <0.5=Reversion)
        - Noise Index: {dna_profile['noise_index']:.3f} (0=Clean, 1=Noisy)
        - Volatility: {dna_profile['volatility']:.5f}

        Reasoning Task:
        1. Analyze the Hurst vs Noise level.
        2. If Hurst > 0.6 and Noise < 0.4: Write a Trend Following strategy (e.g., Supertrend + ADX).
        3. If Hurst < 0.4 and Noise > 0.6: Write a Mean Reversion strategy (e.g., BB Reversal + RSI).
        4. If parameters are ambiguous, use the Volatility to adjust lookback periods (Higher Vol = Shorter Lookback).

        Generate the code now.
        """

        try:
            # STRICT: Use 'deepseek-reasoner' for Chain-of-Thought capabilities
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6 # Slightly creative to find novel logic
            )

            # Extract content (DeepSeek R1 puts reasoning in 'reasoning_content' sometimes, but code is in 'content')
            code = response.choices[0].message.content

            # Clean Markdown if DeepSeek adds it despite instructions
            code = self._sanitize_code(code)

            self._save_to_file(symbol, code)
            return True
        except Exception as e:
            logger.error(f"DeepSeek R1 failed for {symbol}: {e}")
            return False

    def _sanitize_code(self, code):
        # Remove ```python and ``` tags
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.replace("```", "").strip()

    def _save_to_file(self, symbol, code):
        directory = "deriv_quant_py/strategies/generated"
        os.makedirs(directory, exist_ok=True)
        path = f"{directory}/{symbol}_ai.py"
        with open(path, "w") as f:
            f.write("import pandas as pd\nimport pandas_ta as ta\nimport numpy as np\n\n")
            f.write(code)
