import time
from playwright.sync_api import sync_playwright, expect

def verify_backtest():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Dashboard
        print("Navigating to Dashboard...")
        page.goto("http://127.0.0.1:8050/backtest")

        # Wait for button
        print("Waiting for 'Run Grid Search' button...")
        btn = page.locator("#bt-run-btn")
        btn.wait_for(state="visible", timeout=10000)

        # Click button
        print("Clicking button...")
        btn.click()

        # Wait for chart title to change
        print("Waiting for chart update...")
        # The chart title should eventually become "Backtest Grid Search Results"
        # OR "Request Sent..." initially.
        # We can check the chart title text

        chart_title_locator = page.locator("#bt-chart text='Request Sent... Waiting for result'")
        try:
            chart_title_locator.wait_for(state="visible", timeout=5000)
            print("Verified: 'Request Sent...' appeared.")
        except:
            print("Warning: 'Request Sent...' did not appear or was too fast.")

        # Wait for actual results
        # The title should change to "Backtest Grid Search Results (Win Rate %)"
        # This might take a while depending on backend speed.
        # Since we are using R_100 (simulated/real), let's wait up to 15s.

        result_title_locator = page.locator("text=Backtest Grid Search Results")
        try:
            result_title_locator.wait_for(state="visible", timeout=15000)
            print("Verified: Results appeared!")
            page.screenshot(path="verification_backtest_success.png")
        except:
            print("Failed: Results did not appear within timeout.")
            page.screenshot(path="verification_backtest_failed.png")

        browser.close()

if __name__ == "__main__":
    verify_backtest()
