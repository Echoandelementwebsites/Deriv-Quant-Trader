# CT Construction RFP Scraper & Dashboard (v2)

A specialized tool to scrape, filter, and aggregate "Request for Proposal" (RFP) data for construction projects from official Connecticut state sources.

## Features

*   **Targeted Scraping**: Retrieves opportunities from:
    *   [CTsource Bid Board](https://portal.ct.gov/das/ctsource/bidboard) (Official State Portal)
    *   [UConn Capital Projects](https://cpfp.procurement.uconn.edu/construction-current-opportunities-2020-2/)
*   **Strict Date Filtering**: Automatically discards RFPs that are due within **4 days** to ensure only actionable opportunities are presented.
    *   *Logic*: `Deadline >= (Current Date + 4 Days)`
*   **Data Normalization**: Standardizes diverse source data into a clean CSV format.
*   **Interactive Dashboard**: A user-friendly Streamlit interface to trigger scrapes, view progress, and download results.

## Prerequisites

*   Python 3.9+
*   Google Chrome / Chromium (managed by Playwright)

## Installation

1.  Navigate to the project directory:
    ```bash
    cd rfp_scraper
    ```

2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Install Playwright browsers (Chromium):
    ```bash
    playwright install chromium
    ```

## Usage

### Run the Dashboard (Recommended)

Start the Streamlit interface:

```bash
streamlit run app.py
```

This will open a local web server (typically `http://localhost:8501`) where you can:
1.  Select the region (default: Connecticut).
2.  Click **"Scrape RFPs"**.
3.  View the "Total Found" vs. "Qualified" metrics.
4.  Download the filtered CSV.

### Run the Scraper Standalone

You can also run the scraper logic directly from the command line for testing or automation:

```bash
python scraper.py
```
This will save a file named `ct_construction_rfps_[YYYY-MM-DD].csv` in the current directory.

## Output Schema

The generated CSV contains the following columns:
*   `clientName`: Agency issuing the bid (e.g., UConn, CT DAS).
*   `title`: Project title.
*   `slug`: URL-friendly identifier.
*   `description`: Brief summary.
*   `deadline`: Submission deadline (ISO 8601).
*   `budgetMin`: Estimated budget (0 if not specified).
*   `jobCity`: Inferred city location.
*   `portfolioLink`: Direct URL to the solicitation.
*   ...and more.
