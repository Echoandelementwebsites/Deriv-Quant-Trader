import streamlit as st
import pandas as pd
import os
import time
from scraper import CTScraper

st.set_page_config(page_title="CT Construction RFP Dashboard", layout="wide")

st.title("🏗️ CT Construction RFP Scraper (v2)")

# Sidebar
st.sidebar.header("Configuration")
region = st.sidebar.selectbox("Select State/Region", ["Connecticut"])
st.sidebar.info("Target Sources:\n1. CTSource Bid Board\n2. UConn Capital Projects")

# Main Area
st.markdown(
    """
    **Objective**: Fetch and filter construction RFPs.
    \n**Filter Logic**: `Deadline >= (Today + 4 Days)`
    """
)

if st.button("🚀 Scrape RFPs"):
    scraper = CTScraper()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(msg):
        status_text.text(msg)
        # Simple simulated progress for UX
        if "UConn" in msg:
            progress_bar.progress(30)
        elif "CTSource" in msg:
            progress_bar.progress(60)
        elif "Finalizing" in msg:
            progress_bar.progress(90)

    try:
        results, total_found = scraper.scrape(progress_callback=update_progress)
        progress_bar.progress(100)
        status_text.success("Scraping Complete!")

        if results:
            filename, df = scraper.save_csv()

            # Metrics
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Found", total_found)
            with col2:
                st.metric("Qualified (4+ Days)", len(df))

            # Display Data
            st.subheader("Filtered Opportunities")
            st.dataframe(df, use_container_width=True)

            # Download
            with open(filename, "rb") as f:
                st.download_button(
                    label="📥 Download CSV",
                    data=f,
                    file_name=filename,
                    mime="text/csv"
                )

            # Cleanup (Optional: remove file after read if we don't want to clutter,
            # but keeping it is good for audit. The user requested 'The output must be a CSV file...')
            # We will keep it.

        else:
            st.warning("No qualified RFPs found matching the criteria.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
