import re
import time
import datetime
from datetime import timedelta
from dateutil import parser
import pandas as pd
from playwright.sync_api import sync_playwright

# List of Connecticut towns/cities for location inference
CT_CITIES = [
    "Andover", "Ansonia", "Ashford", "Avon", "Barkhamsted", "Beacon Falls", "Berlin", "Bethany", "Bethel", "Bethlehem",
    "Bloomfield", "Bolton", "Bozrah", "Branford", "Bridgeport", "Bridgewater", "Bristol", "Brookfield", "Brooklyn",
    "Burlington", "Canaan", "Canterbury", "Canton", "Chaplin", "Cheshire", "Chester", "Clinton", "Colchester",
    "Colebrook", "Columbia", "Cornwall", "Coventry", "Cromwell", "Danbury", "Darien", "Deep River", "Derby", "Durham",
    "East Granby", "East Haddam", "East Hampton", "East Hartford", "East Haven", "East Lyme", "East Windsor", "Eastford",
    "Easton", "Ellington", "Enfield", "Essex", "Fairfield", "Farmington", "Franklin", "Glastonbury", "Goshen", "Granby",
    "Greenwich", "Griswold", "Groton", "Guilford", "Haddam", "Hamden", "Hampton", "Hartford", "Hartland", "Harwinton",
    "Hebron", "Kent", "Killingly", "Killingworth", "Lebanon", "Ledyard", "Lisbon", "Litchfield", "Lyme", "Madison",
    "Manchester", "Mansfield", "Marlborough", "Meriden", "Middlebury", "Middlefield", "Middletown", "Milford", "Monroe",
    "Montville", "Morris", "Naugatuck", "New Britain", "New Canaan", "New Fairfield", "New Hartford", "New Haven",
    "New London", "New Milford", "Newington", "Newtown", "Norfolk", "North Branford", "North Canaan", "North Haven",
    "North Stonington", "Norwalk", "Norwich", "Old Lyme", "Old Saybrook", "Orange", "Oxford", "Plainfield", "Plainville",
    "Plymouth", "Pomfret", "Portland", "Preston", "Prospect", "Putnam", "Redding", "Ridgefield", "Rocky Hill", "Roxbury",
    "Salem", "Salisbury", "Scotland", "Seymour", "Sharon", "Shelton", "Sherman", "Simsbury", "Somers", "South Windsor",
    "Southbury", "Southington", "Sprague", "Stafford", "Stamford", "Sterling", "Stonington", "Stratford", "Suffield",
    "Thomaston", "Thompson", "Tolland", "Torrington", "Trumbull", "Union", "Vernon", "Voluntown", "Wallingford", "Warren",
    "Washington", "Waterbury", "Waterford", "Watertown", "West Hartford", "West Haven", "Westbrook", "Weston", "Westport",
    "Wethersfield", "Willington", "Wilton", "Winchester", "Windham", "Windsor", "Windsor Locks", "Wolcott", "Woodbridge",
    "Woodbury", "Woodstock"
]

class RFPScraper:
    def __init__(self):
        self.results = []
        self.total_found = 0
        self.now = datetime.datetime.now()

    def parse_date(self, date_str):
        """Robust date parser."""
        if not date_str:
            return None
        try:
            # Handle cases like "1/13/2026 2:00 PM"
            dt = parser.parse(date_str, fuzzy=True)
            return dt
        except Exception:
            # Log warning or simple print
            print(f"Warning: Could not parse date '{date_str}'")
            return None

    def infer_city(self, title, description):
        """Infers city from title or description."""
        text = (title + " " + description).lower()
        for city in CT_CITIES:
            if city.lower() in text:
                return city
        return "Connecticut"  # Default generic

    def create_slug(self, title):
        """Generate a URL-friendly slug."""
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        return slug

    def is_qualified(self, deadline):
        """
        Logic: deadline >= (current_date + 4 days)
        Example: Today Jan 1.
        Deadline Jan 4 (3 days away) -> False.
        Deadline Jan 5 (4 days away) -> True.
        """
        if not deadline:
            return False

        # Compare dates only (ignore time for the "days" calculation mostly,
        # but the prompt says 'today, tomorrow, or fewer than 4 days from now')
        # Let's interpret strict delta: (deadline - now).days >= 4
        # Wait, if deadline is Jan 5 2pm and today is Jan 1 2pm, delta is 4 days. Included.
        # If deadline is Jan 4 2pm and today is Jan 1 2pm, delta is 3 days. Excluded.
        # This matches the user example.

        delta = deadline - self.now
        # We need to account for the fact that 'days' property is just the integer part.
        # If delta is 3 days and 23 hours, .days is 3.
        # The prompt says: "If today is Jan 1st, an RFP due on Jan 4th is excluded."
        # Jan 4 - Jan 1 = 3 days. Excluded.
        # Jan 5 - Jan 1 = 4 days. Included.

        # We should probably compare date objects to be safe and ignore time of day if the user implies calendar days.
        days_diff = (deadline.date() - self.now.date()).days
        return days_diff >= 4

    def save_csv(self, filename=None):
        if not filename:
            date_str = self.now.strftime("%Y-%m-%d")
            filename = f"ct_construction_rfps_{date_str}.csv"

        df = pd.DataFrame(self.results)

        # Ensure schema
        required_columns = [
            "clientName", "title", "slug", "description", "walkthroughDate",
            "rfiDate", "deadline", "budgetMin", "jobStreet", "jobCity",
            "jobState", "jobZip", "portfolioLink", "status"
        ]

        # Fill missing cols
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        # Reorder
        df = df[required_columns]

        df.to_csv(filename, index=False)
        return filename, df

class CTScraper(RFPScraper):
    def scrape(self, progress_callback=None):
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # --- 1. UConn Scraping ---
            if progress_callback: progress_callback("Scraping UConn...")
            try:
                self.scrape_uconn(page)
            except Exception as e:
                print(f"Error scraping UConn: {e}")

            # --- 2. CTSource Scraping ---
            if progress_callback: progress_callback("Scraping CTSource...")
            try:
                self.scrape_ct_source(page)
            except Exception as e:
                print(f"Error scraping CTSource: {e}")

            browser.close()

        if progress_callback: progress_callback("Filtering and finalizing...")
        return self.results, self.total_found

    def scrape_uconn(self, page):
        url = "https://cpfp.procurement.uconn.edu/construction-current-opportunities-2020-2/"
        page.goto(url)
        page.wait_for_selector("body")

        # The text dump shows items often have "Close MM/DD/YYYY ..." format
        # We can look for the structure. Based on the text dump, it looks like a list of text blocks.
        # A robust way is to find elements containing "Close" and parse around them.
        # Or better, look for the 'heading' of the project.

        # Let's grab all text content and split by some delimiter if structure is loose,
        # but let's try to find a repeated element.
        # Usually, these are <div> or <article> elements.

        # Heuristic: Find elements that have "Number" and "Close" text.
        # locator.all() might be too generic.
        # Let's try to get the full text and regex parse it if the DOM is messy,
        # but Playwright is better at DOM.

        # Let's guess the container class based on standard Wordpress/site themes.
        # or just iterate over all paragraph-like containers.

        # Alternative: Search for "Close" labels.
        close_labels = page.get_by_text("Close", exact=False).all()

        # We need to filter only the ones that look like dates.
        # The dump shows: "Close 2/03/2026 2:00 PM"
        # So the text "Close" might be followed by the date in the same element or next.

        # Let's assume standard scraping didn't work and use a text-block approach from the dump style
        # since I can't inspect the DOM directly.
        # However, I can ask Playwright to give me the innerText of the main container.

        # Attempt to find the main content area
        content = page.locator("#content, main, .entry-content").first
        if not content.count():
            content = page.locator("body")

        text = content.inner_text()

        # Regex to find projects
        # Pattern: Title \n Number ... Open ... Close ...
        # This is risky.

        # Better: iterate through elements that might be rows.
        # Let's try to find headers (h3, h4) which are likely titles.
        titles = page.locator("h3, h4").all()

        for title_el in titles:
            title_text = title_el.inner_text().strip()
            if not title_text or "Search" in title_text or "Navigation" in title_text:
                continue

            # The details are likely in the sibling or parent text.
            # Let's get the text of the parent or the next sibling.
            # Assuming the title is the start of the block.

            # Let's get the full text of the section following the title?
            # Or assume the title is INSIDE the block.
            # Let's get the parent of the title.
            card = title_el.locator("..") # Parent
            card_text = card.inner_text()

            # Check if this card has "Close" and "Number"
            if "Close" not in card_text:
                continue

            # Extract Date
            # "Close 2/03/2026 2:00 PM"
            match = re.search(r"Close\s+(\d{1,2}/\d{1,2}/\d{4}.*?)(?:\n|$|\r)", card_text)
            if not match:
                continue

            date_str = match.group(1).strip()
            deadline = self.parse_date(date_str)

            self.total_found += 1
            if not self.is_qualified(deadline):
                continue

            # Extract other fields
            # Number
            number_match = re.search(r"Number\s+([^\n]+)", card_text)
            slug_base = number_match.group(1).strip() if number_match else title_text
            slug = self.create_slug(slug_base + "-" + title_text)

            # Walkthrough / Pre-Bid
            walkthrough = None
            if "Pre-Bid" in card_text:
                # Try to find a date after "Pre-Bid"
                wb_match = re.search(r"Pre-Bid.*?\n\s*(\d{1,2}/\d{1,2}/\d{4}.*?)(?:\n|$)", card_text, re.DOTALL)
                if wb_match:
                    walkthrough = self.parse_date(wb_match.group(1).strip())

            # RFI
            rfi = None
            if "RFI" in card_text:
                rfi_match = re.search(r"RFI\s+(\d{1,2}/\d{1,2}/\d{4}.*?)(?:\n|$)", card_text)
                if rfi_match:
                    rfi = self.parse_date(rfi_match.group(1).strip())

            # Link
            # Try to find a link in the card
            link_el = card.locator("a").first
            link = link_el.get_attribute("href") if link_el.count() else url

            # City
            city = self.infer_city(title_text, card_text)

            row = {
                "clientName": "UConn",
                "title": title_text,
                "slug": slug,
                "description": card_text.replace("\n", " ").strip()[:500], # Truncate for sanity
                "walkthroughDate": walkthrough.isoformat() if walkthrough else "",
                "rfiDate": rfi.isoformat() if rfi else "",
                "deadline": deadline.isoformat(),
                "budgetMin": 0, # Default per spec
                "jobStreet": "",
                "jobCity": city,
                "jobState": "CT",
                "jobZip": "",
                "portfolioLink": link,
                "status": "open"
            }
            self.results.append(row)


    def scrape_ct_source(self, page):
        url = "https://portal.ct.gov/das/ctsource/bidboard"
        page.goto(url)

        # Wait for load
        page.wait_for_timeout(5000)

        # Try to search for "Construction"
        # Look for search box
        try:
            search_box = page.get_by_placeholder("Search", exact=False).first
            if not search_box.count():
                search_box = page.locator("input[type='search']").first

            if search_box.count():
                search_box.fill("Construction")
                search_box.press("Enter")
                page.wait_for_timeout(3000) # Wait for results
        except Exception as e:
            print(f"Could not search on CTSource: {e}")

        # Parse table rows
        # The site likely uses a <table>
        rows = page.locator("table tr").all()

        # If no table, fallback to generic list items or cards
        if len(rows) < 2:
            # Maybe div based?
            # Let's try to find items by date pattern
            pass

        # Iterate rows (skip header)
        for row in rows[1:]:
            text = row.inner_text()

            # We need to find the Deadline/End Date column.
            # Usually strict tables have column indices.
            # Let's grab all cells
            cells = row.locator("td").all()
            if not cells: continue

            # Heuristic: Find the date cell.
            # Usually End Date is one of the last columns.

            deadline = None
            title = ""
            details_link = None

            # Iterate cells to find date-like string
            for cell in cells:
                cell_text = cell.inner_text().strip()
                parsed = self.parse_date(cell_text)
                if parsed:
                    # Check if it's in the future (likely the deadline, not start date)
                    # Solicitations usually have Start and End. End > Start.
                    # If we find multiple dates, the later one is likely the deadline.
                    if deadline is None or parsed > deadline:
                        deadline = parsed

                # Title usually is the longest text or has a link
                if len(cell_text) > 10 and not parsed:
                    if not title:
                        title = cell_text
                    # Check for link
                    link = cell.locator("a").first
                    if link.count():
                        details_link = link.get_attribute("href")
                        if not title: title = cell_text # prioritize link text?

            if not deadline:
                continue

            self.total_found += 1

            if not self.is_qualified(deadline):
                continue

            # Filter by keyword if search didn't work effectively or to be double sure
            keywords = ["Construction", "Paving", "Roofing", "HVAC", "Renovation", "Demolition"]
            if not any(k.lower() in text.lower() for k in keywords):
                continue

            # If we are here, we have a qualified lead.
            # Now we might need to drill down for description/budget if not in table.
            # Optimization: User said "Only extract deep details... for rows that pass".

            description = title # Default
            budget = 0

            if details_link:
                # Determine absolute URL
                if not details_link.startswith("http"):
                    details_link = "https://portal.ct.gov" + details_link

                # Optional: Visit page to get description?
                # For this v2, let's keep it simple/fast. If description is empty, maybe visit.
                # But we have title.
                pass
            else:
                details_link = url

            city = self.infer_city(title, text)

            row_data = {
                "clientName": "CT DAS", # Default for CTSource
                "title": title,
                "slug": self.create_slug(title),
                "description": description,
                "walkthroughDate": "",
                "rfiDate": "",
                "deadline": deadline.isoformat(),
                "budgetMin": budget,
                "jobStreet": "",
                "jobCity": city,
                "jobState": "CT",
                "jobZip": "",
                "portfolioLink": details_link,
                "status": "open"
            }
            self.results.append(row_data)

if __name__ == "__main__":
    scraper = CTScraper()
    print("Starting scrape...")
    results, total = scraper.scrape(progress_callback=lambda x: print(x))
    print(f"Found {len(results)} qualified results out of {total} total.")
    filename, _ = scraper.save_csv()
    print(f"Saved to {filename}")
