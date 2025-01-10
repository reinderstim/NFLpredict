from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Configuration
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"
BASE_URL = "https://fftoday.com/rankings/playerwkproj.php"
YEAR = 2024
POSITION = 30  # Wide Receiver



options = webdriver.ChromeOptions()
#options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Correct initialization
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)


def fetch_predicted_stats():
    all_weeks_data = []
    headers = []

    for week in range(17, 18):  # Test with weeks 1 and 2
        print(f"Fetching predictions for Week {week}...")
        params = f"?Season={YEAR}&GameWeek={week}&PosID={POSITION}&LeagueID="
        url = BASE_URL + params
        print(f"Fetching URL: {url}")

        retries = 3
        for attempt in range(retries):
            try:
                driver.get(url)
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
                )
                break  # Exit loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(5)

        try:
            # Locate the specific table (Table 8)
            tables = driver.find_elements(By.TAG_NAME, "table")
            print(f"Number of tables found: {len(tables)}")
            if len(tables) < 8:
                print("Could not find the expected table.")
                continue

            table = tables[7]  # Select Table 8 (Index starts at 0)
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"Number of rows in Table 8: {len(rows)}")

            # Fetch headers if not already done
            if not headers:
                headers = [cell.text for cell in rows[2].find_elements(By.TAG_NAME, "td")]
                headers.insert(0, "Week")  # Add Week column to headers
                print(f"Headers fetched: {headers}")

            # Fetch data rows
            for row in rows[3:]:  # Data rows start at index 3
                cells = [cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td")]
                if len(cells) == len(headers) - 1:  # Match number of columns (excluding Week)
                    all_weeks_data.append([week] + cells)  # Prepend Week data
                else:
                    print(f"Skipped row due to mismatch: {cells}")

        except Exception as e:
            print(f"Failed to fetch data for Week {week}: {e}")
            with open(f"week_{week}_debug.html", "w") as file:
                file.write(driver.page_source)

    return headers, all_weeks_data


# After Fetching
headers, all_weeks_data = fetch_predicted_stats()

# Verify if data is fetched
if headers and all_weeks_data:
    df = pd.DataFrame(all_weeks_data, columns=headers)
    print(df.head())

    # Ensure the 'Player' column is properly named
    if '"Player\nSort First:    Last:"' in df.columns:
        df.rename(columns={'"Player\nSort First:    Last:"': 'Player'}, inplace=True)
    elif 'Player\nSort First:    Last:' in df.columns:
        df.rename(columns={'Player\nSort First:    Last:': 'Player'}, inplace=True)

    # Save the DataFrame to a CSV file
    output_file = "Week_17_preds.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
else:
    print("No data fetched.")

