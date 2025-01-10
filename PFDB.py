from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Set up Selenium WebDriver
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"
BASE_URL = "https://www.footballdb.com/fantasy-football/index.html"
POSITION = "WR"
YEAR = 2022

# Configure Chrome options
options = webdriver.ChromeOptions()
#options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

# Create a Selenium service and driver
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

def fetch_weekly_stats(start_week, end_week):
    all_weeks_data = []
    headers = []
    MAX_RETRIES = 3

    for week in range(start_week, end_week + 1):
        print(f"Fetching data for Week {week}...")
        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                params = f"?pos={POSITION}&yr={YEAR}&wk={week}"
                driver.get(BASE_URL + params)

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )

                # Locate the table
                table = driver.find_element(By.TAG_NAME, "table")
                rows = table.find_elements(By.TAG_NAME, "tr")

                if not headers:
                    headers = [col.text for col in rows[1].find_elements(By.TAG_NAME, "th")]
                    print("Headers fetched:", headers)

                for row in rows[2:]:
                    cells = [cell.text for cell in row.find_elements(By.TAG_NAME, "td")]
                    if len(cells) == len(headers):
                        all_weeks_data.append([week] + cells)

                success = True  # If successful, exit retry loop

            except TimeoutException:
                retries += 1
                print(f"Retry {retries} for Week {week}...")

            except Exception as e:
                print(f"Error fetching data for Week {week}: {e}")
                break

        if not success:
            print(f"Failed to fetch data for Week {week} after {MAX_RETRIES} retries.")

    return headers, all_weeks_data



# Fetch data for a range of weeks
headers, all_weeks_data = fetch_weekly_stats(4, 4)

driver.quit()

# Create a DataFrame from the scraped data
if headers and all_weeks_data:
    columns = ["Week"] + headers
    df = pd.DataFrame(all_weeks_data, columns=columns)
    print(df.head())

    # Save the DataFrame to a CSV file
    #output_file = "actual_stats_2022.csv"
    output_file = "week4_2022.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
else:
    print("No data fetched.")
