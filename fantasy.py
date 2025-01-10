import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr


class NFLStatPrediction:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2)

    def scrape_season_data(self, year):
        """Scrape season data from Pro Football Reference."""
        url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch data for year {year}. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'fantasy'})

        if table:
            df = pd.read_html(str(table))[0]
            df['season'] = year

            # Flatten multi-level column names
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            print(f"Flattened Columns for year {year}: {df.columns.tolist()}")  # Debugging line
            return df
        else:
            print(f"Fantasy data table not found for year {year}")
            return None

    def load_online_data(self, years):
        """Load data exclusively from online sources for specified years."""
        all_data = []
        for year in years:
            print(f"Scraping data for year {year}...")
            season_data = self.scrape_season_data(year)
            if season_data is not None:
                all_data.append(season_data)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            print("Successfully scraped all data.")
            return df
        else:
            print("No data was successfully scraped.")
            return pd.DataFrame()

    def create_previous_season_data(self, df, feats, groupby_feats):
        """Group and create previous season's stats for a specific position."""
        # Ensure the correct season column is present
        if 'season_' not in df.columns:
            print("Error: 'season_' column is missing.")
            return pd.DataFrame()

        required_columns = set(feats)
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            print(f"Missing columns: {missing_columns}")
            return pd.DataFrame()

        # Group data
        grouped_df = df.loc[:, feats].groupby(groupby_feats, as_index=False).sum()

        # Shift the season column
        _df_prev = grouped_df.copy()
        _df_prev['season_'] = _df_prev['season_'] + 1  # Correct column name
        new_df = grouped_df.merge(_df_prev, on=groupby_feats, suffixes=('', '_prev'), how='left')

        print(f"Columns in merged DataFrame: {new_df.columns.tolist()}")  # Debugging
        return new_df

    def train_model(self, model_data, features, target, train_season, test_season):
        """Train and evaluate a polynomial linear regression model."""
        train_data = model_data[model_data['season_'] == train_season]
        test_data = model_data[model_data['season_'] == test_season]

        # Select features and target
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Debugging: Check for non-numeric values
        print("Checking for non-numeric values in features...")
        non_numeric_rows = X_train[~X_train.applymap(lambda x: isinstance(x, (int, float))).all(axis=1)]
        if not non_numeric_rows.empty:
            print("Found non-numeric values in training data:")
            print(non_numeric_rows)
            raise ValueError("Non-numeric values detected in features. Please check data preprocessing.")

        # Polynomial transformation
        X_train_poly = self.poly.fit_transform(X_train)
        X_test_poly = self.poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        preds = model.predict(X_test_poly)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = pearsonr(y_test, preds)[0] ** 2

        test_data = test_data.copy()
        test_data['preds'] = preds
        return model, test_data, rmse, r2

    def predict_weekly_fantasy_points(self):
        # Define the years to scrape
        years = range(2021, 2024)

        # Load data from online sources
        df = self.load_online_data(years)
        if df.empty:
            print("No data available to run models.")
            return

        # Define features for fantasy points prediction
        fantasy_feats = [
            'season_',  # Updated to match the correct column name
            'Unnamed: 1_level_0_Player',  # Player
            'Unnamed: 2_level_0_Tm',  # Team
            'Receiving_Tgt',  # Receiving targets
            'Receiving_Rec',  # Receiving receptions
            'Receiving_Yds',  # Receiving yards
            'Receiving_TD',  # Receiving touchdowns
            'Fantasy_FantPt'  # Fantasy points
        ]

        groupby_feats = [
            'season_',  # Season
            'Unnamed: 1_level_0_Player',  # Player
            'Unnamed: 2_level_0_Tm'  # Team
        ]

        # Debug: Check columns in the DataFrame
        print(f"Columns in main DataFrame: {df.columns.tolist()}")

        fantasy_df = self.create_previous_season_data(df, fantasy_feats, groupby_feats)

        if fantasy_df.empty:
            print("No data available after creating previous season data.")
            return

        # Debugging: Check for non-numeric data
        print("Data types in merged DataFrame:")
        print(fantasy_df.dtypes)

        print("Cleaning non-numeric values from *_prev columns...")
        for col in ['Receiving_Tgt_prev', 'Receiving_Yds_prev', 'Receiving_TD_prev', 'Fantasy_FantPt']:
            if col in fantasy_df.columns:
                fantasy_df[col] = pd.to_numeric(fantasy_df[col], errors='coerce')

        # Drop rows with NaN in critical columns
        fantasy_df = fantasy_df.dropna(
            subset=['Receiving_Tgt_prev', 'Receiving_Yds_prev', 'Receiving_TD_prev', 'Fantasy_FantPt'])

        # Define features and target for the model
        features = [
            'Receiving_Tgt_prev',  # Previous season targets
            'Receiving_Yds_prev',  # Previous season receiving yards
            'Receiving_TD_prev'  # Previous season receiving touchdowns
        ]
        target = 'Fantasy_FantPt'  # Current season fantasy points

        # Train the model
        fantasy_model, fantasy_test_data, fantasy_rmse, fantasy_r2 = self.train_model(
            fantasy_df, features, target, 2022, 2023
        )

        print(f"Fantasy Points Prediction Model:\nRMSE: {fantasy_rmse}\nR²: {fantasy_r2}")

        # Save predicted vs. actual fantasy points
        comparison = fantasy_test_data[
            ['Unnamed: 1_level_0_Player', 'Unnamed: 2_level_0_Tm', 'Fantasy_FantPt', 'preds']]
        comparison.rename(columns={'Fantasy_FantPt': 'Actual_Fantasy_Points', 'preds': 'Predicted_Fantasy_Points'},
                          inplace=True)
        print("Predicted vs. Actual Fantasy Points:")
        print(comparison.head())

        # Save results to a CSV file for further analysis
        comparison.to_csv('fantasy_points_comparison.csv', index=False)


# Run the prediction
nfl_stat_prediction = NFLStatPrediction()
nfl_stat_prediction.predict_weekly_fantasy_points()
