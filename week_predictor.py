import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


# Load the trained model
model = XGBClassifier()
model.load_model("player_outperformance_model.json")

# Load the saved encoder
with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)


# User-defined function to load weekly data and predict outcomes
def predict_week(file_path):
    # Load the week's data
    week_data = pd.read_csv(file_path)

    # Print the dataset head for verification
    print(f"Dataset head for {file_path}:")
    print(week_data.head())

    # Load merged stats and calculate rolling average
    merged_stats_file = "merged_stats_all_years.csv"
    merged_stats = pd.read_csv(merged_stats_file)

    # Calculate the rolling average of actual points grouped by Player (exclude current week)
    merged_stats['FPts_RollingAvg'] = merged_stats.groupby('Player')['Pts*'].transform(
        lambda x: x.shift().rolling(window=2, min_periods=1).mean()
    )

    # Keep only the most recent rolling average for each player
    latest_stats = merged_stats.sort_values('Week', ascending=False).drop_duplicates('Player')

    # Merge the rolling average into the week's data
    week_data = week_data.merge(
        latest_stats[['Player', 'FPts_RollingAvg']],
        on='Player',
        how='left'
    )

    # Fill missing rolling average with 0 if players have no history
    week_data['FPts_RollingAvg'] = week_data['FPts_RollingAvg'].fillna(0)

    # Map columns to match training feature names
    column_mapping = {
        "Att": "Att_Predicted",
        "TD": "TD_Predicted",
        "Rec": "Rec_Predicted",
        "Yard.1": "Yard.1",
        "TD.1": "TD.1_Predicted"
    }

    week_data = week_data.rename(columns=column_mapping)
    week_data['Loc'] = week_data['Opp'].apply(lambda x: 'Away' if '@' in x else 'Home')
    week_data['Opp'] = week_data['Opp'].str.replace('@', '', regex=False)

    # Define categorical and numeric features
    categorical_features = ['Player', 'Team', 'Opp', 'Loc']
    numeric_features = ['Att_Predicted', 'Yard', 'TD_Predicted', 'Rec_Predicted', 'Yard.1', 'TD.1_Predicted',
                        'FPts_RollingAvg']

    # Check for missing numeric columns and fill them
    for col in numeric_features:
        if col not in week_data.columns:
            print(f"Filling missing column: {col}")
            week_data[col] = 0  # Default for missing numeric columns

    # Ensure all categorical features exist
    for col in categorical_features:
        if col not in week_data.columns:
            print(f"Filling missing column: {col}")
            week_data[col] = "Unknown"  # Default for missing categorical columns

    # Transform categorical features using the loaded encoder
    categorical_encoded = encoder.transform(week_data[categorical_features])

    # Align the feature matrix by ensuring consistency in columns
    train_encoded_columns = encoder.get_feature_names_out(categorical_features)
    week_encoded_df = pd.DataFrame(categorical_encoded, columns=train_encoded_columns)

    # Combine numeric and encoded categorical features
    week_X = np.hstack([week_encoded_df.values, week_data[numeric_features].values])

    # Check for mismatched feature dimensions
    if week_X.shape[1] != model.n_features_in_:
        print(f"Feature shape mismatch: model expects {model.n_features_in_}, got {week_X.shape[1]}")

        # Identify extra and missing columns
        expected_columns = set(train_encoded_columns.tolist() + numeric_features)
        actual_columns = set(week_encoded_df.columns.tolist() + numeric_features)
        extra_columns = actual_columns - expected_columns
        missing_columns = expected_columns - actual_columns

        print(f"Extra columns: {extra_columns}")
        print(f"Missing columns: {missing_columns}")

        # Adjust features by dropping extra columns and adding missing ones
        week_encoded_df = week_encoded_df.drop(columns=list(extra_columns), errors="ignore")
        for col in missing_columns:
            week_encoded_df[col] = 0  # Add missing columns with default value 0

        # Recombine the feature matrix
        week_X = np.hstack([week_encoded_df.values, week_data[numeric_features].values])

    # Ensure final feature matrix aligns
    if week_X.shape[1] != model.n_features_in_:
        raise ValueError("Feature alignment failed after adjustment.")

    # Predict "Outperformed" for the week
    week_predictions = model.predict(week_X)

    # Add predictions to the dataset
    week_data["Outperformed"] = week_predictions

    # Save predictions
    output_file = file_path.replace(".csv", "_predictions.csv")
    week_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

    # Group by player and show unique predictions
    unique_predictions = week_data.groupby("Player").first().reset_index()
    print("\nSample Predictions (Unique Players):")
    print(unique_predictions[["Player", "Team", "Opp", "Outperformed"]].head(35))


# Example usage
week_file = "Week_17_preds.csv"  # Replace with the desired week file
predict_week(week_file)
