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

# Load Week 17 predictions
week_17_file = "Week_17_preds.csv"
week_17_data = pd.read_csv(week_17_file)

# Print Week 17 dataset head for verification
print("Week 17 dataset head:")
print(week_17_data.head())

# Map Week 17 columns to match training feature names
column_mapping = {
    "Att": "Att_Predicted",
    "TD": "TD_Predicted",
    "Rec": "Rec_Predicted",
    "Yard.1": "Yard.1",
    "TD.1": "TD.1_Predicted"
}

# Rename columns to match training dataset
week_17_data = week_17_data.rename(columns=column_mapping)
week_17_data['Loc'] = week_17_data['Opp'].apply(lambda x: 'Away' if '@' in x else 'Home')
week_17_data['Opp'] = week_17_data['Opp'].str.replace('@', '', regex=False)

# Define categorical and numeric features
categorical_features = ['Team', 'Opp', 'Loc']
numeric_features = ['Att_Predicted', 'Yard', 'TD_Predicted', 'Rec_Predicted', 'Yard.1', 'TD.1_Predicted', "Net_score"]


# Check for missing numeric columns and fill them
for col in numeric_features:
    if col not in week_17_data.columns:
        print(f"Filling missing column: {col}")
        week_17_data[col] = 0  # Default for missing numeric columns

# Ensure all categorical features exist
for col in categorical_features:
    if col not in week_17_data.columns:
        print(f"Filling missing column: {col}")
        week_17_data[col] = "Unknown"  # Default for missing categorical columns

# Transform categorical features using the loaded encoder
categorical_encoded = encoder.transform(week_17_data[categorical_features])

# Align the feature matrix by ensuring consistency in columns
train_encoded_columns = encoder.get_feature_names_out(categorical_features)
week_17_encoded_df = pd.DataFrame(categorical_encoded, columns=train_encoded_columns)

# Combine numeric and encoded categorical features
week_17_X = np.hstack([week_17_encoded_df.values, week_17_data[numeric_features].values])

# Check for mismatched feature dimensions
if week_17_X.shape[1] != model.n_features_in_:
    print(f"Feature shape mismatch: model expects {model.n_features_in_}, got {week_17_X.shape[1]}")

    # Identify extra and missing columns
    expected_columns = set(train_encoded_columns.tolist() + numeric_features)
    actual_columns = set(week_17_encoded_df.columns.tolist() + numeric_features)
    extra_columns = actual_columns - expected_columns
    missing_columns = expected_columns - actual_columns

    print(f"Extra columns: {extra_columns}")
    print(f"Missing columns: {missing_columns}")

    # Adjust features by dropping extra columns and adding missing ones
    week_17_encoded_df = week_17_encoded_df.drop(columns=list(extra_columns), errors="ignore")
    for col in missing_columns:
        week_17_encoded_df[col] = 0  # Add missing columns with default value 0

    # Recombine the feature matrix
    week_17_X = np.hstack([week_17_encoded_df.values, week_17_data[numeric_features].values])

# Ensure final feature matrix aligns
if week_17_X.shape[1] != model.n_features_in_:
    raise ValueError("Feature alignment failed after adjustment.")

# Predict "Outperformed" for Week 17
week_17_predictions = model.predict(week_17_X)

# Add predictions to the Week 17 dataset
week_17_data["Outperformed"] = week_17_predictions

# Save predictions
output_file = "Week_17_predictions.csv"
week_17_data.to_csv(output_file, index=False)
print(f"Week 17 predictions saved to {output_file}.")

# Print sample predictions
print("\nSample Predictions for Week 17:")
print(week_17_data[["Player", "Team", "Opp", "Outperformed"]].head(20))
