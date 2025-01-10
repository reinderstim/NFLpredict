import pandas as pd
from scipy.stats import describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

merged_stats_file = "merged_stats_all_years.csv"
merged_stats = pd.read_csv(merged_stats_file)

print("Dataset head:")
print(merged_stats.head())

merged_stats['Outperformed'] = (merged_stats['Net_score'] > 0).astype(int)


categorical_features = ['Team', 'Opp', 'Loc']
numeric_features = ['Att_Predicted', 'Yard', 'TD_Predicted', 'Rec_Predicted', 'Yard.1', 'TD.1_Predicted']
all_features = categorical_features + numeric_features
target = 'Outperformed'

# Ensure all columns exist in the dataset
missing_columns = [col for col in all_features + [target] if col not in merged_stats.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    raise ValueError("Some required columns are missing in the dataset.")

# Fill missing numeric values with 0 (if appropriate) and drop rows with missing critical target values
merged_stats[numeric_features] = merged_stats[numeric_features].fillna(0)
merged_stats = merged_stats.dropna(subset=[target])

# Print dataset size after relaxed filtering
print(f"Dataset size after relaxed filtering: {merged_stats.shape}")
print(f"Outperformed Stats: {pd.value_counts(merged_stats['Outperformed'])}")

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
try:
    categorical_encoded = encoder.fit_transform(merged_stats[categorical_features])
except ValueError as e:
    print(f"Error during encoding: {e}")
    raise

# Save the one-hot encoder
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Combine numeric and encoded categorical features
X = np.hstack([categorical_encoded, merged_stats[numeric_features].values])
y = merged_stats[target].values

# Verify sizes of X and y
print(f"Feature matrix shape after relaxed filtering: {X.shape}, Target vector shape: {y.shape}")

if X.shape[0] == 0:
    raise ValueError("Filtered dataset has no rows remaining. Check data preprocessing steps.")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Print training and testing set sizes
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Initialize the XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save_model("player_outperformance_model.json")
print("Model saved to 'player_outperformance_model.json'")



