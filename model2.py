import pandas as pd
from scipy.stats import describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance

merged_stats_file = "merged_stats_all_years.csv"
merged_stats = pd.read_csv(merged_stats_file)
merged_stats['Interaction_Cat'] = merged_stats['Loc'] + "_" + merged_stats['Opp']
merged_stats['Lag1_FPts'] = merged_stats.groupby('Player')['Pts*'].shift(1)
merged_stats['Lag2_FPts'] = merged_stats.groupby('Player')['Pts*'].shift(2)
merged_stats['FPts_RollingAvg'] = merged_stats.groupby('Player')['Pts*'] \
    .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
merged_stats['Interaction_Num'] = merged_stats['Lag1_FPts'] * merged_stats['FPts_RollingAvg']


print("Dataset head:")
print(merged_stats.head())

merged_stats['Outperformed'] = (merged_stats['Net_score'] > 0).astype(int)


categorical_features = ['Player','Team', 'Opp', 'Loc', 'Interaction_Cat']
numeric_features = ['Interaction_Num', 'Rec_Predicted', 'Yard.1', 'TD.1_Predicted', "FPts_RollingAvg"]
numeric_features += ['Lag1_FPts', 'Lag2_FPts']
all_features = categorical_features + numeric_features
target = 'Outperformed'

missing_columns = [col for col in all_features + [target] if col not in merged_stats.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    raise ValueError("Some required columns are missing in the dataset.")

# Fill missing numeric values with 0
merged_stats[numeric_features] = merged_stats[numeric_features].fillna(0)
merged_stats = merged_stats.dropna(subset=[target])
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Calculate scale_pos_weight
class_0_count = (y == 0).sum()
class_1_count = (y == 1).sum()
scale_pos_weight = class_0_count / class_1_count


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [13, 15],
    'n_estimators': [300, 400, 500],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [scale_pos_weight]
}
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss"),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
model = grid_search.best_estimator_
#model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight)


# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save_model("player_outperformance_model.json")
print("Model saved to 'player_outperformance_model.json'")

plot_importance(model)



