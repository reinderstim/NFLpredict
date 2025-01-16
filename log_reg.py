import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged stats dataset
merged_stats_file = "merged_stats_all_years.csv"
merged_stats = pd.read_csv(merged_stats_file)
merged_stats['FPts_RollingAvg'] = merged_stats.groupby('Player')['Pts*'] \
    .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())

print("Dataset head:")
print(merged_stats.head())

merged_stats['Outperformed'] = (merged_stats['Net_score'] > 0).astype(int)
merged_stats['Lag1_FPts'] = merged_stats.groupby('Player')['Pts*'].shift(1)
merged_stats['Lag2_FPts'] = merged_stats.groupby('Player')['Pts*'].shift(2)
merged_stats['Interaction_Num'] = merged_stats['Lag1_FPts'] * merged_stats['FPts_RollingAvg']
merged_stats['Interaction_Cat'] = merged_stats['Loc'] + "_" + merged_stats['Opp']


categorical_features = ['Player', 'Team', 'Opp', 'Loc', 'Interaction_Cat']
numeric_features = ['Rec_Predicted', 'Yard.1', 'TD.1_Predicted', "FPts_RollingAvg", "Interaction_Num"]
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
print(f"Outperformed Stats: {pd.Series(merged_stats['Outperformed']).value_counts()}")

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_encoded = encoder.fit_transform(merged_stats[categorical_features])

# Save the one-hot encoder
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Standardize numeric features
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(merged_stats[numeric_features])
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

X = np.hstack([categorical_encoded, numeric_scaled])
y = merged_stats[target].values

print(f"Feature matrix shape after relaxed filtering: {X.shape}, Target vector shape: {y.shape}")

if X.shape[0] == 0:
    raise ValueError("Filtered dataset has no rows remaining. Check data preprocessing steps.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

'''
# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
'''

# Save the model
with open("player_outperformance_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved to 'player_outperformance_model.pkl'")
# Perform 5-fold cross-validation

skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
cv_predictions = cross_val_predict(model, X, y, cv=5)

# Print cross-validation accuracy
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y, cv_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, cv_predictions))

# Analyze Misclassified Samples
misclassified_indices = np.where(cv_predictions != y)[0]
misclassified_samples = merged_stats.iloc[misclassified_indices]

# Plot Misclassification by Opponent
plt.figure(figsize=(12, 6))
sns.countplot(data=misclassified_samples, x='Opp', order=misclassified_samples['Opp'].value_counts().index)
plt.title("Misclassifications by Opponent Team")
plt.xticks(rotation=45)
plt.show()

# Plot Misclassification by Home/Away
plt.figure(figsize=(6, 4))
sns.countplot(data=misclassified_samples, x='Loc', hue='Outperformed')
plt.title("Misclassifications by Location")
plt.show()


