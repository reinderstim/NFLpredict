import pandas as pd
import matplotlib.pyplot as plt

# Load the merged dataset
merged_stats = pd.read_csv("merged_stats_all_years.csv")


# Select columns for predicted vs actual fantasy points
predicted = merged_stats["FPts"]
actual = merged_stats["Pts*"]
plt.boxplot(merged_stats['Net_score'])
min_value = merged_stats["Net_score"].min()
min_index = merged_stats['Net_score'].idxmin()
min_label = merged_stats.index[min_index]
print("Lowest Net Score:", min_label, min_value)
percentage = (merged_stats['Net_score'] < 0).mean() * 100

print(percentage)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(predicted, actual, alpha=0.7)
plt.plot([min(predicted), max(predicted)], [min(predicted), max(predicted)], color='red', linestyle='--', label="Perfect Prediction")
plt.title("Predicted vs. Actual Fantasy Points")
plt.xlabel("Predicted Fantasy Points")
plt.ylabel("Actual Fantasy Points")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
