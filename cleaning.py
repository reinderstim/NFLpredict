import pandas as pd

# List of years to process
years = [2022, 2023, 2024]

# Placeholder for all merged datasets
all_merged_stats = []

for year in years:
    print(f"Processing data for {year}...")

    predicted_file = f"{year}_predicted_stats.csv"
    actual_file = f"actual_stats_{year}.csv"

    predicted_stats = pd.read_csv(predicted_file)
    actual_stats = pd.read_csv(actual_file)

    print(f"Predicted stats columns for {year}: {predicted_stats.columns.tolist()}")
    print(f"Actual stats columns for {year}: {actual_stats.columns.tolist()}")

    #Player name column
    if '"Player\nSort First:    Last:"' in predicted_stats.columns:
        predicted_stats.rename(columns={'"Player\nSort First:    Last:"': 'Player'}, inplace=True)
    elif 'Player\nSort First:    Last:' in predicted_stats.columns:
        predicted_stats.rename(columns={'Player\nSort First:    Last:': 'Player'}, inplace=True)

    #Check
    if 'Player' not in predicted_stats.columns or 'Week' not in predicted_stats.columns:
        print(f"Missing 'Player' or 'Week' column in predicted stats for {year}. Skipping this year.")
        continue

    predicted_stats.rename(columns=lambda x: x.strip(), inplace=True)
    actual_stats.rename(columns=lambda x: x.strip(), inplace=True)

    # Verify if required columns exist
    if 'Player' not in predicted_stats.columns or 'Week' not in predicted_stats.columns:
        print(f"Missing 'Player' or 'Week' column in predicted stats for {year}. Skipping this year.")
        continue

    if 'Player' not in actual_stats.columns or 'Week' not in actual_stats.columns:
        print(f"Missing 'Player' or 'Week' column in actual stats for {year}. Skipping this year.")
        continue

    predicted_stats['Loc'] = predicted_stats['Opp'].apply(lambda x: 'Away' if '@' in x else 'Home')
    predicted_stats['Opp'] = predicted_stats['Opp'].str.replace('@', '', regex=False)

    merged_stats = pd.merge(predicted_stats, actual_stats, on=['Player', 'Week'], suffixes=('_Predicted', '_Actual'))


    merged_stats['Year'] = year

    # Calculate Net Score
    merged_stats['Net_score'] = merged_stats["Pts*"] - merged_stats["FPts"]

    # Append to the overall list
    all_merged_stats.append(merged_stats)

# Concatenate all merged datasets
if all_merged_stats:
    final_merged_stats = pd.concat(all_merged_stats, ignore_index=True)

    #Save the file
    output_file = "merged_stats_all_years.csv"
    final_merged_stats.to_csv(output_file, index=False)
    print(f"All years' data merged and saved to {output_file}.")
else:
    print("No data was merged due to missing columns.")
