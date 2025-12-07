import pandas as pd

df = pd.read_csv('sleep_clean.csv')

# Keep ONLY these columns
keep_cols = [
    'Age',
    'Sleep duration',
    'Awakenings',
    'Caffeine consumption',
    'Alcohol consumption',
    'Exercise frequency',
    'Gender_Male',
    'Smoking status_Yes',
    'Sleep efficiency'  # Target variable
]

df_clean = df[keep_cols]
df_clean.to_csv('sleep_clean_8features.csv', index=False)
print(f"Saved! Columns: {list(df_clean.columns)}")
