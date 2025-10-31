import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Show current directory
print("Current directory:", os.getcwd())

# Load data
print("\nLoading data...")
eviction_df = pd.read_csv('evictionsInput.csv')

# Convert month to datetime
eviction_df['month'] = eviction_df['month'].astype(str)
eviction_df['date'] = pd.to_datetime(eviction_df['month'].str[:7] + '-01')

# Sum evictions by month
monthly = eviction_df.groupby('date')['evictions'].sum().reset_index()
monthly = monthly.sort_values('date')

print(f"Data from {monthly['date'].min()} to {monthly['date'].max()}")
print(f"Total months: {len(monthly)}")

# Create features from ALL data
monthly['year'] = monthly['date'].dt.year
monthly['month_num'] = monthly['date'].dt.month
monthly['time_index'] = range(len(monthly))
monthly['quarter'] = monthly['date'].dt.quarter
monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num'] / 12)
monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num'] / 12)

# Add lag features
for lag in [1, 2, 3, 6, 12]:
    monthly[f'lag_{lag}'] = monthly['evictions'].shift(lag)

# Add rolling features
for window in [3, 6, 12]:
    monthly[f'rolling_mean_{window}'] = monthly['evictions'].rolling(window=window).mean()
    monthly[f'rolling_std_{window}'] = monthly['evictions'].rolling(window=window).std()

# Remove rows with NaN
train_data = monthly.dropna().copy()

print(f"Training samples: {len(train_data)}")

# Prepare features
feature_cols = ['time_index', 'year', 'month_num', 'quarter', 'month_sin', 'month_cos'] + \
               [col for col in train_data.columns if 'lag_' in col or 'rolling_' in col]

X_train = train_data[feature_cols]
y_train = train_data['evictions']

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Create 2025 prediction data
future_dates = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_df = pd.DataFrame({'date': future_dates})
future_df['year'] = 2025
future_df['month_num'] = future_df['date'].dt.month
future_df['quarter'] = future_df['date'].dt.quarter
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month_num'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month_num'] / 12)
future_df['time_index'] = range(len(monthly), len(monthly) + 12)

# Get last known values for predictions
all_values = list(monthly['evictions'].values)

# Predict for 2025
predictions_2025 = []

for i, row in future_df.iterrows():
    idx = len(monthly) + i
    
    # Calculate lag features
    future_df.loc[i, 'lag_1'] = all_values[idx - 1] if idx >= 1 else 0
    future_df.loc[i, 'lag_2'] = all_values[idx - 2] if idx >= 2 else 0
    future_df.loc[i, 'lag_3'] = all_values[idx - 3] if idx >= 3 else 0
    future_df.loc[i, 'lag_6'] = all_values[idx - 6] if idx >= 6 else 0
    future_df.loc[i, 'lag_12'] = all_values[idx - 12] if idx >= 12 else 0
    
    # Calculate rolling features
    recent = all_values[max(0, idx-12):idx]
    future_df.loc[i, 'rolling_mean_3'] = np.mean(recent[-3:]) if len(recent) >= 3 else 0
    future_df.loc[i, 'rolling_mean_6'] = np.mean(recent[-6:]) if len(recent) >= 6 else 0
    future_df.loc[i, 'rolling_mean_12'] = np.mean(recent[-12:]) if len(recent) >= 12 else 0
    future_df.loc[i, 'rolling_std_3'] = np.std(recent[-3:]) if len(recent) >= 3 else 0
    future_df.loc[i, 'rolling_std_6'] = np.std(recent[-6:]) if len(recent) >= 6 else 0
    future_df.loc[i, 'rolling_std_12'] = np.std(recent[-12:]) if len(recent) >= 12 else 0
    
    # Predict
    X_pred = future_df.loc[i:i, feature_cols]
    pred = max(0, rf_model.predict(X_pred)[0])
    all_values.append(pred)
    predictions_2025.append(int(pred))

future_df['evictions'] = predictions_2025

# Print predictions
print("\n2025 Predictions:")
print("="*40)
for _, row in future_df.iterrows():
    print(f"{row['date'].strftime('%B')}: {row['evictions']:,} evictions")

# Save predictions to CSV
output_df = future_df[['date', 'evictions']].copy()
output_df['month'] = output_df['date'].dt.strftime('%Y-%m')
output_df = output_df[['month', 'evictions']]
output_df.to_csv('predictions_2025.csv', index=False)
print("\nSaved to predictions_2025.csv")

# Prepare data for plotting by year
monthly['month_name'] = monthly['date'].dt.month

# Separate by year
data_2022 = monthly[monthly['year'] == 2022].copy()
data_2023 = monthly[monthly['year'] == 2023].copy()
data_2024 = monthly[monthly['year'] == 2024].copy()

# Add month names for 2025
future_df['month_name'] = future_df['date'].dt.month

# Create the plot
plt.figure(figsize=(14, 7))

# Month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_positions = list(range(1, 13))

# Plot historical years in shades of blue
if len(data_2022) > 0:
    plt.plot(data_2022['month_name'], data_2022['evictions'], 
             'o-', color='#1f77b4', linewidth=2, markersize=6, label='2022')

if len(data_2023) > 0:
    plt.plot(data_2023['month_name'], data_2023['evictions'], 
             's-', color='#4a9eff', linewidth=2, markersize=6, label='2023')

if len(data_2024) > 0:
    plt.plot(data_2024['month_name'], data_2024['evictions'], 
             '^-', color='#89c4ff', linewidth=2, markersize=6, label='2024')

# Plot 2025 predictions in green
plt.plot(future_df['month_name'], future_df['evictions'], 
         'D-', color='#2ecc71', linewidth=3, markersize=7, label='2025 (Predicted)')

# Format the plot
plt.xlabel('Month', fontsize=14, fontweight='bold')
plt.ylabel('Number of Evictions', fontsize=14, fontweight='bold')
plt.title('Monthly Evictions by Year (2022-2025)', fontsize=16, fontweight='bold')
plt.xticks(month_positions, month_labels)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the plot
plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
print("Chart saved to predictions.png")

# Summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

for year in [2022, 2023, 2024]:
    year_data = monthly[monthly['year'] == year]
    if len(year_data) > 0:
        print(f"{year}: {int(year_data['evictions'].sum()):>8,} total | {int(year_data['evictions'].mean()):>6,} avg/month")

print(f"2025: {sum(predictions_2025):>8,} total | {int(np.mean(predictions_2025)):>6,} avg/month (Predicted)")

print("\nDone!")