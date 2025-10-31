import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Show current directory
print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
for file in os.listdir():
    print(f"  {file}")

# Try to find the CSV file
print("\nLooking for evictionInput.csv...")
if os.path.exists('evictionInput.csv'):
    print("Found it!")
else:
    print("Not found in current directory")
    print("\nPlease move evictionInput.csv, ACS.xlsx, and LaborData.xlsx to:")
    print(os.getcwd())
    input("\nPress Enter after moving the files...")

# Load data
print("\nLoading data...")
eviction_df = pd.read_csv('evictionInput.csv')

# Convert month to datetime
eviction_df['month'] = eviction_df['month'].astype(str)
eviction_df['date'] = pd.to_datetime(eviction_df['month'].str[:7] + '-01')

# Sum evictions by month
monthly = eviction_df.groupby('date')['evictions'].sum().reset_index()
monthly = monthly.sort_values('date')

print(f"Data from {monthly['date'].min()} to {monthly['date'].max()}")
print(f"Total months: {len(monthly)}")

# Simple prediction: use last 6 months average + trend
last_6 = monthly.tail(6)
avg = last_6['evictions'].mean()
trend = (last_6['evictions'].iloc[-1] - last_6['evictions'].iloc[0]) / 6

# Predict 2025 (12 months)
predictions = []
for month in range(1, 13):
    predicted = avg + (trend * month)
    predicted = max(0, predicted)  # No negative values
    predictions.append({
        'month': f'2025-{month:02d}',
        'evictions': int(predicted)
    })

# Convert to dataframe
pred_df = pd.DataFrame(predictions)

# Print predictions
print("\n2025 Predictions:")
print("="*40)
for _, row in pred_df.iterrows():
    print(f"{row['month']}: {row['evictions']:,} evictions")

# Save to CSV
pred_df.to_csv('predictions_2025.csv', index=False)
print("\nSaved to predictions_2025.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly['date'], monthly['evictions'], 'b-', label='Historical', marker='o')
plt.plot(pd.to_datetime(pred_df['month']), pred_df['evictions'], 'r--', label='2025 Predictions', marker='s')
plt.xlabel('Date')
plt.ylabel('Evictions')
plt.title('Eviction Predictions for 2025')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predictions.png')
print("Chart saved to predictions.png")
plt.show()

print("\nDone!")
