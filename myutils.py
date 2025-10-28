import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def data_normalization(data, min_val=None, max_val=None):
    """
    Normalize data to [0, 1] range
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data), min_val, max_val
    
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def prepare_sequence_data(data, input_len, step=1):
    """
    Create sequences for LSTM input
    data: 2D array (samples, features)
    input_len: number of time steps to look back
    step: prediction horizon
    """
    X, y = [], []
    for i in range(len(data) - input_len - step + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len+step-1])
    
    return np.array(X), np.array(y)


def data_preparation(train_start, test_end, input_feature_len, step, 
                     acs_data, labor_data, eviction_data, static_feature_list):
    """
    Prepare train, validation, and test data for the MARTIAN model
    
    Returns:
    - Training data (static, dynamic, cases, y, y_ground_truth)
    - Validation data (static, dynamic, cases, y, y_ground_truth)
    - Test data (static, dynamic, cases, y, y_ground_truth)
    - min_data, max_data for denormalization
    """
    
    # Convert dates to datetime
    train_start = pd.to_datetime(train_start)
    test_end = pd.to_datetime(test_end)
    
    # Calculate validation and test split dates
    total_days = (test_end - train_start).days
    val_start = train_start + timedelta(days=int(total_days * 0.6))
    test_start = train_start + timedelta(days=int(total_days * 0.8))
    
    # Process eviction data (target variable and cases)
    eviction_data['month'] = pd.to_datetime(eviction_data['month'])
    eviction_filtered = eviction_data[
        (eviction_data['month'] >= train_start) & 
        (eviction_data['month'] <= test_end)
    ].copy()
    
    # Get unique geoids
    geoids = eviction_filtered['geoid'].unique()
    
    # Process labor data (dynamic features)
    labor_data['Period'] = pd.to_datetime(labor_data['Period'])
    labor_filtered = labor_data[
        (labor_data['Period'] >= train_start) & 
        (labor_data['Period'] <= test_end)
    ].copy()
    
    # Select numeric columns from labor data
    labor_features = ['labor force participation', 'employment', 'unemployment', 'unemployment rate']
    labor_features = [col for col in labor_features if col in labor_data.columns]
    
    # Process ACS data (static features)
    # Assuming ACS data has 'geoid_2020_census_tract' column
    if 'geoid_2020_census_tract' in acs_data.columns:
        acs_geoid_col = 'geoid_2020_census_tract'
    else:
        # Find the geoid column
        geoid_cols = [col for col in acs_data.columns if 'geoid' in col.lower()]
        acs_geoid_col = geoid_cols[0] if geoid_cols else acs_data.columns[0]
    
    # Prepare data arrays
    all_static = []
    all_dynamic = []
    all_cases = []
    all_targets = []
    all_dates = []
    
    for geoid in geoids:
        # Get eviction data for this geoid
        geoid_evictions = eviction_filtered[eviction_filtered['geoid'] == geoid].sort_values('month')
        
        if len(geoid_evictions) < input_feature_len + step:
            continue
        
        # Get static features for this geoid
        acs_row = acs_data[acs_data[acs_geoid_col] == geoid]
        if len(acs_row) == 0:
            # Use mean values if geoid not found
            static_features = acs_data[static_feature_list].mean().values
        else:
            static_features = acs_row[static_feature_list].values[0]
        
        # Get dynamic features (labor data) - using aggregated monthly values
        dynamic_features = []
        for idx, row in geoid_evictions.iterrows():
            month = row['month']
            labor_month = labor_filtered[
                (labor_filtered['Period'].dt.year == month.year) &
                (labor_filtered['Period'].dt.month == month.month)
            ]
            if len(labor_month) > 0:
                dynamic_features.append(labor_month[labor_features].values[0])
            else:
                # Use previous month's values or zeros
                if len(dynamic_features) > 0:
                    dynamic_features.append(dynamic_features[-1])
                else:
                    dynamic_features.append(np.zeros(len(labor_features)))
        
        dynamic_features = np.array(dynamic_features)
        
        # Get eviction counts as time series (cases)
        cases = geoid_evictions['evictions'].values
        
        # Store data
        for i in range(len(cases) - input_feature_len - step + 1):
            all_static.append(static_features)
            all_dynamic.append(dynamic_features[i:i+input_feature_len])
            all_cases.append(cases[i:i+input_feature_len])
            all_targets.append(cases[i+input_feature_len+step-1])
            all_dates.append(geoid_evictions.iloc[i+input_feature_len+step-1]['month'])
    
    # Convert to arrays
    all_static = np.array(all_static)
    all_dynamic = np.array(all_dynamic)
    all_cases = np.array(all_cases)
    all_targets = np.array(all_targets)
    all_dates = pd.Series(all_dates)
    
    # Reshape cases for LSTM (samples, timesteps, features)
    all_cases = all_cases.reshape(all_cases.shape[0], all_cases.shape[1], 1)
    
    # Split into train, val, test based on dates
    train_mask = all_dates < val_start
    val_mask = (all_dates >= val_start) & (all_dates < test_start)
    test_mask = all_dates >= test_start
    
    # Training data
    train_static = all_static[train_mask]
    train_dynamic = all_dynamic[train_mask]
    train_cases = all_cases[train_mask]
    train_y_gt = all_targets[train_mask]
    
    # Validation data
    val_static = all_static[val_mask]
    val_dynamic = all_dynamic[val_mask]
    val_cases = all_cases[val_mask]
    val_y_gt = all_targets[val_mask]
    
    # Test data
    test_static = all_static[test_mask]
    test_dynamic = all_dynamic[test_mask]
    test_cases = all_cases[test_mask]
    test_y_gt = all_targets[test_mask]
    
    # Normalize target variable
    train_y, min_y, max_y = data_normalization(train_y_gt)
    val_y = (val_y_gt - min_y) / (max_y - min_y) if max_y != min_y else np.zeros_like(val_y_gt)
    test_y = (test_y_gt - min_y) / (max_y - min_y) if max_y != min_y else np.zeros_like(test_y_gt)
    
    return (train_static, train_dynamic, train_cases, train_y, train_y_gt,
            val_static, val_dynamic, val_cases, val_y, val_y_gt,
            test_static, test_dynamic, test_cases, test_y, test_y_gt,
            min_y, max_y)