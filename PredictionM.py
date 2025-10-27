import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from datetime import date
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate
from keras.callbacks import EarlyStopping
from dateutil.relativedelta import relativedelta

# Load datasets
acs_data = pd.read_excel('ACS.xlsx')
labor_data = pd.read_excel('LaborData.xlsx')
eviction_data = pd.read_excel('evictionsInput.csv')

# Get unique geoids from ACS data
census_tract_list = acs_data['geoid_2020_census_tract'].unique().tolist()

# Static features from ACS data
static_feature_list = ['total-households', 'total-renter-occupied-households', 
                       'total-owner-occupied-households-mortgage', 'total-owner-occupied-households']

def load_and_prepare_data(train_start, train_end, val_start, val_end, test_start, test_end, lookback=6):
    """
    Prepare data for training, validation, and testing
    """
    # Filter eviction data by date range
    eviction_data['month'] = pd.to_datetime(eviction_data['month'])
    
    # Filter labor data by date range
    labor_data['Period'] = pd.to_datetime(labor_data['Year'].astype(str) + '-' + labor_data['Period'])
    
    all_data = []
    
    for geoid in census_tract_list:
        # Get static features for this geoid
        static_features = acs_data[acs_data['geoid_2020_census_tract'] == geoid][static_feature_list].values
        
        if len(static_features) == 0:
            continue
            
        static_features = static_features[0]
        
        # Get eviction history for this geoid
        geoid_evictions = eviction_data[eviction_data['geoid'] == geoid].sort_values('month')
        
        # Get labor statistics (same for all geoids in Arizona)
        labor_features = labor_data[['labor force participation', 'employment', 
                                     'labor force', 'employed', 'unemployed', 'unemployment rate']].values
        
        all_data.append({
            'geoid': geoid,
            'static': static_features,
            'evictions': geoid_evictions,
            'labor': labor_features
        })
    
    return all_data

def create_sequences(eviction_history, labor_history, lookback=6):
    """
    Create sequences for LSTM from time series data
    """
    X_evictions, X_labor, y = [], [], []
    
    for i in range(lookback, len(eviction_history)):
        X_evictions.append(eviction_history[i-lookback:i])
        X_labor.append(labor_history[i-lookback:i])
        y.append(eviction_history[i])
    
    return np.array(X_evictions), np.array(X_labor), np.array(y)

def data_normalization(data, min_val=None, max_val=None):
    """
    Normalize data to [0, 1] range
    """
    if min_val is None:
        min_val = np.min(data)
        max_val = np.max(data)
    
    normalized = (data - min_val) / (max_val - min_val + 1e-8)
    return normalized, min_val, max_val

def data_denormalization(y_min, y_max, prediction):
    """
    Denormalize predictions back to original scale
    """
    prediction = (prediction * (y_max - y_min)) + y_min
    return prediction

def get_common_reg_metrics(ground_truth, prediction):
    """
    Calculate RMSE and Spearman correlation
    """
    ground_truth, prediction = np.squeeze(ground_truth), np.squeeze(prediction)
    rmse = mean_squared_error(ground_truth, prediction, squared=False)
    spearman, _ = stats.spearmanr(ground_truth, prediction)
    return rmse, spearman

def MultiView_model(x_train1, x_train2, x_train3, y_train, x_val1, x_val2, x_val3, y_val, 
                    x_test1, x_test2, x_test3):
    """
    Multi-view neural network model combining static, dynamic, and case history features
    """
    y_train = np.reshape(y_train, (-1, 1))
    y_val = np.reshape(y_val, (-1, 1))
    
    # View 1: Case history (evictions) - LSTM
    input_view1 = Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    x1 = LSTM(units=16, return_sequences=False)(input_view1)
    x1 = Dense(8, activation="relu")(x1)
    output_layer_1 = Dense(4, activation="relu")(x1)
    
    # View 2: Dynamic features (labor statistics) - LSTM
    input_view2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    x2 = LSTM(units=16, return_sequences=False)(input_view2)
    x2 = Dense(8, activation="relu")(x2)
    output_layer_2 = Dense(1, activation='sigmoid')(x2)
    
    # View 3: Static features (ACS demographic data) - Dense
    input_view3 = Input(shape=(x_train3.shape[1],))
    x3 = Dense(16, activation="relu")(input_view3)
    x3 = Dense(8, activation="relu")(x3)
    output_layer_3 = Dense(1, activation='sigmoid')(x3)
    
    # Combine all views
    final_layer = Concatenate()([output_layer_1, output_layer_2, output_layer_3])
    output_layer = Dense(1, activation='sigmoid')(final_layer)
    
    # Build and compile model
    model = Model(inputs=[input_view1, input_view2, input_view3], outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    
    # Train model
    model.fit(x=[x_train1, x_train2, x_train3], 
              y=y_train, 
              batch_size=32, 
              epochs=200, 
              verbose=0,
              callbacks=[es], 
              validation_data=([x_val1, x_val2, x_val3], y_val))
    
    # Make predictions
    test_prediction = model.predict([x_test1, x_test2, x_test3], verbose=0)
    
    return test_prediction

def MARTIAN_algorithm(input_feature_len=6, step=1):
    """
    Main algorithm for eviction rate prediction using sliding window approach
    """
    train_start = date(2019, 1, 1)
    train_end = date(2024, 12, 31)
    
    # Initialize result storage
    all_predictions = []
    rmse_total, spearman_total, counter = 0, 0, 0
    
    # Sliding window validation from Oct 2020 to Dec 2024
    for valid_start_date in pd.date_range(start='2020-10-01', end='2024-07-01', freq='3MS'):
        counter += 1
        
        val_start = valid_start_date.to_pydatetime().date()
        val_end = (val_start + relativedelta(months=3, days=-1))
        test_start = (valid_start_date + relativedelta(months=3)).to_pydatetime().date()
        test_end = (test_start + relativedelta(months=3, days=-1))
        
        print(f"\nIteration {counter}: Val {val_start} to {val_end}, Test {test_start} to {test_end}")
        
        # Prepare data would go here - this is a placeholder
        # In practice, you'd call your data_preparation function
        # sw_train_x_static, sw_train_x_dynamic, sw_train_x_cases, sw_train_y, etc.
        
        # For now, this is a skeleton - you need to implement data_preparation
        # based on your specific data structure
        
    # Predict for 2025 (12 months)
    print("\n" + "="*50)
    print("Predicting eviction rates for 2025...")
    print("="*50)
    
    predictions_2025 = []
    
    for month in range(1, 13):
        pred_date = date(2025, month, 1)
        print(f"Predicting for {pred_date.strftime('%B %Y')}...")
        
        # Here you would:
        # 1. Prepare test data for this month
        # 2. Use the trained model to predict
        # 3. Denormalize predictions
        
        for geoid in census_tract_list:
            predictions_2025.append({
                'geoid': geoid,
                'month': pred_date.strftime('%Y-%m'),
                'predicted_evictions': 0  # Placeholder - replace with actual prediction
            })
    
    # Create final output DataFrame
    results_df = pd.DataFrame(predictions_2025)
    results_df = results_df[['geoid', 'month', 'predicted_evictions']]
    
    # Save to Excel
    results_df.to_excel('eviction_predictions_2025.xlsx', index=False)
    print(f"\nPredictions saved to 'eviction_predictions_2025.xlsx'")
    print(f"Total predictions: {len(results_df)} ({len(census_tract_list)} geoids × 12 months)")
    
    return results_df

# Run the algorithm
if __name__ == "__main__":
    print("Starting MARTIAN Eviction Prediction Algorithm...")
    print(f"Number of census tracts: {len(census_tract_list)}")
    
    results = MARTIAN_algorithm(input_feature_len=6, step=1)
    print("\nPrediction complete!")
    print(results.head(20))