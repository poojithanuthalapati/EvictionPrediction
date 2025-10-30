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
from myutils import data_preparation, data_normalization, parse_labor_period

# Load datasets
acs_data = pd.read_excel("ACS.xlsx")              
labor_data = pd.read_excel("LaborData.xlsx")      
eviction_data = pd.read_csv("evictionsInput.csv")

# Define static features from ACS data
static_feature_list = ['DP05_0039PE', 'DP05_0044PE', 'DP05_0038PE', 'DP05_0052PE',
                       'DP05_0037PE','DP02_0061PE', 'DP02_0068PE', 'DP02_0062PE',
                       'DP02_0060PE', 'DP02_0063PE', 'DP03_0088E', 'DP03_0119PE',
                       'DP05_0001E']


def data_denormalization(y_min, y_max, prediction):
    denominator = y_max - y_min
    if denominator == 0:
        denominator = 1
    prediction = (prediction * denominator) + y_min
    prediction = np.reshape(prediction, (prediction.shape[0], 1))
    return prediction


def get_common_reg_metrics(ground_truth, prediction):
    ground_truth, prediction = np.squeeze(ground_truth), np.squeeze(prediction)
    rmse = mean_squared_error(ground_truth, prediction, squared=False)
    spearman, _ = stats.spearmanr(ground_truth, prediction)
    return rmse, spearman


def MultiView_model(x_train1, x_train2, x_train3, y_train, x_val1, x_val2, x_val3, y_val, x_test1, x_test2, x_test3):
    y_train, y_val = np.reshape(y_train, (-1, 1)), np.reshape(y_val, (-1, 1))

    input_view3 = Input(shape=(x_train3.shape[1],))
    x3 = Dense(16, activation="relu")(input_view3)
    x4 = Dense(8, activation="relu")(x3)
    output_layer_3 = Dense(1, activation='sigmoid')(x4)

    input_view2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    x3_2 = LSTM(units=16)(input_view2)
    x4_2 = Dense(8, activation="relu")(x3_2)
    output_layer_2 = Dense(1, activation='sigmoid')(x4_2)

    input_view1 = Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    x2_3 = LSTM(units=16)(input_view1)
    x4_3 = Dense(8, activation="relu")(x2_3)
    output_layer_1 = Dense(4, activation="relu")(x4_3)

    final_layer = Concatenate()([output_layer_1, output_layer_2, output_layer_3])
    output_layer = Dense(1, activation='sigmoid')(final_layer)

    model = Model(inputs=[input_view1, input_view2, input_view3], outputs=output_layer)
    my_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=my_opt, loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model.fit(x=[x_train1, x_train2, x_train3], verbose=0, y=y_train, batch_size=32, epochs=200, callbacks=[es],
              validation_data=([x_val1, x_val2, x_val3], y_val))
    test_prediction = model.predict([x_test1, x_test2, x_test3])

    return test_prediction


def train_final_model_and_predict_2025(input_feature_len=6, step=1):
    """
    Train model on all available historical data (2019-2024) and predict 2025
    """
    # Use all historical data for training
    train_st_date = date(2019, 1, 1)
    train_end_date = date(2024, 12, 31)  # Train up to end of 2024
    
    print("Training model on historical data (2019-2024)...")
    print("=" * 60)
    
    # Prepare data for training
    sw_train_x_static, sw_train_x_dynamic, sw_train_x_cases, sw_train_y, sw_train_y_gt, \
    sw_val_x_static, sw_val_x_dynamic, sw_val_x_cases, sw_val_y, sw_val_y_gt, \
    sw_test_x_static, sw_test_x_dynamic, sw_test_x_cases, sw_test_y, sw_test_y_gt, \
    min_data, max_data = data_preparation(
        train_st_date, train_end_date, input_feature_len, step,
        acs_data, labor_data, eviction_data, static_feature_list)
    
    print(f"Training samples: {len(sw_train_y)}")
    print(f"Validation samples: {len(sw_val_y)}")
    print(f"Test samples: {len(sw_test_y)}")
    
    # Train the model
    print("\nTraining MultiView model...")
    model = train_multiview_model(
        sw_train_x_cases, sw_train_x_dynamic, sw_train_x_static, sw_train_y,
        sw_val_x_cases, sw_val_x_dynamic, sw_val_x_static, sw_val_y
    )
    
    # Now prepare data for 2025 predictions
    print("\nPreparing data for 2025 predictions...")
    prediction_results = []
    
    # Predict for each month in 2025
    for month in range(1, 13):
        predict_date = date(2025, month, 1)
        print(f"Predicting for {predict_date.strftime('%B %Y')}...")
        
        # Get the most recent data to make predictions
        pred_x_static, pred_x_dynamic, pred_x_cases, geoids = prepare_prediction_data(
            predict_date, input_feature_len, acs_data, labor_data, 
            eviction_data, static_feature_list
        )
        
        if len(pred_x_static) > 0:
            # Make predictions
            predictions = model.predict([pred_x_cases, pred_x_dynamic, pred_x_static], verbose=0)
            predictions_denorm = data_denormalization(min_data, max_data, predictions)
            
            # Store results
            for i, geoid in enumerate(geoids):
                prediction_results.append({
                    'geoid': geoid,
                    'month': predict_date,
                    'predicted_evictions': max(0, predictions_denorm[i][0])  # Ensure non-negative
                })
    
    # Create output DataFrame
    predictions_df = pd.DataFrame(prediction_results)
    
    # Save to CSV
    output_file = 'eviction_predictions_2025.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Predictions saved to: {output_file}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"\nSample predictions:")
    print(predictions_df.head(10))
    print(f"\nSummary statistics:")
    print(predictions_df.groupby('month')['predicted_evictions'].agg(['mean', 'median', 'sum']))
    
    return predictions_df


def train_multiview_model(x_train1, x_train2, x_train3, y_train, x_val1, x_val2, x_val3, y_val):
    """
    Train and return the model (not just predictions)
    """
    y_train, y_val = np.reshape(y_train, (-1, 1)), np.reshape(y_val, (-1, 1))

    input_view3 = Input(shape=(x_train3.shape[1],))
    x3 = Dense(16, activation="relu")(input_view3)
    x4 = Dense(8, activation="relu")(x3)
    output_layer_3 = Dense(1, activation='sigmoid')(x4)

    input_view2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    x3_2 = LSTM(units=16)(input_view2)
    x4_2 = Dense(8, activation="relu")(x3_2)
    output_layer_2 = Dense(1, activation='sigmoid')(x4_2)

    input_view1 = Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    x2_3 = LSTM(units=16)(input_view1)
    x4_3 = Dense(8, activation="relu")(x2_3)
    output_layer_1 = Dense(4, activation="relu")(x4_3)

    final_layer = Concatenate()([output_layer_1, output_layer_2, output_layer_3])
    output_layer = Dense(1, activation='sigmoid')(final_layer)

    model = Model(inputs=[input_view1, input_view2, input_view3], outputs=output_layer)
    my_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=my_opt, loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    
    history = model.fit(
        x=[x_train1, x_train2, x_train3], 
        y=y_train, 
        batch_size=32, 
        epochs=200, 
        callbacks=[es],
        validation_data=([x_val1, x_val2, x_val3], y_val),
        verbose=1
    )
    
    print(f"Training completed. Best epoch: {len(history.history['loss']) - 10}")
    
    return model


def prepare_prediction_data(predict_date, input_feature_len, acs_data, labor_data, eviction_data, static_feature_list):
    """
    Prepare data for making predictions for a specific future date
    """
    # Get historical data up to the prediction date
    eviction_data['month'] = pd.to_datetime(eviction_data['month'])
    
    # Get data from the past 'input_feature_len' months
    start_date = predict_date - relativedelta(months=input_feature_len)
    historical_evictions = eviction_data[
        (eviction_data['month'] >= pd.to_datetime(start_date)) & 
        (eviction_data['month'] < pd.to_datetime(predict_date))
    ].copy()
    
    geoids = historical_evictions['geoid'].unique()
    
    static_features_list = []
    dynamic_features_list = []
    cases_list = []
    valid_geoids = []
    
    # Find geoid column in ACS data
    if 'geoid_2020_census_tract' in acs_data.columns:
        acs_geoid_col = 'geoid_2020_census_tract'
    else:
        geoid_cols = [col for col in acs_data.columns if 'geoid' in col.lower()]
        acs_geoid_col = geoid_cols[0] if geoid_cols else acs_data.columns[0]
    
    # Parse labor data period
    if 'Period' in labor_data.columns:
        labor_data['Period'] = labor_data['Period'].apply(parse_labor_period)
    else:
        if 'Year' in labor_data.columns and len(labor_data.columns) > 1:
            period_col = labor_data.columns[1]
            labor_data['Period'] = labor_data['Year'].astype(str) + ' ' + labor_data[period_col].astype(str)
            labor_data['Period'] = labor_data['Period'].apply(parse_labor_period)
    
    labor_features = ['labor force participation', 'employment', 'unemployment', 'unemployment rate']
    available_labor_features = [col for col in labor_features if col in labor_data.columns]
    
    if len(available_labor_features) == 0:
        numeric_cols = labor_data.select_dtypes(include=[np.number]).columns
        available_labor_features = [col for col in numeric_cols if col not in ['Year']]
    
    labor_features = available_labor_features
    
    for geoid in geoids:
        geoid_data = historical_evictions[historical_evictions['geoid'] == geoid].sort_values('month')
        
        if len(geoid_data) < input_feature_len:
            continue
        
        # Get static features
        geoid_str = str(int(geoid)) if isinstance(geoid, (int, float)) else str(geoid)
        acs_row = acs_data[acs_data[acs_geoid_col].astype(str) == geoid_str]
        
        if len(acs_row) == 0:
            acs_row = acs_data[acs_data[acs_geoid_col] == geoid]
        
        if len(acs_row) == 0:
            static_features = acs_data[static_feature_list].mean().values
        else:
            static_features = acs_row[static_feature_list].values[0]
        
        # Get dynamic features
        dynamic_features = []
        for idx, row in geoid_data.iterrows():
            month = row['month']
            labor_month = labor_data[
                (labor_data['Period'].dt.year == month.year) &
                (labor_data['Period'].dt.month == month.month)
            ]
            if len(labor_month) > 0:
                dynamic_features.append(labor_month[labor_features].values[0])
            else:
                if len(dynamic_features) > 0:
                    dynamic_features.append(dynamic_features[-1])
                else:
                    dynamic_features.append(np.zeros(len(labor_features)))
        
        dynamic_features = np.array(dynamic_features[-input_feature_len:])
        cases = geoid_data['evictions'].values[-input_feature_len:]
        
        if len(dynamic_features) == input_feature_len and len(cases) == input_feature_len:
            static_features_list.append(static_features)
            dynamic_features_list.append(dynamic_features)
            cases_list.append(cases)
            valid_geoids.append(geoid)
    
    if len(static_features_list) == 0:
        return np.array([]), np.array([]), np.array([]), []
    
    static_array = np.array(static_features_list)
    dynamic_array = np.array(dynamic_features_list)
    cases_array = np.array(cases_list).reshape(len(cases_list), input_feature_len, 1)
    
    return static_array, dynamic_array, cases_array, valid_geoids


def MARTIAN_algorithm(input_feature_len=6, step=1):
    """
    Original algorithm for model evaluation on historical test data
    """
    train_st_date, train_end_date = date(2019, 1, 1), date(2021, 7, 30)
    ground_truth_data_test, predicted_data_test = None, None
    rmse, spearsman, counter = 0, 0, 0

    print("Running MARTIAN algorithm for model evaluation...")
    print("=" * 60)

    for valid_start_date in pd.date_range(start='10/1/2020', end=train_end_date, freq='3MS'):
        counter += 1
        test_st_date = (valid_start_date + relativedelta(months=3)).to_pydatetime().date()
        test_end_date = (test_st_date + relativedelta(months=3, days=-1))

        print(f"\nIteration {counter}: Testing period {test_st_date} to {test_end_date}")

        sw_train_x_static, sw_train_x_dynamic, sw_train_x_cases, sw_train_y, sw_train_y_gt, sw_val_x_static, sw_val_x_dynamic, sw_val_x_cases, sw_val_y, sw_val_y_gt, sw_test_x_static, sw_test_x_dynamic, sw_test_x_cases, sw_test_y, sw_test_y_gt, min_data, max_data = data_preparation(
            train_st_date, test_end_date, input_feature_len, step,
            acs_data, labor_data, eviction_data, static_feature_list)

        predicted_test = MultiView_model(x_train1=sw_train_x_cases, x_train2=sw_train_x_dynamic,
                                         x_train3=sw_train_x_static, y_train=sw_train_y, x_val1=sw_val_x_cases,
                                         x_val2=sw_val_x_dynamic, x_val3=sw_val_x_static , y_val=sw_val_y,
                                         x_test1=sw_test_x_cases, x_test2=sw_test_x_dynamic, x_test3=sw_test_x_static)
        predicted_test_i, ground_truth_test_i = data_denormalization(min_data, max_data, predicted_test), sw_test_y_gt
        rmse_test, spearsman_test = get_common_reg_metrics(ground_truth_test_i, predicted_test_i)

        print(f"  RMSE: {rmse_test:.4f}, Spearman: {spearsman_test:.4f}")

        rmse += rmse_test
        spearsman += spearsman_test

        if ground_truth_data_test is None:
            ground_truth_data_test = ground_truth_test_i
            predicted_data_test = predicted_test_i
        else:
            ground_truth_data_test = np.concatenate((ground_truth_data_test, ground_truth_test_i))
            predicted_data_test = np.concatenate((predicted_data_test, predicted_test_i))

    print(f"\n{'=' * 60}")
    print(f"Average RMSE: {rmse / counter:.4f}")
    print(f"Average Spearman: {spearsman / counter:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Run evaluation on historical data
    print("STEP 1: Model Evaluation on Historical Data")
    MARTIAN_algorithm()
    
    # Train final model and predict 2025
    print("\n\nSTEP 2: Training Final Model and Predicting 2025")
    predictions_2025 = train_final_model_and_predict_2025()
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)