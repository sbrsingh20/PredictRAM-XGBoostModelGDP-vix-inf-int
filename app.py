import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Define directories
stocks_folder = 'stockdata'  # Folder containing stock data
gdp_file_path = 'GDP data.xlsx'  # Path to GDP data file
results_folder = 'results'  # Folder to save results

# Load GDP Data with Inflation, Interest Rate, and VIX
gdp_data = pd.read_excel(gdp_file_path, engine='openpyxl')
gdp_data['Date'] = pd.to_datetime(gdp_data['Date'], format='%d-%m-%Y')  # Ensure correct date format
gdp_data.set_index('Date', inplace=True)

# Adjust GDP Data to Daily Frequency (Forward Fill missing dates)
gdp_data_daily = gdp_data.resample('D').ffill()

# Initialize a dictionary to hold all models
all_models = {}

# Process each stock data file and train models
for stock_file in os.listdir(stocks_folder):
    if stock_file.endswith(".xlsx") and not stock_file.startswith("~$"):  # Skip temporary Excel files
        stock_file_path = os.path.join(stocks_folder, stock_file)
        
        try:
            # Load the stock data
            stock_df = pd.read_excel(stock_file_path, engine='openpyxl')
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df.set_index('Date', inplace=True)

            # Merge stock data with GDP data (on Date index)
            merged_df = pd.merge(stock_df[['Close']], gdp_data_daily[['GDP', 'Inflation', 'Interest Rate', 'VIX']], 
                                 left_index=True, right_index=True, how='inner')

            # Calculate stock returns based on 'Close' price
            merged_df['Stock_Returns'] = merged_df['Close'].pct_change()

            # Drop NaN values that result from pct_change or missing values
            merged_df.dropna(inplace=True)

            # Features: GDP data, Inflation, Interest Rate, and VIX
            X = merged_df[['GDP', 'Inflation', 'Interest Rate', 'VIX']]
            # Target: Stock Returns (based on 'Close' price)
            y = merged_df['Stock_Returns']

            # Split data into training and testing sets (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a pipeline for preprocessing and modeling
            pipeline = Pipeline(steps=[ 
                ('scaler', StandardScaler()),  # Scaling the features
                ('model', xgb.XGBRegressor(n_estimators=100, random_state=42))  # XGBoost model
            ])

            # Fit the model using the pipeline
            pipeline.fit(X_train, y_train)

            # Save the trained model to the all_models dictionary
            all_models[stock_file] = {
                'model': pipeline,
                'evaluation': {
                    'accuracy': 0,  # Placeholder for accuracy (if needed)
                    'r2_score': 0,  # Placeholder for R2 score (if needed)
                    'mean_squared_error': 0  # Placeholder for MSE (if needed)
                }
            }

            # Save the trained model as a separate file
            model_filename = os.path.join(results_folder, f"{stock_file}_xgb_model.pkl")
            joblib.dump(pipeline, model_filename)

        except Exception as e:
            print(f"Error processing {stock_file}: {e}")

# Streamlit App Section
st.title("Stock Prediction using XGBoost")

# Dropdown to select a stock model
stock_model = st.selectbox('Select a stock model', list(all_models.keys()))

# If the user selects a valid model, make predictions
if stock_model:
    # Load the selected model from all_models
    model = all_models[stock_model]['model']

    # Input fields for GDP, Inflation, Interest Rate, and VIX
    gdp = st.number_input('Enter GDP value', value=0.0)
    inflation = st.number_input('Enter Inflation rate', value=0.0)
    interest_rate = st.number_input('Enter Interest Rate', value=0.0)
    vix = st.number_input('Enter VIX value', value=0.0)

    # Create a DataFrame for prediction
    input_data = pd.DataFrame([[gdp, inflation, interest_rate, vix]], columns=['GDP', 'Inflation', 'Interest Rate', 'VIX'])

    # Predict stock returns if inputs are valid
    if st.button('Predict Stock Returns'):
        if input_data.isnull().values.any():
            st.error("Please fill in all the input fields.")
        else:
            prediction = model.predict(input_data)
            st.write(f"Predicted Stock Return: {prediction[0]:.4f}")
else:
    st.error("Please select a valid stock model.")

