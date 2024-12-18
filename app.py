import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define directories
stocks_folder = 'stockdata'  # Folder containing stock data
gdp_file_path = 'GDP data.xlsx'  # Path to GDP data file
results_folder = 'results'  # Folder to save results

# Create results folder if it does not exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Function to load data, train and save models
def train_models():
    # Load GDP Data with Inflation, Interest Rate, and VIX
    gdp_data = pd.read_excel(gdp_file_path, engine='openpyxl')
    gdp_data['Date'] = pd.to_datetime(gdp_data['Date'], format='%d-%m-%Y')  # Ensure correct date format
    gdp_data.set_index('Date', inplace=True)

    # Adjust GDP Data to Daily Frequency (Forward Fill missing dates)
    gdp_data_daily = gdp_data.resample('D').ffill()

    # Initialize a dictionary to hold all models
    all_models = {}

    # Process each stock data file
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

                # If merged dataframe is empty, continue to next file
                if merged_df.empty:
                    continue

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

                # Make predictions using the test data
                y_pred = pipeline.predict(X_test)

                # Evaluate the model's performance
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                accuracy = r2 * 100  # R-squared percentage

                # Add the trained model to all_models
                all_models[stock_file] = {
                    'model': pipeline,
                    'evaluation': {
                        'r2_score': r2,
                        'mean_squared_error': mse,
                        'accuracy': accuracy,
                        'model_type': 'XGBoost'
                    }
                }

                # Save the trained model as a separate file
                model_filename = os.path.join(results_folder, f"{stock_file}_xgb_model.pkl")
                joblib.dump(pipeline, model_filename)

            except Exception as e:
                st.write(f"Error processing {stock_file}: {e}")

    # Save all models in a Pickle File
    all_models_filename = os.path.join(results_folder, 'all_stock_gdp_vix_models.pkl')
    joblib.dump(all_models, all_models_filename)

    st.write("Model training and evaluation complete for all stocks. Results saved.")


# Function to load pre-trained models and make predictions
def make_predictions():
    # Load all models
    all_models_filename = os.path.join(results_folder, 'all_stock_gdp_vix_models.pkl')
    all_models = joblib.load(all_models_filename)

    # Select stock model
    stock_model = st.selectbox("Select Stock Model", list(all_models.keys()))

    # Input fields for GDP, Inflation, Interest Rate, and VIX
    gdp = st.number_input("Enter GDP Value", value=0.0)
    inflation = st.number_input("Enter Inflation Value", value=0.0)
    interest_rate = st.number_input("Enter Interest Rate Value", value=0.0)
    vix = st.number_input("Enter VIX Value", value=0.0)

    # Prepare input data
    input_data = pd.DataFrame({
        'GDP': [gdp],
        'Inflation': [inflation],
        'Interest Rate': [interest_rate],
        'VIX': [vix]
    })

    # Load the selected model and make a prediction
    model = all_models[stock_model]['model']
    prediction = model.predict(input_data)

    st.write(f"Predicted Stock Return for {stock_model}: {prediction[0]}")

    # Display model evaluation results
    eval_results = all_models[stock_model]['evaluation']
    st.write(f"Model Evaluation Results for {stock_model}:")
    st.write(f"Accuracy (R-squared %): {eval_results['accuracy']:.4f}%")
    st.write(f"Mean Squared Error: {eval_results['mean_squared_error']:.4f}")


# Streamlit UI

# Title of the web app
st.title("Stock Return Prediction Using GDP, Inflation, Interest Rate, and VIX")

# Select between training the models or making predictions
app_mode = st.selectbox("Choose Action", ["Train Models", "Make Predictions"])

if app_mode == "Train Models":
    st.write("This will train models using the provided stock data and GDP data.")
    if st.button("Train Models"):
        train_models()
elif app_mode == "Make Predictions":
    st.write("Make predictions for stock returns using pre-trained models.")
    make_predictions()
