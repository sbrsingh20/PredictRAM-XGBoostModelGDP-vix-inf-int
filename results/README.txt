Stock Prediction Model Using XGBoost

1. To load a model, use the following code:
   model = joblib.load('path_to_model.pkl')

2. To predict stock returns, pass a DataFrame with the following columns:
   - 'GDP': Gross Domestic Product
   - 'Inflation': Inflation rate
   - 'Interest Rate': Interest rate
   - 'VIX': Volatility Index

3. Model Evaluation:
   - Accuracy (R-squared %)
   - Mean Squared Error

For further instructions, refer to the model evaluation file and data.