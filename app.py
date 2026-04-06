from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# 1. Load model and data
# Use a relative path or a properly escaped absolute path
model = joblib.load('model.pkl')
# Added 'r' before the string to handle backslashes in Windows paths
df = pd.read_csv(r"D:\ml_project1\AI_Economy_Complete_Index (1).csv")

# 2. Create separate encoders for each column
label_encoders = {}
for col in ['Ticker', 'Sector', 'Industry', 'Role']:
    le = LabelEncoder()
    # Ensure we handle NaNs before encoding
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    historical_data = None
    selected_ticker = None
    selected_date = None

    # Get unique tickers for the dropdown (Inverse transform to show original names)
    tickers = sorted(label_encoders['Ticker'].inverse_transform(df['Ticker'].unique()).tolist())

    if request.method == 'POST':
        selected_ticker = request.form.get('ticker')
        selected_date = request.form.get('date')

        try:
            # Encode the selected ticker
            encoded_ticker = label_encoders['Ticker'].transform([selected_ticker])[0]

            # 3. Filter data for this ticker
            ticker_df = df[df['Ticker'] == encoded_ticker].copy()

            if not ticker_df.empty:
                # Prepare features: We take the most recent row to get company context
                # NOTE: In a real production app, you'd need to convert 'selected_date' 
                # into features if your model relies on time-based inputs.
                sample_data = ticker_df.iloc[-1].drop(['Close', 'Date'])
                features = sample_data.values.reshape(1, -1)
                
                # Predict
                pred_value = model.predict(features)[0]
                prediction = round(float(pred_value), 4)

                # 4. Fetch historical data (Last 10 entries)
                ticker_data = ticker_df.tail(10)
                historical_data = {
                    'dates': ticker_data['Date'].astype(str).tolist(),
                    'closes': ticker_data['Close'].tolist()
                }
                
                # Append predicted value for visualization
                historical_data['dates'].append(f"{selected_date} (Predicted)")
                historical_data['closes'].append(prediction)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = "Error processing request"

    return render_template(
        'index.html',
        prediction=prediction,
        tickers=tickers,
        historical_data=historical_data,
        selected_ticker=selected_ticker,
        selected_date=selected_date
    )

if __name__ == '__main__':
    app.run(debug=True)