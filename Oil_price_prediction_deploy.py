import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smf
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import streamlit as st
import datetime

# Read the CSV file
df = pd.read_csv(r"C:\Users\user\Downloads\crude-oil-price.csv")

# Extract prices and scale the data
prices = df['price'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Load the trained model and scaler
model = load_model(r"C:\Users\user\Downloads\model.h5")  # Replace 'your_model_path.h5' with the actual path to your saved model

# Streamlit app
def main():
    st.title("Crude Oil Price Prediction")

    # User input for prediction
    st.subheader("Predict Oil Price for Input Date")
    input_date = st.date_input("Select an input date for prediction")

    # Check if the input date is in the future
    if input_date < datetime.date.today():
        st.write("Please select a future date for prediction.")
        return

    # Prepare input data for prediction
    sequence_length = 10
    last_sequence = scaled_prices[-sequence_length:]
    input_data = last_sequence.reshape(1, sequence_length, 1)

    # Calculate the number of days from today to the input date
    num_days = (input_date - datetime.date.today()).days

    # Make prediction for input date
    predicted_prices = []
    for _ in range(num_days):
        predicted_price = model.predict(input_data)
        predicted_prices.append(predicted_price)
        input_data = np.concatenate((input_data[:, 1:, :], predicted_price.reshape(1, 1, 1)), axis=1)

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Display the predicted price for the input date
    st.write(f"Predicted Oil Price for {input_date}: {predicted_prices[-1][0]}")

    # Additional visualization of historical data
    st.subheader("Historical Oil Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['price'], label='Actual Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Oil Price')
    ax.set_title('Historical Oil Prices')
    ax.legend()
    st.pyplot(fig)
    
if __name__ == '__main__':
    main()
