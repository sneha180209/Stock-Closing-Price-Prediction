from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as pdr

app = Flask(__name__)

key = "30c0973b8a648106ae38faea9031b9a3924c7469"

# Load the dataset
df = pdr.get_data_tiingo('DIS', api_key=key)
df.to_csv('DIS.csv')
df = pd.read_csv('DIS.csv', date_parser=True)

# Preprocess the data
data = df.filter(['close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
np.save('scaler_params.npy', [scaler.min_, scaler.scale_])

# Create the training dataset
train_data_len = int(len(dataset) * 0.8)
train_data = scaled_data[0:train_data_len, :]

x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build and train the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10)
model.save('stock_price_prediction_model.h5')

# Load the trained model
model = tf.keras.models.load_model('stock_price_prediction_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    # Load the scaler parameters
    scaler_params = np.load('scaler_params.npy', allow_pickle=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_, scaler.scale_ = scaler_params

    # Preprocess the latest data
    latest_data = df.filter(['close'])
    latest_dataset = latest_data.values
    latest_scaled_data = scaler.transform(latest_dataset)
    last_60_days = latest_scaled_data[-60:].reshape(1, -1, 1)

    # Get the predicted price
    pred_price = model.predict(last_60_days)
    pred_price = scaler.inverse_transform(pred_price)

    # Plot and save the graph
    plt.figure(figsize=(16, 8))
    plt.title('Model Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(dataset, label='Actual Data')
    plt.plot(np.append(dataset[-60:], pred_price[0]), label='Prediction', linestyle='dashed')
    plt.legend()
    plt.savefig('static/prediction_plot.png')

    return jsonify({'pred_price': pred_price[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
