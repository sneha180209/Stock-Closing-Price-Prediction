from flask import Flask, render_template, request, jsonify
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pandas_datareader as pdr

app = Flask(__name__)
training_data_len = 0

# Function to load and preprocess the data
def load_data():
    key = "30c0973b8a648106ae38faea9031b9a3924c7469"
    df = pdr.get_data_tiingo('DIS', api_key=key)
    df.to_csv('DIS.csv')
    df = pd.read_csv('DIS.csv', date_parser=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df.filter(['close']).values
    scaled_data = scaler.fit_transform(data)

    return df, scaler, scaled_data

# Function to build and train the LSTM model
def build_train_model(scaled_data):
    training_data_len = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    return model


def predict_visualize_data(model, scaled_data, scaler, df):
    test_data = scaled_data[training_data_len - 60:, :]

    x_test, y_test = [], df['close'][training_data_len:].values
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    if len(x_test.shape) == 2:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    elif len(x_test.shape) == 3:
        x_test = x_test
    else:
        raise ValueError("Unexpected shape of x_test:", x_test.shape)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(f"RMSE: {rmse}")

    train = df[:training_data_len]
    test = df[training_data_len:]
    test['predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['close'])
    plt.plot(test[['close', 'predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
    plt.savefig('static/plot.png')


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the button click event
@app.route('/run_model', methods=['POST'])
def run_model():
    df, scaler, scaled_data = load_data()
    model = build_train_model(scaled_data)
    predict_visualize_data(model, scaled_data, scaler, df)
    return jsonify({'result': 'Model executed successfully'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
