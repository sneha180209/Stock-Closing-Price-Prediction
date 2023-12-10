from flask import Flask, render_template, request
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    key = "30c0973b8a648106ae38faea9031b9a3924c7469"
    df = pdr.get_data_tiingo('DIS', api_key=key)
    df.to_csv('DIS.csv')
    df = pd.read_csv('DIS.csv', date_parser=True)

    # Rest of your code to train the model and make predictions...
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df['close'])
    plt.show()


    #create a dataframe with only 'Close' column
    data=df.filter(['close'])

    #covert dataframe into numpy array
    dataset=data.values

    #get the number of rows to train the model on
    training_data_len=math.ceil(len(dataset)*.8)
    test_data_len=len(dataset)-training_data_len
    # training_data_len

    #scale data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
    # scaled_data

    #create the scaled training data set
    train_data=scaled_data[0:training_data_len, :]

    #split data into x_train and y_train data sets
    x_train=[]
    y_train=[]

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
        if i<=60:
            print(x_train)
            print(y_train)
            print()

    #convert the x_train and y_train to numpy arrays
    x_train, y_train=np.array(x_train), np.array(y_train)

    #reshape the data
    x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model=Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #train the model with 10 epochs and batch size of 32
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    #create the testing dataset
    test_data=scaled_data[training_data_len-60:, :]
    #create the datasets x_test and y_test
    x_test=[]
    y_test=dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    #convert the data to a numpy array
    x_test=np.array(x_test)

    #reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #get the models predicted price values
    predictions=model.predict(x_test)
    predictions=scaler.inverse_transform(predictions)

    #get the rmse
    rmse=np.sqrt(np.mean(predictions-y_test)**2)
    # rmse

    # Save the plot
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['close'])
    plt.plot(test[['close', 'predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
    plot_path = 'static/model_plot.png'  # Save the plot to a static folder
    plt.savefig(plot_path)

    return render_template('result.html', plot_path=plot_path, pred_price=pred_price)

if __name__ == '__main__':
    app.run(debug=True)
