#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:05:48 2024

@author: bouzid-s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, RepeatedKFold


def prepare_data(file_path):


    # Load the data
    data = pd.read_csv(file_path)

    # Convert columns to numeric, coerce errors to NaN
    columns_to_convert = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagnetoX','MagnetoY','MagnetoZ','X', 'Y']
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')


    # Extract features and labels
    features = ['AccelY', 'AccelX', 'AccelZ', 'GyroZ', 'GyroX','MagnetoX','MagnetoY','MagnetoZ', 'GyroY']
    X_data = data[features].values
    Y_data = data[['X', 'Y']].values

    # Normalize features
    scaler = MinMaxScaler()
    X_data_normalized = scaler.fit_transform(X_data)

    # Split into train and test sets ( 70% of the data for training and the remaining 30% for testing.)
    train_size = int(len(X_data) * 0.7)
    X_train, X_test = X_data_normalized[:train_size], X_data_normalized[train_size:]
    y_train, y_test = Y_data[:train_size], Y_data[train_size:]

    return X_train, y_train, X_test, y_test, Y_data



def create_sequence(X, y, seq_length):

    Input = []
    Output = []

    # fenetre glissant
    for i in range(seq_length, len(X)):
        # Include Y(t-1)
        seq_x = np.hstack([X[i-seq_length:i], y[i-seq_length:i]])
        Input.append(seq_x)
        Output.append(y[i])

    return np.array(Input), np.array(Output)


def build_model(X_train):
    model = Sequential()

    model.add(LSTM(units=248, activation='relu',  return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=124, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='linear'))  #add a dense layer with two output unit: X,Y
    #define the loss function and optimizer
    model.compile(optimizer='adam', loss='mse') #y^-y=mse

    model.summary()
    return model

def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    # enumerate folds
    for train_ix, val_ix in cv.split(X_train):
        # prepare data
        X_train_, X_val_ = X_train[train_ix, :], X_train[val_ix, :]
        y_train_, y_val_ = y_train[train_ix, :], y_train[val_ix, :]
        # fit the keras model on the dataset
        history = model.fit(X_train_, y_train_, epochs=200, batch_size=32, validation_data=(X_val_, y_val_),
                            use_multiprocessing=True)

    # history = model.fit(X_train, y_train, epochs=200, batch_size=32,validation_split=0.1428, shuffle=False,
    #                 verbose=1)#, callbacks=[early_stopping])
    #c'est posible avec plusieurs data set. meme model.
    return history

def test_model(model, X_test, y_test, seq_length):
    # Inicializar arrays para almacenar las secuencias de predicción
    y_pred_test = np.zeros_like(y_test)

    y_pred_test[:] = y_test[:]

    Input = []
    Output = []

    for i in range(seq_length, len(X_test)):
        # Include Y(t-1)
        seq_x = np.hstack([X_test[i-seq_length:i], y_pred_test[i-seq_length:i]])
        Input.append(seq_x)


        seq_x_reshaped = seq_x.reshape(1, seq_length, -1)
        seq_y = model.predict(seq_x_reshaped)
        y_pred_test[i] = seq_y
        Output.append(seq_y)


    Input = np.array(Input)

    Output = np.array(Output)
    Output = Output.squeeze()



    return Input, Output

def evaluate_model(model, X_real_test, y_real_test):
    loss = model.evaluate(X_real_test, y_real_test) #a verifier la fonction on aura y_test_pred

    # y_test_pred = model.predict(X_test)# ) #a verifier la fonction on aura
    # y_test_real ()
    # loss = mse(y_test,y_pred_test)
    return loss

def plot_results(history, data, data_train, data_test, future_values, label=""):
    # Losses Plot in Semilog
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))

    # Graficar las series originales de prueba
    plt.plot(np.arange(len(data)), data[:, -2], label='Original Test X', linestyle='-', color='b')
    plt.plot(np.arange(len(data)), data[:, -1], label='Original Test Y', linestyle='-', color='r')

    # Graficar las predicciones futuras
    plt.plot(np.arange(len(data_train), len(data_train) + len(future_values)), future_values[:, 0], label='Predicted Future X', linestyle='--', color='b')
    plt.plot(np.arange(len(data_train), len(data_train) + len(future_values)), future_values[:, 1], label='Predicted Future Y', linestyle='--', color='r')

    plt.title("Original Test Data and Predicted Future Values "+label)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file_path = 'trajectories/data_square_motion_samsung.csv'
    #file_path = 'trajectories\data_horizontal_motion_esp32.csv'
    #file_path = 'trajectories\data_vertical_motion_esp32.csv'
    #file_path = 'trajectories\data_square_motion_samsung.csv'
    #file_path = 'trajectories\data_vertical_motion_samsung.csv'


    seq_length =6

    X_train, y_train, X_test, y_test, Y_data = prepare_data(file_path)



    #LSTM input: (samples, time_steps, features)
    X_train, y_train =  create_sequence(X_train, y_train, seq_length)
    X_real_test, y_real_test = create_sequence(X_test, y_test, seq_length)

    #Building the LSTM Model
    model = build_model(X_train)

    #Training model
    history = train_model(model, X_train, y_train)

    # Predicting future values
    X_pred_test, y_pred_test = test_model(model, X_test, y_test, seq_length)


    # Evaluate the model

    loss = evaluate_model(model, X_real_test, y_pred_test)

    #plot
    plot_results(history, Y_data, y_train, y_test, y_pred_test, "[VS]")