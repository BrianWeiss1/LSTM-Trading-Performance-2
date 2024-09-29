import tensorflow as tf
from keras.layers import LSTM, Dropout, Dense
import os
from keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the saved model

FUTURE_PERIOD_PREDICT = 20 # how much candles in the future we're predicting
SEQ_LEN = FUTURE_PERIOD_PREDICT*20 # how much candles we're looking at before predicting

def classtify(current, future):
    if float(future) > float(current)*1.005: # will gain 0.5% or more
        return 1
    elif float(future) < float(current)*1.005: # will gain 0.5% or more
        return -1
    else:
        return 0 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

df['future'] = main_df[f"RA"]

def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))),
    model.add(Dropout(0.1))
    
    model.add(LSTM(units=50), return_sequences=True)
    model.add(Dropout(0.1))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    return model


if os.path.exists('Model/path_to_my_model.h5'):
    model = tf.keras.models.load_model('Model/path_to_my_model.h5')
else:
    model = create_model()


# # Save the entire model
# model.save('path_to_my_model.h5')  # Save to an .h5 file or any path