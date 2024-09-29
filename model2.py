import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

from special_functions import inverse_log_returns

# Load your data
  # Replace this with your data loading logic
df = pd.read_csv("Data/SPY/Normalized/normalized_5min_data_SPY_2019_to_2024.csv.csv", index_col=0, parse_dates=True)

train_data = df[(df.index < '2024-01-01') & (df.index > '2019-01-01')]  # Data from 2019 to end of 2023 for training
test_data = df[df.index >= '2024-01-01']  # Data from 2024 onwards for testing
df.reset_index(drop=True, inplace=True)


# Remove the datetime column if it exists
if 'datetime' in df.columns:
    df.drop(columns=['datetime'], inplace=True)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
print(df.head())
# Define the model checkpoint
checkpoint = ModelCheckpoint(
    'model_epoch_{epoch:02d}.keras',  # Filename format, saving each model as a .h5 file
    save_freq='epoch',              # Save every epoch
    save_best_only=False,           # Change to True if you only want to save the best model
    # period=2,                       # Save every 2 epochs
    verbose=1                       # Print messages when saving
)

class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch + 1} completed.')



# Create sequences
def create_sequences(data, timesteps=400, target_steps=20):
    X, y = [], []
    for i in range(len(data) - timesteps - target_steps):
        X.append(data[i:(i + timesteps)])
        y.append(data.iloc[i + timesteps:i + timesteps + target_steps, 0])  # Using iloc for proper indexing
    return np.array(X), np.array(y)

# Create sequences with 400 timesteps and predicting the next 20 candles
print("Start making")


X_train, y_train = create_sequences(train_data, timesteps=400)
X_test, y_test = create_sequences(test_data, timesteps=400)
print("created sequences")

# Split into training and testing sets
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Add the Input layer
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.1))

model.add(Dense(y_train.shape[1]))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
print("Starting model training...")
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.1, callbacks=[EpochLogger(), checkpoint])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Making predictions
predictions = model.predict(X_test)
print(predictions)
predictions = inverse_log_returns(predictions)  # Rescale back to original values if necessary
print(predictions)
# Now, classify trades and calculate total profit/loss
# Assume `actual_data` is your DataFrame with the actual prices (not normalized)
actual_data = pd.read_csv("Data/SPY/Actual/actual_5min_data_SPY_2019_to_2024.csv")

# Initialize variables
total_profit = 0
position = 0  # 1 for long position, -1 for short position, 0 for no position
buy_price = 0


def classtify(current, future):
    if float(future) > float(current)*1.005: # will gain 0.5% or more
        return 1
    elif float(future) < float(current)*1.005: # will gain 0.5% or more
        return -1
    else:
        return 0 


# Iterate through the predictions and actual prices
for i in range(len(predictions) - 1):
    current_price = actual_data.iloc[i]["close"]  # Get the current price from the actual data
    future_price = predictions[i + 1][0]  # Get the predicted future price (adjust index if needed)

    action = classtify(current_price, future_price)

    if action == 1 and position == 0:  # Buy signal
        position = 1  # Enter long position
        buy_price = current_price
        print(f"Buying at {buy_price}")

    elif action == -1 and position == 1:  # Sell signal
        position = 0  # Exit long position
        profit = current_price - buy_price
        total_profit += profit
        print(f"Selling at {current_price}, Profit: {profit}")

    elif action == -1 and position == 0:  # Short signal (if you want to implement shorting)
        position = -1  # Enter short position
        buy_price = current_price
        print(f"Shorting at {buy_price}")

    elif action == 1 and position == -1:  # Closing short position
        position = 0  # Exit short position
        profit = buy_price - current_price
        total_profit += profit
        print(f"Covering short at {current_price}, Profit: {profit}")

# If still holding a position at the end, calculate final profit/loss
if position == 1:  # Still holding a long position
    final_profit = actual_data.iloc[-1]["close"] - buy_price
    total_profit += final_profit
    print(f"Final Selling at {actual_data.iloc[-1]['close']}, Profit: {final_profit}")

elif position == -1:  # Still holding a short position
    final_profit = buy_price - actual_data.iloc[-1]["close"]
    total_profit += final_profit
    print(f"Final Covering at {actual_data.iloc[-1]['close']}, Profit: {final_profit}")

print(f"Total Profit/Loss: {total_profit}")
# predictions now contains the predicted future prices