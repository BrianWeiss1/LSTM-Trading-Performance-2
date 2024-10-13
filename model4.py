import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

class TradingEnvironment:
    def __init__(self, data, lookback_window=400):
        self.data = data
        self.lookback_window = lookback_window
        self.current_step = lookback_window
        self.holding_time = 0
        self.max_holding_time = 20  # Maximum number of steps to hold a position

    def reset(self):
        self.current_step = self.lookback_window
        self.holding_time = 0
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step - self.lookback_window:self.current_step].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate reward
        if action == 1:  # Buy
            reward = (next_price - current_price) / current_price
            self.holding_time += 1
        elif action == 2:  # Sell
            reward = (current_price - next_price) / current_price
            self.holding_time += 1
        else:  # Hold
            reward = 0
            self.holding_time = 0

        # Penalize for holding too long
        if self.holding_time > self.max_holding_time:
            reward -= 0.001 * self.holding_time

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, done

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.state_size, 5), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Load and preprocess data
df = pd.read_csv("Data/SPY/Normalized/normalized_5min_data_SPY_2019_to_2024.csv.csv", index_col=0, parse_dates=True)
train_data = df[(df.index < '2024-01-01') & (df.index > '2019-01-01')]
test_data = df[df.index >= '2024-01-01']


# Initialize environment and agent
env = TradingEnvironment(train_data)
agent = TradingAgent(400, 3)  # 400 lookback window, 3 actions (buy, sell, hold)

# Training
episodes = 100
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 400, 5])  # Reshape for LSTM input
    for time in range(len(train_data) - 401):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, 400, 5])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Testing
env = TradingEnvironment(test_data)
state = env.reset()
state = np.reshape(state, [1, 400, 5])
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    next_state = np.reshape(next_state, [1, 400, 5])
    total_reward += reward
    state = next_state

print(f"Total reward on test data: {total_reward}")