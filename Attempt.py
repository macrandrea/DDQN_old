import tensorflow as tf
import numpy as np
import random as rnd
from Ambiente import Ambiente

class StockTrader:
    def __init__(self, initial_budget, stock_prices):
        self.budget = initial_budget
        self.stock_prices = stock_prices
        self.stock_count = 0
        self.history = []

    def take_action(self, action):
        if action == "buy" and self.budget >= self.stock_prices[0]:
            self.budget -= self.stock_prices[0]
            self.stock_count += 1
            self.history.append(("buy", self.stock_prices[0]))
        elif action == "sell" and self.stock_count > 0:
            self.budget += self.stock_prices[0]
            self.stock_count -= 1
            self.history.append(("sell", self.stock_prices[0]))
        else:
            self.history.append(("hold", self.stock_prices[0]))
        
        self.stock_prices = self.stock_prices[1:]
        
    def get_state(self):
        budget = self.budget
        stock_count = self.stock_count
        stock_price = self.stock_prices[0]
        stock_price_next = self.stock_prices[1]
        return budget, stock_count, stock_price, stock_price_next
    
    def get_reward(self):
        if len(self.history) < 2:
            return 0
        
        prev_action, prev_price = self.history[-2]
        curr_action, curr_price = self.history[-1]
        
        if prev_action == "buy" and curr_action == "sell":
            return curr_price - prev_price
        elif prev_action == "sell" and curr_action == "buy":
            return prev_price - curr_price
        else:
            return 0

# Define Q-network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=(3,), activation="relu")) #24
#model.add(tf.keras.layers.Dense(3, activation="relu")) #24
model.add(tf.keras.layers.Dense(1, activation="linear")) #3
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Define target network
target_model = tf.keras.Sequential()
target_model.add(tf.keras.layers.Dense(3, input_shape=(3,), activation="relu"))
#target_model.add(tf.keras.layers.Dense(3, activation="relu"))
target_model.add(tf.keras.layers.Dense(1, activation="linear"))#3
target_model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Copy weights from main network to target network
target_model.set_weights(model.get_weights())

# Define update frequency for target network
target_update_freq = 10

# Define Q-learning parameters
num_episodes = 1000
epsilon = 0.1
epsilon_decay = 0.995
discount_factor = 0.95
  
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        
    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def sample(self, batch_size):
        rnd_indices = np.random.choice(len(self.buffer), size=batch_size)
        #data = self.buffer[np.random.choice(len(self.buffer),batch_size)]#list(self.buffer[:rnd_indices])#list(self.buffer[:4])[rnd_indices[0]]
        #return data
        #return np.random.choice(self.buffer, size=batch_size, replace=False)
        return rnd.sample(self.buffer, batch_size)#

# Define replay buffer
replay_buffer = ReplayBuffer(buffer_size=1000)

for episode in range(num_episodes):

    trader = StockTrader(20, Ambiente().gbm().flatten())
    done = False

    while not done:
        # Get current state
        budget, stock_count, curr_price, next_price = trader.get_state()

        # Select an action
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(model.predict(np.array([[budget, stock_count, curr_price]]))[0])

        # Take action and observe new state and reward
        trader.take_action(action)
        new_budget, new_stock_count, _, new_price = trader.get_state()
        reward = trader.get_reward()

        # Add experience to replay buffer
        replay_buffer.add((budget, stock_count, curr_price, action, reward, new_budget, new_stock_count, new_price)) #

        # Sample experiences from replay buffer
        batch = replay_buffer.sample(batch_size=32)
        states  = np.array([batch[:3]])
        actions = np.array([batch[4]])#np.array([e[3] for e in batch])
        rewards = np.array([batch[5]])#np.array([e[4] for e in batch])
        next_states = np.array([batch[5:]])#np.array([e[5:] for e in batch])

        # Update Q-value
        q_val = model.predict(states)
        q_next = target_model.predict(next_states)
        q_val = rewards + discount_factor * np.max(q_next)#[np.arange(len(batch)), actions], axis=1
        model.fit(states, q_val, epochs=1, verbose=0)

        epsilon *= epsilon_decay

        # Update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

    print(actions)