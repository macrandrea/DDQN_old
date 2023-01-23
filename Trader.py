import numpy as np
import math as m
import statistics as stat
import tensorflow as tf
import random as rnd

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

from collections import namedtuple, deque
from Ambiente import Ambiente

env = Ambiente()

#Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
#State = namedtuple("State", "inventory, time, qdr_var, price")
#Action = namedtuple("Action", "amount_sold")


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self,inv, time, price, var, x, next_state, reward): #inv, time, price, var, x, next_state, reward state, action, next_state, reward
        self.memory.append([inv, time, price, var, x, next_state, reward])

    def sample(self, batch_size):
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agente():

    def __init__(self):
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()
        self.time_subdivisions = 5
        self.starting_trading_time = 0
        self.time = 0
        self.inventory = 20
        self.amount_sold = 0
        self.a_penalty = 0.0001
        self.epsilon = 0.5
        self.passo = 1200
        self.batch_size = 64
        self.gamma = 0.95
        self.memory = ReplayMemory(500)

    @staticmethod
    def _build_dqn_model():
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.

        :return: Q network
        
        q_net = Sequential()
        q_net.add(tf.keras.layers.Input(shape=(4,))) #4 = [inventory, time, price, stddev, prev_action] -> state -> diventano 5
        q_net.add(Dense(20, activation = 'relu'  )) 
        q_net.add(Dense(20, activation = 'relu'  ))
        q_net.add(Dense(20, activation = 'relu'  )) 
        q_net.add(Dense(20, activation = 'relu'  ))
        q_net.add(Dense(20, activation = 'relu'  ))      
        q_net.add(Dense(1 , activation = 'linear'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')"""
        q_net = Sequential()
        q_net.add(tf.keras.layers.Input(shape=(5,))) #4 = [inventory, time, price, stddev] -> state
        q_net.add(Dense(64, activation = 'relu'  )) #Draws samples from a uniform distribution within
        q_net.add(Dense(32, activation = 'relu'  ))
        q_net.add(Dense(1 , activation = 'linear'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')
        return q_net

    #def step(self, curr_state, data, x, min_p, max_p, min_v, max_v):
    def step(self, inv, time, price, var, data, x,   min_p, max_p, min_v, max_v):    
        #q, t, p, v, x = env.normalise(curr_state.inventory, curr_state.time, curr_state.price, curr_state.qdr_var, x, min_p, max_p, min_v, max_v)
        time_step = time# + 1
        x = self.collect_policy(inv, time, price, var)
        reward = self.calc_reward(inv, time, price, var, x, data)
        price = env.returns(data)
        var = env.var(data)
        next_state = [inv - x, time_step + 1, price[time_step], var[time_step]]

        self.memory.add(inv, time, price[time_step], var[time_step], x, next_state, reward)

        if len(self.memory) < self.batch_size:

            return 1, 0, reward, next_state, self.epsilon

        else:

            transitions = self.memory.sample(self.batch_size)
            loss, grad = self.train(transitions, data)

            return loss, grad, reward, next_state, self.epsilon

    def reset(self, data):
        ''' returns: inv, time, price, var, action '''
        price = Ambiente().returns(data[0:2])
        var   = Ambiente().var(price[0:2])

        return self.inventory, 0, 0.1 , price[0], rnd.uniform(0, 1)#var

    def collect_policy(self, inv, time, price, var):

        q = int(inv)
        t = int(time)

        if q == 0:

            return 0

        if self.epsilon > rnd.uniform(0, 1):

            rand_act = np.random.binomial(q, 1 / (self.time_subdivisions ))#- t
            self.epsilon *= 0.
            return rand_act
        else: 

            x = self.choose_best_action(inv, time, price, var)
            
            return x
    
    def choose_best_action(self, inv, time, price, var):
        #q, t, price, var, x = env.normalise(state.inventory, state.time, state.price, state.qdr_var, x, min_p, max_p, min_v, max_v ) 
        state = [[inv, time, price, var]]
        action_q = self.q_net(tf.convert_to_tensor(np.asarray(state).astype('float32')))
        action = np.argmax(action_q, axis=0)

        return action[0]
    
    def calc_reward(self, inv, time, price, var, action, data_set):
        # da cambiare, in ottica che io già divido in 5 intervalli, il reward per ogni intervallo deve essere considerato.
        # al posto di data set i+1 - dataset i io darei direttamente il price, è già il ritorno
        # x sembrerebbe essere già per ogni secondo perchè è considerato in tesp che ti da il ritorno per due intervalli consecutivi
        ##########################
        # è sbagliato forse, per ogni azione va un reward, qui fa per una sola molti reward -> cambia!!!!!!!!!!!!!!!!!!
        ##########################
        reward = 0
        M = self.time_subdivisions
        t = time
        q = inv
        x = action
        a = self.a_penalty
        xs = x / M  # amount sold each second
        T_0 = self.starting_trading_time        
        inventory_left = q
        reward -= inventory_left * (data_set[t]- data_set[t-1]) - a * (xs ** 2)
        inventory_left -= xs

        return reward

    def call_tensor_q_net(self, state_batch, action_batch):
    # devi riempirlo e poi mandarlo in q_net
        tensor = []
        q, t, price, qdr_var  = state_batch[0], state_batch[1] , state_batch[2], state_batch[3]
        tensor.append([ q, t, price, qdr_var])
        tensor = np.array(tensor)     
        return self.q_net(tensor)

    def call_tensor_q_net_tgt(self, state_batch, action_batch):
    # devi riempirlo e poi mandarlo in q_net
        tensor = []
        q, t, price, qdr_var  = state_batch[0], state_batch[1] , state_batch[2], state_batch[3]
        tensor.append([ q, t, price, qdr_var])
        tensor = np.array(tensor)
        return self.target_q_net(tensor)






    def train(self, transitition, data):

        batch = np.array(transitition)
        state_batch      = batch[:,:4]
        action_batch     = batch[:, 4]
        next_state_batch = batch[:, 5]
        reward_batch     = batch[:, 6]
        q_val = self.call_tensor_q_net(state_batch.tolist(), action_batch)#self.q_net.predict(state_batch)
        q_next = self.call_tensor_q_net_tgt(next_state_batch.tolist(), action_batch)#self.target_q_net.predict(next_state_batch)
        q_val = reward_batch + self.gamma * np.max(q_next)
        training = self.q_net.fit(state_batch.tolist(), q_val.tolist(), epochs=10, verbose=0)
        loss = training.history['loss']
        grad_norm = 0
        #update della rete q
        return loss, grad_norm
