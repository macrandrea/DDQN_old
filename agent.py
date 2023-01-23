import numpy as np
import math as m
import statistics as stat
import tensorflow as tf
import random as rnd

from tensorflow import keras
from keras import Sequential, layers
from keras.layers import Dense
#from tensorflow.keras import layers

from collections import namedtuple, deque
from Ambiente import Ambiente
layer = layers.Normalization()

env = Ambiente()

initializer = tf.keras.initializers.HeUniform()

q_net = Sequential()
q_net.add(tf.keras.layers.Input(shape=(5,))) #4 = [inventory, time, price, stddev, prev_action, state]
q_net.add(Dense(20, activation = 'relu'  , kernel_initializer = initializer)) 
q_net.add(Dense(20, activation = 'relu'  ))
q_net.add(Dense(20, activation = 'relu'  )) 
q_net.add(Dense(20, activation = 'relu'  ))
q_net.add(Dense(20, activation = 'relu'  ))      
q_net.add(Dense(1 , activation = 'linear')) # esce un vettore di q values per ogni azione 
q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

target_q_net = Sequential()
target_q_net.add(tf.keras.layers.Input(shape=(5,))) #4 = [inventory, time, price, stddev, prev_action, state]
target_q_net.add(Dense(20, activation = 'relu' , kernel_initializer = initializer) )
target_q_net.add(Dense(20, activation = 'relu'  ))
target_q_net.add(Dense(20, activation = 'relu'  )) 
target_q_net.add(Dense(20, activation = 'relu'  ))
target_q_net.add(Dense(20, activation = 'relu'  ))      
target_q_net.add(Dense(1 , activation = 'linear')) # esce un vettore di q values per ogni azione 
target_q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

target_q_net.set_weights(q_net.get_weights())

class ReplayMemory():

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    def add(self,inv, time, price, var, x, next_inv, next_time, next_price, next_var, x_new, reward): #inv, time, price, var, x, next_state, reward state, action, next_state, reward
        
        self.memory.append([inv, time, price, var, x, next_inv, next_time, next_price, next_var, x_new, reward])

    def sample(self, batch_size):
        
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        
        return len(self.memory)

class Agente():

    def __init__(self, numTrain):

        #self.q_net = self._build_dqn_model()
        #self.target_q_net = self._build_dqn_model()
        self.time_subdivisions = 5
        self.inventory = 20
        self.a_penalty = 0.0001
        self.epsilon = 0.5
        self.epsilon_decay = 0.95
        self.passo = 1200
        self.batch_size = 128
        self.gamma = 0.95
        self.memory = ReplayMemory(6200)
        self.numTrain = numTrain

    def action(self, state, x):

        if state[0] <= 0:
            return 0
        # elif i == 5:
        #     return state[0]
        elif np.random.binomial(1,0.5) <= self.epsilon:
            rand_act = rnd.randrange(0,self.inventory)#np.random.rand()#- tstate[0], 1 / (self.passo)
            self.epsilon *= self.epsilon_decay
            #return rnd.randrange(self.inventory)
            return rand_act
        else:
            state.append(x)
            action = np.argmax(q_net(np.array([state]).astype('float32')))#[0], verbose = 0

        return action
        #if np.random.rand() <= self.epsilon:
        #return random.randrange(self.action_size)

    def rew(self, state, action, reward,p_0):

        #reward = 0 #mh
        if state[0] == 0:
            reward == 0
            state[0] = 0
            return reward
        else:
            xs = action
            reward += state[0] * (state[2]) - self.a_penalty * (xs ** 2)
            tot_rew = -self.inventory*p_0 + reward
            return tot_rew

    def reset(self, data):

        price = Ambiente().returns(data[0:2])
        var   = Ambiente().var(price[0:2])
        action =  rnd.uniform(0, 1)

        return self.inventory, 0, price[0], var, action #rnd.uniform(0, 1)

    def step(self, inv, time, price, var,  x, i, p_0):
        #self.memory = ReplayMemory(128)
        reward = 0
        state = [inv, time-1 , price[time-1], var[time-1]]
        x_new = self.action(state, x)
        reward = self.rew(state, x_new, reward, p_0)
        next_state = [inv - x_new, time , price[time], var[time]]
        self.memory.add(inv, time-1, price[time-1], var[time-1], x, inv - x_new, time, price[time], var[time], x_new, reward)

        if len(self.memory) < self.batch_size:

            return 1, reward, next_state, x_new, self.epsilon#0,

        else:

            transitions = self.memory.sample(self.batch_size)
            loss, grad = self.train(transitions, i)

            return loss, reward, next_state, x_new, self.epsilon

    def train(self, trans, i):

        batch = np.asarray(trans).astype('float32')
        layer.adapt(batch)
        batch_t = layer(batch)
        #batch_t = scaler.fit_transform(batch) #normalizzazione
        state_act_batch      = batch_t[:,:5]
        next_state_act_batch = batch_t[:, 5: 10]
        reward_batch     = batch_t[:, 10]
        #normalizza
        #
        ##
        q_val = q_net.predict(state_act_batch, verbose = 0)#self.q_net.predict(state_batch)
        q_next = target_q_net.predict(next_state_act_batch, verbose = 0)#self.target_q_net(next_state_batch)
        if i < 4:
            q_val = reward_batch + self.gamma * np.max(q_next)

            training = q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        elif i == 4:
            correction_term = (next_state_act_batch[0][0]) * (next_state_act_batch[0][2]) - self.a_penalty * ((next_state_act_batch[0][0]) ** 2)

            q_val = reward_batch + self.gamma * correction_term

            training = q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        else:
            q_val = reward_batch

            training = q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        grad_norm = 0
        #update della rete q

        if state_act_batch[1][0]%10 == 0:
            #q_next = q_val
            target_q_net.set_weights(q_net.get_weights())

        return loss, grad_norm

    