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
delayer = layers.Normalization(axis=-1, invert =True)
layer_one = layers.Normalization()

env = Ambiente()

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

        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()
        self.time_subdivisions = 5
        self.inventory = 20
        self.a_penalty = 0.0001
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.passo = 1200
        self.batch_size = 256
        self.gamma = 0.95
        self.memory = ReplayMemory(6200)
        self.numTrain = numTrain
    
    @staticmethod
    def _build_dqn_model():
        # rete neurale feed-forward
        q_net = Sequential()
        q_net.add(tf.keras.layers.Input(shape=(5,))) #4 = [inventory, time, price, stddev, prev_action, state]
        q_net.add(Dense(20*4, activation = 'relu'  )) 
        q_net.add(Dense(20*2, activation = 'relu'  ))
        q_net.add(Dense(20, activation = 'relu'  )) 
        q_net.add(Dense(20*0.5, activation = 'relu'  ))
        q_net.add(Dense(20*0.25, activation = 'relu'  ))      
        q_net.add(Dense(1 , activation = 'linear')) # esce un vettore di q values per ogni azione possibile
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

        return q_net # la devo ri-compilare ogni volta che cambia lo stat? non avrebbe molto sensoi credo

    def action(self, state, x):
        # azione da eseguire: estrae un numero a caso: se questo è minore di epsilon allora fa azione casuale x=(0,q_t), altrimenti fa argmax_a(Q(s,a))
        if state[0] <= 0:
            return 0
        # elif i == 5:
        #     return state[0]
        elif np.random.rand() <= self.epsilon:
            rand_act = np.random.binomial(np.floor(state[0]), 1 / (self.time_subdivisions))#- t
            self.epsilon *= self.epsilon_decay
            #return rnd.randrange(np.floor(state[0]))
            return rand_act
        else:
            state.append(x)
            state = np.array([state])
            layer.adapt(state)
            state = layer(state)
            action = self.q_net.predict(state, 20, verbose=0)[0]
            action = np.argmax(delayer(action))
        return action
        #if np.random.rand() <= self.epsilon:
        #return random.randrange(self.action_size)

    def rew(self, state, action, reward,p_0):
        # calcola il reward della vendita di una quantità z di azioni data dal metodo action()
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
        #resetta lo stato della simulazione
        price = Ambiente().returns(data[0:2])
        var   = Ambiente().var(price[0:2])
        action =  rnd.uniform(0, 1)

        return self.inventory, 0, price[0], var, action #rnd.uniform(0, 1)

    def step(self, inv, time, price, var,  x, i, p_0):
        #considera lo stato -> compie un'azione dato lo stato -> calcola il rewadr di quell'azione -> vede il prossimo stato 
        # -> aggiunge questo stato al buffer di replay, altrimenti se il buffer è pieno aggiorna i pesi della rete neurale facendo il fit di questa con 
        #gli elementi presenti in memoria -> i pesi della rete neurale si aggiornano e qando chiamo l'azione , questa esce con dei pesi nuovi
        #self.memory = ReplayMemory(128)
        reward = 0
        state = [inv, time - 1 , price[time - 1], var[time - 1]]
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
        # fa il train della rete neurale: ovvero aggiorna i pesi della rete da usare quando si compie l'azione argmax_a(Q(s,a)) 
        # nota bene che se siamo nel penultimo periodo ha una reward function diversa da quella solita, se siamo nell'ultimo periodo è differente perchè 
        # deve liquidare tutto quello che ha in termini di inventario.
        batch = np.asarray(trans).astype('float32')
        layer_one.adapt(batch)
        batch_t = layer_one(batch)# normalizza 
        #batch_t = scaler.fit_transform(batch) #normalizzazione
        state_act_batch      = np.array([tup[:5] for tup in batch_t])#batch_t[:,:5]
        next_state_act_batch = np.array([tup[5:10] for tup in batch_t])#batch_t[:, 5: 10]
        reward_batch     = np.array([tup[10] for tup in batch_t])#batch_t[:, 10]
        
        ##
        #q_val = self.q_net.predict(state_act_batch,verbose  = 0)#self.q_net.predict(state_batch)
        ##self.target_q_net.predict(next_state_batch)
        if i < 4:
            q_next = self.target_q_net.predict(next_state_act_batch, verbose = 0)
            q_val = reward_batch + self.gamma * np.amax(q_next)
            
            training = self.q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        elif i == 4:
            correction_term = (next_state_act_batch[0][0]) * (next_state_act_batch[0][2]) - self.a_penalty * ((next_state_act_batch[0][0]) ** 2)

            q_val = reward_batch + self.gamma * correction_term

            training = self.q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        else:
            q_val = reward_batch

            training = self.q_net.fit(state_act_batch, q_val, epochs=self.numTrain, verbose=0)
            loss = training.history['loss']
        grad_norm = 0

        #update della rete q
        if state_act_batch[1][0]%5 == 0:
            q_next = q_val

        return loss, grad_norm

        ###################################################################################
        # non funziona la scelta dell'azione -> sceglie sempre azione = 0, - SEMBRA RISOLTO MA NON SEMPRE E' OTTIMALE LA SUA SCELTA
        # forse problemi con l'esplorazione dello spazio delle azioni e degli stati
        # non capisco come delimitare lo spazio delle azioni che escono fuori dalla rete neurale x_t \in (0,q_t) 
        # e come delimitare quello delle azioni epsilon-greedy 
        # in più il codice è lento e vorrei riuscire a farlo andare più veloce 
        # inoltre non sono convinto della rete neurale adoperata nel paper di riferimento
        ###################################################################################