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
        """
        q_net = Sequential()
        q_net.add(tf.keras.layers.Input(shape=(4,))) #4 = [inventory, time, price, stddev] -> state
        q_net.add(Dense(64, activation = 'relu'  )) #Draws samples from a uniform distribution within
        q_net.add(Dense(32, activation = 'relu'  ))
        q_net.add(Dense(1 , activation = 'linear'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

        return q_net

    #def step(self, curr_state, data, x, min_p, max_p, min_v, max_v):
    def step(self, inv, time, price, var, data, x,   min_p, max_p, min_v, max_v):    
        #q, t, p, v, x = env.normalise(curr_state.inventory, curr_state.time, curr_state.price, curr_state.qdr_var, x, min_p, max_p, min_v, max_v)
        #curr_state = State(q, t, p, v)
        time_step = time# + 1
        x = self.collect_policy(inv, time, price, var)#, x, min_p, max_p, min_v, max_v)
        reward = self.calc_reward(inv, time, price, var, x, data)
        price = env.returns(data)
        var = env.var(data)
        next_state = [inv - x, time_step + 1, price[time_step], var[time_step]]

        self.memory.add(inv, time, price[time_step], var[time_step], x, next_state, reward)

        if len(self.memory) < self.batch_size:

            return 1, 0, reward, next_state, self.epsilon

        else:

            transitions = self.memory.sample(self.batch_size)

        return *self.train(transitions, data), reward, next_state, self.epsilon

    def reset(self, data):
        ''' returns: inv, time, price, var, action '''
        price = Ambiente().returns(data[0:2])
        var   = Ambiente().var(price[0:2])

        return self.inventory, 0, 0.1 , price[0], rnd.uniform(0, 1)#var

    def collect_policy(self, inv, time, price, var):#, x, min_p, max_p, min_v, max_v):

        q = int(inv)
        t = int(time)

        if q == 0:

            return 0

        if self.epsilon > rnd.uniform(0, 1):

            rand_act = np.random.binomial(q, 1 / (self.passo - t))

            return rand_act
        else: 

            x = self.choose_best_action(inv, time, price, var)#, x, min_p, max_p, min_v, max_v)
            
            return x
    
    def choose_best_action(self, inv, time, price, var):#, x, min_p, max_p, min_v, max_v):

        #q, t, price, var, x = env.normalise(state.inventory, state.time, state.price, state.qdr_var, x, min_p, max_p, min_v, max_v ) 
        #state = State(q, t, price, var) 
        #input = tf.convert_to_tensor(np.asarray(state).reshape(1,-1).astype('float32'))#, dtype=tf.float32
        state = [[inv, time, price, var]]
        action_q = self.q_net(tf.convert_to_tensor(np.asarray(state).astype('float32')))#.reshape(1,-1) # da sempre errore qui!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        #for i in range(T_0 + M * t, T_0 + M * (t + 1)):

            #if i + 1 < len(data_set):
        reward -= inventory_left * (data_set[t]- data_set[t-1]) - a * (xs ** 2)
        inventory_left -= xs

        return reward

    def call_tensor_q_net(self, state_batch, action_batch):
    # devi riempirlo e poi mandarlo in q_net
        tensor = []#np.empty((64, 4, 1))
        #for (current_state, current_action) in zip(state_batch, action_batch):
        q, t, price, qdr_var  = state_batch[0], state_batch[1] , state_batch[2], state_batch[3]
        tensor.append([ q, t, price, qdr_var])#.astype('float32')
        #tf.cast(np.array(tensor) , dtype=tf.float32)
        tensor = np.array(tensor)
        #tensor = np.array(tensor)
            #tensor.type(tf.float64)

        #if net == 'main_net':
        #    return self.main_net(in_features).type(torch.float64)        

        return self.q_net(tensor)

    def call_tensor_q_net_tgt(self, state_batch, action_batch):
    # devi riempirlo e poi mandarlo in q_net
        tensor = []#np.empty((64, 4, 1))
        #for (current_state, current_action) in zip(state_batch, action_batch):
        q, t, price, qdr_var  = state_batch[0], state_batch[1] , state_batch[2], state_batch[3]
        tensor.append([ q, t, price, qdr_var])#.astype('float32') #AGGIUSTA Q0 CHE OGNI TANTO E INT OGNI TANTO E' ARRAY
        #tf.cast(np.array(tensor) , dtype=tf.float32)
        #tensor = np.array(tensor[0])
        tensor = np.array(tensor)
            #tensor.type(tf.float64)

        #if net == 'main_net':
        #    return self.main_net(in_features).type(torch.float64)        

        return self.target_q_net(tensor)

    def train(self, transitition, data):

        batch = transitition
    # PROBLEMA : DONE E' TRUE QUANDO SONO AD INTERVALLO 5, GLIELO DEVO DIRE E PASSARE L'INFO NEI METODI 
        state_batch = batch[0][:4] 
        action_batch = batch[0][4] 
        next_state_batch = batch[0][5] 
        #next_state_batch = [next_state_batch[0][0], next_state_batch[1], next_state_batch[2], next_state_batch[3]]
        reward_batch = batch[0][6]  #("state", "action", "next_state", "reward"), done_batch
        current_q  = self.call_tensor_q_net(state_batch, action_batch)#q_net(np.array(state_batch).astype('float32'))#.numpy() # non funziona con batch - > ordinali in un tensore [64,4,1]
        #mi dice setting an array element with a sequence -> come risolvere??
        #crea una funzione che ti ridia indietro current_q
        target_q   = np.copy(current_q)
        next_q     = self.call_tensor_q_net_tgt(next_state_batch, action_batch)#.numpy()#
        max_next_q = np.amax(next_q, axis=1)

        for i in range(10):#state_batch.shape[0]

            target_q_val = reward_batch#[i]
            
            #if not done_batch[i]: #
            
            target_q_val += reward_batch + self.gamma * max_next_q#gamma = 0.95[i][i] 
            
            target_q= target_q_val#[i][i][action_batch] 
        
        training_history = self.q_net.fit(x=np.array(state_batch).transpose(), y=target_q, verbose=0) #->> riempili per batch size e ci siamo su sta parte, 
        #devono essere x=(64,4,1) e y =(64,1)

        # c'è da riempire il batch con 64 episodi di 0 e via via poi ci siamo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        loss = training_history.history['loss']
        #computing the loss, it must be 
            #target = r + self.gamma * Qp_eval
            #loss = torch.mean(( target.detach() - Q_eval )**2)
        return loss