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

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
State = namedtuple("State", "inventory, time, qdr_var, price")
Action = namedtuple("Action", "amount_sold")


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

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
        q_net.add(Dense(64, activation = 'relu'  , kernel_initializer = 'he_uniform')) #Draws samples from a uniform distribution within
        q_net.add(Dense(32, activation = 'relu'  , kernel_initializer = 'he_uniform'))
        q_net.add(Dense(1 , activation = 'linear', kernel_initializer = 'he_uniform'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

        return q_net

    def step(self, curr_state, data, x, min_p, max_p, min_v, max_v):

        time_step = curr_state.time# + 1
        x = self.collect_policy(curr_state, x, min_p, max_p, min_v, max_v)
        reward = self.calc_reward(curr_state, x, data)
        price = env.returns(data)
        var = env.var(data)
        next_state = State(curr_state.inventory - x.amount_sold, time_step + 1, price[time_step], var[time_step])

        self.memory.add(curr_state, x, next_state, reward)

        if len(self.memory) < self.batch_size:

            return 1, 0, reward, next_state, self.epsilon

        else:

            transitions = self.memory.sample(self.batch_size)

        return *self.train(transitions, data), reward, next_state, self.epsilon

    def reset(self, data):

        price = Ambiente().returns(data[0:2])
        var   = Ambiente().var(price[0:2])

        return State(self.inventory, 0, 0.1 , price), Action(rnd.uniform(0, 1))#var

    def collect_policy(self, state, x, min_p, max_p, min_v, max_v):

        q = state.inventory
        t = state.time

        if q == 0:

            return Action(0)

        if self.epsilon > rnd.uniform(0, 1):

            rand_act = np.random.binomial(q, 1 / (self.passo - t))

            return Action(rand_act)
        else: 

            x = self.choose_best_action(state, x, min_p, max_p, min_v, max_v)
            
            return x
    
    def choose_best_action(self, state, x, min_p, max_p, min_v, max_v):

        q, t, price, var, x = env.normalise(state.inventory, state.time, state.price, state.qdr_var, x, min_p, max_p, min_v, max_v ) 
        state = State(q, t, price, var) 
        input = tf.convert_to_tensor(np.asarray(state).reshape(1,-1).astype('float32'))#, dtype=tf.float32
        action_q = self.q_net(input)
        action = Action(np.argmax(action_q.numpy()[0], axis=0))

        return action
    
    def calc_reward(self, state, action, data_set):
        # da cambiare, in ottica che io già divido in 5 intervalli, il reward per ogni intervallo deve essere considerato.
        # al posto di data set i+1 - dataset i io darei direttamente il price, è già il ritorno
        # x sembrerebbe essere già per ogni secondo perchè è considerato in tesp che ti da il ritorno per due intervalli consecutivi
        ##########################
        # è sbagliato forse, per ogni azione va un reward, qui fa per una sola molti reward -> cambia!!!!!!!!!!!!!!!!!!
        ##########################
        reward = 0
        M = self.time_subdivisions
        t = state.time
        q = state.inventory
        x = action.amount_sold
        a = self.a_penalty
        xs = x / M  # amount sold each second
        T_0 = self.starting_trading_time        
        inventory_left = q
        for i in range(T_0 + M * t, T_0 + M * (t + 1)):

            if i + 1 < len(data_set):
                reward += inventory_left * (data_set[i + 1] - data_set[i]) - a * (xs ** 2)
                inventory_left -= xs

        return reward

    def call_tensor_q_net(self,state_batch):
        

        return self.q_net(tensor)
    def train(self, transitition, data):

        batch = Transition(*zip(*transitition))
    # PROBLEMA : DONE E' TRUE QUANDO SONO AD INTERVALLO 5, GLIELO DEVO DIRE E PASSARE L'INFO NEI METODI 
        state_batch, action_batch, next_state_batch, reward_batch = batch #("state", "action", "next_state", "reward"), done_batch
        current_q  = self.q_net(np.array(state_batch).astype('float32'))#.numpy() # non funziona con batch - > ordinali in un tensore [64,4,1]
        #mi dice setting an array element with a sequence -> come risolvere??
        #crea una funzione che ti ridia indietro current_q
        target_q   = np.copy(current_q)
        next_q     = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)

        for i in range(state_batch.shape[0]):

            target_q_val = reward_batch[i]
            
            if not done_batch[i]: #
            
                target_q_val += reward_batch[i] + self.gamma * max_next_q[i] #gamma = 0.95
            
            target_q[i][action_batch[i]] = target_q_val
        
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)

        loss = training_history.history['loss']
        #computing the loss, it must be 
            #target = r + self.gamma * Qp_eval
            #loss = torch.mean(( target.detach() - Q_eval )**2)
        return loss


        '''
            def evaluate_Q(self, current_state, current_action, type_input, net='main_net'):

        if type_input == 'scalar':
            # print(current_state.inventory, current_state.time, current_action.amount_sold)
            q, t, qdr_var, price, x = self.normalization(current_state.inventory,
                                                         current_state.time,
                                                         current_state.qdr_var,
                                                         current_state.price,
                                                         current_action.amount_sold
                                                         )

            in_features = torch.tensor(
                [
                    q,
                    t,
                    qdr_var,
                    price,
                    x
                ],
                dtype=torch.float
            )

        elif type_input == 'tensor':
            in_features = []
            for (current_state, current_action) in zip(current_state, current_action):
                q, t, qdr_var, price, x = self.normalization(current_state.inventory,
                                                             current_state.time,
                                                             current_state.qdr_var,
                                                             current_state.price,
                                                             current_action.amount_sold
                                                             )

                in_features.append(torch.tensor(
                    [
                        q,
                        t,
                        qdr_var,
                        price,
                        x,
                    ],
                    dtype=torch.float
                )
                )

            in_features = torch.stack(in_features)
            in_features.type(torch.float64)

        if net == 'main_net':
            return self.main_net(in_features).type(torch.float64)
        elif net == 'target_net':
            return self.target_net(in_features).type(torch.float64)
        
        
        '''