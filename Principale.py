from Agente import Agente
from Agente import ReplayMemory
#from agent import Agente
#from agent import ReplayMemory
import random as rnd
from Ambiente import Ambiente
import numpy as np
import math as m
from tqdm import tqdm
from collections import namedtuple, deque

numItTrain = 1
numSlice = 5
numTraj = 2
inventory = 20
reward_history = []
epsilon_history = []
grad_norm_history = []
loss_history = []
action_history = []

#env = Ambiente()
#age = Agente(numItTrain)
#mem = ReplayMemory(500)

for i in tqdm(range(numTraj)): # loop su quante traiettorie montecarlo considerate
    env = Ambiente()
    age = Agente(numItTrain)
    dati = env.gen_paths(1).flatten()
    ret = env.returns(dati)
    v = env.var(ret)
    inv, time, price, var, x = inventory, 0, ret[0], v[0], rnd.uniform(0, 1)  #reset
    reward_episode = 0
    time = 0
    counter = 0


    for ii in tqdm(range(int(len(dati)/numSlice),len(dati),int(len(dati)/numSlice))): # divide in 5 steps e massimizza la Q per ognuno di questi 5 intervalli

        ritorni = ret[:ii]
        vola = v[:ii]
        counter +=1

        for iii in tqdm(range(1, int(len(dati)/numSlice))): #per ogni singolo time-step vede lo stato -> fa un'azione -> vede reward -> setta il prossimo stato
        
            (loss, reward, next_state, x_new, epsilon) = age.step(inv, iii, ritorni, vola, x, counter, ret[0])
            inv, time, x = next_state[0], next_state[1], x_new
            reward_episode += reward

        loss_history.append(loss)
        epsilon_history.append(epsilon)
        reward_history.append(reward_episode/int(len(dati)/numSlice))
        action_history.append(x)

print(reward_history, action_history, epsilon_history, loss_history)    