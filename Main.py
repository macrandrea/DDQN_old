from Trader import Agente
#from Agente_pytorch import Agente
from Trader import ReplayMemory
from Ambiente import Ambiente
import numpy as np
import math as m
from tqdm import tqdm
from collections import namedtuple, deque


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
State = namedtuple("State", "inventory, time, qdr_var, price")
Action = namedtuple("Action", "amount_sold")

env = Ambiente()
age = Agente()
mem = ReplayMemory(500)

def sliceData(price, slici):

    step = int(len(price)/slici)
    y = np.zeros((slici,step))

    for i, ii in zip(range(slici), range(step, len(price), step)):
        it = step * i
        y[i, :] = price[it:ii]

    return y

def matriciIntervalli(data,numSlice):

    dati = sliceData(data, numSlice)
    price= np.empty((numSlice, int(len(data) /numSlice) -1))
    var  = np.empty((numSlice, int(len(data)/numSlice) -1))

    for i in range(dati.shape[0]): # divido per numSlice intervalli
        price[i,:] = env.returns(dati[i,:])
        var  [i,:] = env.var(dati[i,:])

    return price, var, dati

#def main():
#    agent = Agente()
#    #cicli
#    #chiama step
#    #salva le loss ecc
#
#    reward_history, epsilon_history, grad_norm_history, loss_history, qdr_var_history, price_history = doTrain(agent)
#    return reward_history, epsilon_history, grad_norm_history, loss_history, qdr_var_history, price_history

#if __name__ == "main":
numItTrain = 1
numSlice = 6
numTraj = 10
reward_history = []
epsilon_history = []
grad_norm_history = []
loss_history = []

data = env.gbm().flatten()
pri, v, datiSli = matriciIntervalli(data, numSlice=numSlice)
state = age.reset(datiSli)

for i in tqdm(range(numSlice - 1)):

    inv, time, price, var, action = age.reset(datiSli[i, :])
    min_p, max_p, min_v, max_v = pri[i, :].min(), pri[i, :].max(), v[i, :].min(), v[i, :].max()
    reward_episode = 0

    for ii in tqdm(range(numItTrain)):

        for iii in tqdm(range(len(datiSli[i, :]))):

            (loss, grad_norm, reward, next_state, epsilon) = age.step(inv, time, price, var, datiSli[i, :], action, min_p, max_p, min_v, max_v)#
            current_state = next_state
            # modifica anche la funzione step -> gli devi far capire che siamo in 5, e se siamo in 6 deve assolutamente liquidare tutto quanto!

            reward_episode += reward
        loss_history.append(loss)
        epsilon_history.append(epsilon)
        reward_history.append(reward_episode)
        #grad_norm_history.append(grad_norm)m.log(loss)

print(reward_history, loss_history)
'''    

for t in range(numTraj):# ciclo di quante traiettorie vuoi considerare
    data = Ambiente().gbm().flatten()
    dati = sliceData(data, numSlice)
    price= np.empty((numSlice, int(len(data) /numSlice) -1))
    var  = np.empty((numSlice, int(len(data)/numSlice) -1))
    for i in range(dati.shape[0]): # divido per 5 intervalli
        price[i,:] = Ambiente().returns(dati[i,:])
        var  [i,:] = Ambiente().var(dati[i,:])
        #for j in range(numItTrain): # epochs di train




'''
    #price = Environment().returns(data)
    #var   = Environment().var(data)
    #for i in range(data.shape[1]):# per ogni traiettoria MC
    #    state, x = Agente().reset()
    #    rewards = []
    #    for ii in range(numItTrain):# numero di volte in cui fai il train: epochs
    #        # questo va fatto sui 5 sotto insiemi di data
    #        loss, reward, next_state, epsilon = Agente().step(state, data)
    #        state = next_state
    #        rewards.append(reward)
# capire come spezzare in 5 intervalli i dati
    #for i in range()

    #reward_history, epsilon_history, grad_norm_history, loss_history, qdr_var_history, price_history = main()


#stock_dict = {"train_drift": 6000}
#stock_dict_pre_training = {"train_drift": 6000}
#for stock in tqdm(stock_dict.keys()):
#    for i in tqdm(range(stock_dict[stock])):
#        print('a')