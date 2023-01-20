from Agente import Agente
from Agente import ReplayMemory
import random as rnd
#from Trader import Agente
#from Trader import ReplayMemory
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
#state = age.reset(datiSli)
action = rnd.uniform(0, 1)

for i in tqdm(range(numSlice - 1)):# quanti periodi dividi la giornata, in questo caso 5

    inv, time, price, var, action = age.reset(datiSli[i, :], action, i) # questo pezzo va messo fuori dai loop
    min_p, max_p, min_v, max_v = pri[i, :].min(), pri[i, :].max(), v[i, :].min(), v[i, :].max()
    reward_episode = 0

    for ii in tqdm(range(numItTrain)):# quante volte vuoi fare il train

        for iii in tqdm(range(2 ,(len(datiSli[i, :])) + 1)): # per ogni 1200 time steps fai questo

            (loss, reward, next_state, x_new, epsilon) = age.step(inv, iii-2, pri[i], v[i], action, i)#datiSli[i, :], min_p, max_p, min_v, max_v
            inv, time, action = next_state[0], next_state[1], x_new
            current_state = next_state
            # modifica anche la funzione step -> gli devi far capire che siamo in 5, e se siamo in 6 deve assolutamente liquidare tutto quanto!

            reward_episode += reward
        loss_history.append(loss)
        epsilon_history.append(epsilon)
        reward_history.append(reward_episode/1200)
        #grad_norm_history.append(grad_norm)m.log(loss)

print(reward_history, loss_history)
