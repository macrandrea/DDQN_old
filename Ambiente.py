import numpy as np
import math as m
import statistics
from collections import namedtuple, deque

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
State = namedtuple("State", "inventory, time, qdr_var, price")
Action = namedtuple("Action", "amount_sold")

class Ambiente():

    def __init__(self, S0 = 100, mu = 0.01, kappa = 5, theta = 1, sigma = 0.1, lambd = 0.1, t0 = 0, t = 1, T = 7200, numIt = 1_000): #T = 1.0, M = 7200, I = 1_000
        
        self.S0 = S0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = 1/T
        self.T = T
        self.t0 = t0
        self.tau = t-t0
        self.lambd = lambd
        self.numIt = numIt
        self.initial_capital = 20

    def gbm(self, num_states = 1): # da fare n colonne
        np.random.seed(457778)
        M = self.T
        T = self.tau
        I = num_states
        dt = float(T) / self.T
        paths = np.zeros((M + 1, I), np.float64)
        paths[0] = self.S0
        for t in range(1, M + 1):
            rand = np.random.standard_normal(I)
            paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * rand)
        return paths

    def gen_paths(self, I):#S0, r, sigma, T, M, I):#rand = (rand - rand.mean()) / rand.std()
        T = 1.0
        M = 200 # 1/M sono gli steps che fa
        dt = float(T) / M
        paths = np.zeros((M + 1, I), np.float64)
        paths[0] = self.S0
        for t in range(1, M + 1):
            rand = np.random.standard_normal(I)
            paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt +
                                             self.sigma * np.sqrt(dt) * rand)
        return paths

    def returns(self, s):
        price = [] 
        for t in range(1,len(s)):
            price.append(s[t] - s[t - 1])
        return np.array(price)

    def var(self, s):
        var = [] 
        for t in range(1,len(s)):
            var.append((s[t] - s[t - 1]) ** 2)
        return np.array(var)

    def inventory_action_transform(self, q, x):

        q_0 = self.initial_capital + 1

        q = q / q_0 - 1
        x = x / q_0
        r = m.sqrt(q ** 2 + x ** 2)
        theta = m.atan((-x / q))
        z = -x / q

        if theta <= m.pi / 4:
            r_tilde = r * m.sqrt((pow(z, 2) + 1) * (2 * (m.cos(m.pi / 4 - theta)) ** 2))
        else:
            r_tilde = r * m.sqrt(
                (pow(z, -2) + 1) * (2 * (m.cos(theta - m.pi / 4)) ** 2)
            )
        return 2 * (-r_tilde * m.cos(theta)) + 1, 2 * (r_tilde * m.sin(theta)) - 1

    def time_transform(self, t):

        tc = (5 - 1) / 2
        return (t - tc) / tc

    def qdr_var_normalize(self, qdr_var, min_v, max_v):

        middle_point = (max_v + min_v) / 2
        half_length = (max_v - min_v) / 2

        qdr_var = (qdr_var - middle_point) / half_length

        return qdr_var

    def price_normalise(self, price, min_p, max_p):

        middle_point = (max_p + min_p) / 2
        half_length = (max_p - min_p) / 2

        price = (price - middle_point) / half_length

        return price

    def normalise(self, inventory, time, price, var, x, min_p, max_p, min_v, max_v):
        # dagli per bene il tempo (quindi l'iteratore i) e i max e min per normalizzare il prezzo
        q, x = self.inventory_action_transform(inventory, x)
        t = self.time_transform(time)#
        p = self.price_normalise(price, min_p, max_p)
        sig = self.qdr_var_normalize(var, min_v, max_v)
        return q, t, p, sig, x