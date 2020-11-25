"""
DES program to simulate M/M/n queue's
"""

import random
import simpy
import csv
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
import numpy as np


LAMBDA = 3.0  # Generate new customers roughly every x seconds
MU = 10 # Total waiting time???? 

class System(object):
    """Class for whole server queue system"""
    def __init__(self, env, n_server, mu, lambd):
        self.env = env
        self.server = simpy.Resource(env, capacity=n_server)
        self.waittime = 0
        self.sojourn = 0
        self.total_cust = 0
        self.mu = mu
        self.lambd = lambd

def customer(env, system):
    """Customer arrives, is served and leaves."""
    arrive = env.now

    with system.server.request() as req:
        yield req 

        wait = env.now - arrive
        system.waittime += wait

        # Time in system
        tis = random.expovariate(1.0 / system.mu)
        yield env.timeout(tis)
        
        sojourn = env.now - arrive
        system.sojourn += sojourn

def setup(env,system):
    while True:
        system.total_cust += 1
        env.process(customer(env, system)) # Add customer to process
        t = random.expovariate(1.0 / system.lambd)
        yield env.timeout(t)

N_servers = [1,2,4]
N_sim = 10

cust_list = [[],[],[]]
wait_list = [[],[],[]]
mean_wait_list = [[],[],[]]

# data = open('data.csv','w')
for i in range(len(N_servers)):
    for n_sim in range(N_sim):
        # Setup and start the simulation
        RANDOM_SEED = random.randint(1,700)
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        system = System(env, N_servers[i], MU, LAMBDA/N_servers[i])

        env.process(setup(env, system))
        env.run(until=1000)
        # data.write(str(system.waittime) + "\n")
        cust_list[i].append(system.total_cust)
        wait_list[i].append(system.waittime)
        mean_wait_list[i].append(system.waittime/system.total_cust)

print("N_cust:",cust_list)
print("Total wait:",wait_list)
print("Mean wait:", mean_wait_list)

for i in range(len(N_servers)):
    print(np.mean(cust_list[i]))
    print(np.mean(wait_list[i]))
    print(np.mean(mean_wait_list[i]))
