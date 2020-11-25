"""
DES program to simulate M/M/n queue's
"""

import random
import simpy
import pandas as pd
import numpy as np


LAMBDA = 10.0  # Generate new customers roughly every lambda seconds
MU = 9 # Expected waiting time per customer in seconds
N_CUSTOMERS = 1000

class System(object):
    """Class for one server queue system"""
    def __init__(self, env, n_server, n_cust, mu, lambd):
        self.env = env
        self.server = simpy.Resource(env, capacity=n_server)
        self.waittime = 0
        self.sojourn = 0
        self.total_cust = 0
        self.mu = mu
        self.lambd = lambd
        self.n_cust = n_cust

def customer(env, system):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    with system.server.request() as req:
        yield req 

        wait = env.now - arrive
        system.waittime += wait

        # Time in system
        tis = random.expovariate(1/system.mu)
        yield env.timeout(tis)
        
        sojourn = env.now - arrive
        system.sojourn += sojourn

def setup(env,system):
    """Adding customers for one system simulations"""
    for c in range(system.n_cust):
        system.total_cust += 1
        env.process(customer(env, system)) # Add customer to process
        t = random.expovariate(1/system.lambd)
        yield env.timeout(t)

N_servers = [1,2,4]
N_sim = 100
data_list = [[],[]]

# For loop to run all simulations
for i in range(len(N_servers)):
    for n_sim in range(N_sim):

        # Setup and start the simulation
        RANDOM_SEED = random.randint(1,700)
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        system = System(env, N_servers[i], N_CUSTOMERS, MU, LAMBDA/N_servers[i])

        env.process(setup(env, system))
        env.run()

        data_list[0].append(i)
        data_list[1].append(system.waittime/system.total_cust)

data = {"Server": data_list[0], "Mean Wait": data_list[1]}
df = pd.DataFrame(data)
df.to_csv("data.csv")

print("Done!")