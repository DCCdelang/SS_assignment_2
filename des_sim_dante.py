"""
DES program to simulate M/M/n queue's
"""

import random
import simpy
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

LAMBDA = 10.0  # Generate new customers roughly every lambda seconds
MU = 9 # Expected waiting time per customer in seconds
N_CUSTOMERS = 10000
N_servers = [1,2,4]
N_sim = 1000

class System(object):
    """Class for one server queue system"""
    def __init__(self, env, n_server, n_cust, mu, lambd):
        self.env = env
        self.server = simpy.Resource(env, capacity=n_server)
        self.waittime = 0
        self.waitlist = []
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

        # Time in system
        tis = random.expovariate(1/system.mu)
        yield env.timeout(tis)
        
        # Sojourn time, real waiting time
        wait = env.now - arrive

        # Append only steady state values of waiting time > x customers
        if system.total_cust > 500:
            system.waittime += wait
        system.waitlist.append(wait)

def setup(env,system):
    """Adding customers for one system simulations"""
    for _ in range(system.n_cust):
        system.total_cust += 1
        env.process(customer(env, system)) # Add customer to process
        t = random.expovariate(1/system.lambd)
        yield env.timeout(t)

t0 = time.time()

data_list = [[],[]]
sum_waiting = [[],[],[]]

# Running all simulations for every n_server
for i in range(len(N_servers)):
    print("Server", i)
    for n_sim in range(N_sim):
        # Choose random random seed
        RANDOM_SEED = random.randint(1,100000)
        random.seed(RANDOM_SEED)

        # Setup and start the simulation
        env = simpy.Environment()
        system = System(env, N_servers[i], N_CUSTOMERS, MU, LAMBDA/N_servers[i])

        env.process(setup(env, system))
        env.run()

        # Filling large data_list with mean waiting time per simulation
        # will be used for csv (Could be optimized)
        data_list[0].append(i)
        data_list[1].append(system.waittime/system.total_cust)

        # Collecting all customer waiting times to determine steady state
        if n_sim == 0:
            # Adding first list with waiting times per customer
            sum_waiting[i] = system.waitlist 
        else:
            # Continue adding sum_waiting list
            before = sum_waiting[i]
            sum_waiting[i] = np.add(before, system.waitlist)
            
    print("Finished all", N_sim, "simulations")

# Creating a csv with the mean waiting time per customer for all servers
mean_waiting_pc = []
serverlist = []
for i in range(len(N_servers)):
    sublist = list(np.array(sum_waiting[i])/N_sim)
    mean_waiting_pc.extend(sublist)
    for c in range(N_CUSTOMERS):
        serverlist.append(i)

sum_data = {"Server":serverlist, "Waiting pc": mean_waiting_pc}
df3 = pd.DataFrame(sum_data)
df3.to_csv("waiting_pc.csv")

# Data file with all mean waiting times per simulation
data = {"Server": data_list[0], "Mean Wait": data_list[1]}
df = pd.DataFrame(data)
df.to_csv("data.csv")

t1 = time.time()
print("\nDid", N_sim, "simulations with", N_CUSTOMERS, "customers, for server", N_servers, "in total time of ", round(t1-t0,3), "seconds\n")

