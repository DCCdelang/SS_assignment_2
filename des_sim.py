"""
DES program to simulate M/M/n, M/D/n and M/H/n queue's
Authors: 
Dante de Lang (dccdelang@gmail.com)
Karim Semin (karimsemin@gmail.com)
"""

import random
import simpy
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

LAMBDA = 1.0  # Generate new customers roughly every lambda seconds
MU = .92 # Expected waiting time per customer in seconds
N_CUSTOMERS = 10000
N_servers = [1,2,4]
N_sim = 1000

# Shortest Job First Scheduling (boolean)
SJF = False

# Service time distribution (0,1,2) : (M/M/n, M/D/n, M/H/n)
# Important to check .csv file names for overwriting
DIST = 1

'''
todo:
So if we want our data to be all close enough to the theoretical mean. 
It would be nice if you can first investigate a bit about how many numbers of 
customers (or total time of your simulation) you would need in the simulation 
at different rho (workload) rather than setting a constant value for all 
different rho situations. 
'''

class System(object):
    """Class for one server queue system"""
    def __init__(self, env, n_server, n_cust, mu, lambd, SJF, DIST):
        self.env = env
        self.server = simpy.Resource(env, capacity=n_server)
        self.server_sjf = simpy.resources.resource.PriorityResource(env, \
        capacity=n_server)
        self.SJF = SJF
        self.DIST = DIST
        self.waittime = 0
        self.waitlist = []
        self.sojourn = 0
        self.total_cust = 0
        self.mu = mu
        self.lambd = lambd
        self.n_cust = n_cust

def long_tail():
    """ Function to create a long-tail distribution """
    randint = random.randint(0,3)
    # Values are based on a mean mu of 0.92
    if randint != 0:
        tis = random.expovariate(1/0.68)
    else:
        tis =random.expovariate(1)
    return tis

def customer(env, system):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    # Time in system, depending on distribution
    if system.DIST == 0:
        tis = random.expovariate(1/system.mu)
    elif system.DIST == 1:
        tis = 1/system.mu
    elif system.DIST == 2:
        tis = long_tail()

    if system.SJF == False:
        request = system.server.request()
    elif system.SJF == True:
        request = system.server_sjf.request(priority=tis)

    with request as req:
        yield req 
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
        system = System(env, N_servers[i], N_CUSTOMERS, MU, \
                        LAMBDA/N_servers[i], SJF, DIST)

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

# Creating list with the average waiting time per customer for all servers
mean_waiting_pc = []
serverlist = []
for i in range(len(N_servers)):
    sublist = list(np.array(sum_waiting[i])/N_sim)
    mean_waiting_pc.extend(sublist)
    for c in range(N_CUSTOMERS):
        serverlist.append(i)

if SJF == False:
    # Converting list with to csv for waiting time per customer
    sum_data = {"Server":serverlist, "Waiting pc": mean_waiting_pc}
    df_waiting_pc = pd.DataFrame(sum_data)
    df_waiting_pc.to_csv("waiting_pc_det.csv")

    # Data file with all mean waiting times per simulation
    data = {"Server": data_list[0], "Mean Wait": data_list[1]}
    df_data = pd.DataFrame(data)
    df_data.to_csv("data_det.csv")

elif SJF == True:
    # Converting list with to csv for waiting time per customer
    sum_data = {"Server":serverlist, "Waiting pc": mean_waiting_pc}
    df_waiting_pc_sjf = pd.DataFrame(sum_data)
    df_waiting_pc_sjf.to_csv("waiting_pc_sjf.csv")

    # Data file with all mean waiting times per simulation
    data = {"Server": data_list[0], "Mean Wait": data_list[1]}
    df_data_jsf = pd.DataFrame(data)
    df_data_jsf.to_csv("data_sjf.csv")

# Print statement to give summary of total run
t1 = time.time()
print("\nDid", N_sim, "simulations with", N_CUSTOMERS, "customers, for server",\
    N_servers, "in total time of ", round(t1-t0,3), "seconds\n")

