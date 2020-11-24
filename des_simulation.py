import numpy as np
import random
import simpy


RANDOM_SEED = 420
LAMBDA = 5.0  # Generate new customers roughly every x seconds


def generate_customers(env, arrival_rate, counter):
    """generates customers randomly"""
    id = 0
    while True:
        id += 1
        mu = random.expovariate(1.0/(arrival_rate))
        c = customer(env, 'Customer%02d' % id, counter, waiting_time=mu)
        env.process(c)
        # mu = random.expovariate(1.0/arrival_rate)
        wait_time.append(mu)
        yield env.timeout(mu)


def customer(env, name, counter, waiting_time):
    """Customer arrives, is served and leaves."""
    with counter.request() as req:
        arrive = env.now
        yield env.timeout(waiting_time)
        wait_time.append(env.now-arrive)


    
# Setup and start the simulation with n = 1, 2 and 4. 
servers = [1,2,4]

for n in servers:
    rho = []
    for i in range(100):
        # list to hold the time until customer is helped
        wait_time = [] 

        # print('\nSERVERS:', n, '\n')

        # random.seed(RANDOM_SEED)
        env = simpy.Environment()
        counter = simpy.Resource(env, capacity=n)
        env.process(generate_customers(env, LAMBDA * n, counter))
        env.run(until=1000)
        
        mu = np.mean(wait_time)
        rho.append(LAMBDA / (mu/n))
    print(f'rho with {n} servers: {np.mean(rho)}')
