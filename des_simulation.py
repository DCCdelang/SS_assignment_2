import numpy as np
import random
import simpy


RANDOM_SEED = 420
# NEW_CUSTOMERS = 5  # Total number of customers
LAMBDA = 3.0  # Generate new customers roughly every x seconds
MIN_PATIENCE = 1  # Min. customer patience
MAX_PATIENCE = 10  # Max. customer patience


def generate_customers(env, arrival_rate, counter):
    """generates customers randomly"""
    id = 0
    while True:
        id += 1
        mu = random.expovariate(1.0/arrival_rate)
        c = customer(env, 'Customer%02d' % id, counter, time_in_bank=mu)
        env.process(c)
        # mu = random.expovariate(1.0/arrival_rate)
        wait_time.append(mu)
        yield env.timeout(mu)

    


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    # arrive = env.now
    # print('%7.4f %s: Here I am' % (arrive, name))

    with counter.request() as req:
        arrive = env.now
        yield req
        # patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)
        # # Wait for the counter or abort at the end of our tether
        # results = yield req | env.timeout(patience)

        wait = env.now - arrive
        
        # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
        # tib = random.expovariate( time_in_bank)
        yield env.timeout(time_in_bank)
        wait_time.append(env.now-arrive)
        # print('%7.4f %s: Finished' % (env.now, name))

        # else:
        #     # We reneged
        #     print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))

    return wait

        

    
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
        env.process(generate_customers(env, LAMBDA, counter))
        env.run(until=1000)
        
        mu = np.mean(wait_time)
        rho.append(LAMBDA / (n * mu))
    print(f'rho with {n} servers: {np.mean(rho)}')
