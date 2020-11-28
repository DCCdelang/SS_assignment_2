import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Bocplot function for all servers combined 
def boxplot_wait():
    """Create boxplot of mean time, according to csv"""
    df = pd.read_csv("data.csv")
    sns.boxplot(x = df["Server"], y = df["Mean Wait"])
    plt.show()
boxplot_wait()

#%%
# Give three plots of the distributions found by the mean waiting time for 
# all simulations per server
for n in range(3):
    df2 = pd.read_csv("data.csv")
    serverdata = df2.loc[df2["Server"] == n, "Mean Wait"]
    print("Shape of waitingdata per server is:", serverdata.shape)
    sns.distplot(serverdata, bins= 20, rug=True)
    plt.show()

# %%
# Gives three plots of the waiting time per customer for one simulation
for i in range(3):
    df3 = pd.read_csv("test_server_"+str(i)+".csv")
    sns.lineplot(data=df3["Waiting"])
    plt.show()

# %%
# Creates three plots with the mean waiting time per customer number
# useful to determine when the steady state starts
for n in range(3):
    df4 = pd.read_csv("waiting_pc.csv")
    waitingdata = df4.loc[df4["Server"] == n, "Waiting pc"]
    print("Shape of waitingdata per server is:", waitingdata.shape)
    sns.lineplot(data=waitingdata)
    plt.show()
# %%
