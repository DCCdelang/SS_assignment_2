#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

#%%
# perform serveral signifance tests for the mean waiting time for each server

df = pd.read_csv("data.csv")

# split data batch means of for each server
MM_1 = df[df["Server"] == 0]['Mean Wait']
MM_2 = df[df["Server"] == 1]['Mean Wait']
MM_4 = df[df["Server"] == 2]['Mean Wait']

# perform t-tests for different combinations
ttest_1_2 = stats.ttest_ind(MM_1, MM_2, equal_var = False)
ttest_2_4 = stats.ttest_ind(MM_2, MM_4, equal_var = False)

print('p value for t-test 1 and 2 servers:', ttest_1_2.pvalue)
print('p value for t-test 2 and 4 servers:', ttest_2_4.pvalue)

anova = stats.f_oneway(MM_1, MM_2, MM_4)
print('p value for ANOVA 1, 2 and 4 servers:', anova.pvalue)

#%%
# Bocplot function for all servers combined 
def boxplot_wait():
    """Create boxplot of mean time, according to csv"""
    sns.boxplot(x = df["Server"], y = df["Mean Wait"])
    plt.show()
boxplot_wait()

#%%
# Give three plots of the distributions found by the mean waiting time for 
# all simulations per server
for n in range(3):
    serverdata = df.loc[df2["Server"] == n, "Mean Wait"]
    sns.distplot(serverdata, bins= 20, rug=True)
    plt.show()

# %%
# Gives three plots of the waiting time per customer for one simulation
for i in range(3):
    df = pd.read_csv("test_server_"+str(i)+".csv")
    sns.lineplot(data=df["Waiting"])
    plt.show()

# %%
# Plot to determine when the steady state starts
df4 = pd.read_csv("waiting_pc.csv")
for n in range(3):
    waitingdata = df4.loc[df4["Server"] == n, "Waiting pc"]
    sns.lineplot(data=waitingdata)
    plt.show()