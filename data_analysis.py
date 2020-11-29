import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

N_servers = [1,2,4]

df = pd.read_csv("data.csv")
df_sjf = pd.read_csv("data_sjf.csv")
df_pc = pd.read_csv("waiting_pc.csv")
df_pc_sjf = pd.read_csv("waiting_pc.csv")

# perform serveral signifance tests for the mean waiting time for each server
# split data batch means of for each server
def statistics(df):
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

# Bocplot function for all servers combined 
def boxplot_wait(df):
    """Create boxplot of mean time, according to csv"""
    plot = sns.boxplot(x = df["Server"], y = df["Mean Wait"])
    plot.set_xticklabels(N_servers)
    plot.set(ylabel="Mean waiting time",xlabel="Server c =")
    plt.title("Distributions of mean waiting time for different c")
    plt.show()

# Give three plots of the distributions found by the mean waiting time for 
# all simulations per server
def mean_waiting(df,single):
    for n in range(3):
        serverdata = df.loc[df["Server"] == n, "Mean Wait"]
        plot = sns.distplot(serverdata, label = "Server c="+str(N_servers[n]))
        plot.set(ylabel="Proportion",xlabel="Mean waiting time")
        plt.title("PDF of mean waiting time for different c")
        plt.legend()
        if single == True:
            plt.show()
    if single != True:
        plt.show()

# Plot to determine when the steady state starts
def waiting_pc(df):
    for n in range(3):
        waitingdata = df.loc[df["Server"] == n, "Waiting pc"]
        x = np.arange(waitingdata.shape[0])
        plot = sns.lineplot(x=x,y=waitingdata, label = "Server c="+str(N_servers[n]))
        plot.set(ylabel="Waiting time per customer", xlabel="Customer number")
        plt.title("Average waiting time per customer for different c")
        plt.legend()
    plt.show()

# Plot mean waiting time, with and without SJF for server n
def compare_sjf(df,df_sjf,n):
    serverdata = df.loc[df["Server"] == n, "Mean Wait"]
    plot = sns.distplot(serverdata, label = "Server")
    serverdata_sjf = df_sjf.loc[df["Server"] == n, "Mean Wait"]
    plot = sns.distplot(serverdata_sjf, label = "Server SJF")
    plot.set(ylabel="Proportion",xlabel="Mean waiting time")
    plt.title("PDF of mean waiting time with and without SJF for c = 1")
    plt.legend()
    plt.show()

""" Choose df or df_sjf """
# boxplot_wait(df)
# statistics(df)
# mean_waiting(df, single = False)
# mean_waiting(df_sjf, single = False)
# compare_sjf(df, df_sjf,0)

""" Choose df_pc or df_pc_sjf """
# waiting_pc(df_pc) 


