import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

N_servers = [1,2,4]

""" Dataframes based on different configurations"""
# Mean waiting data with mu = .92 and lambda = 1
df_92 = pd.read_csv("data/data_92.csv")
df_sjf_92 = pd.read_csv("data/data_sjf_92.csv")

# PC waiting data with mu = .92 and lambda = 1
df_pc_92 = pd.read_csv("data/waiting_pc_92.csv")
df_pc_sjf_92 = pd.read_csv("data/waiting_pc_sjf_92.csv")

# Deterministic data with mu = .92 and lambda = 1
df_det = pd.read_csv("data/data_det.csv")
df_pc_det = pd.read_csv("data/waiting_pc_det.csv")

# Deterministic data with mu = .92 and lambda = 1
df_lt = pd.read_csv("data/data_lt.csv")
df_pc_lt = pd.read_csv("data/waiting_pc_lt.csv")

# Matrix dataframe
df_matrix_92 = pd.read_csv("data/waiting_matrix_92.csv")
df_matrix_90 = pd.read_csv("data/waiting_matrix_90.csv")
df_matrix_80 = pd.read_csv("data/waiting_matrix_80.csv")
df_matrix_50 = pd.read_csv("data/waiting_matrix_50.csv")

# perform serveral signifance tests for the mean waiting time for each server
# split data batch means of for each server
def statistics(df):
    MM_1 = df[df["Server"] == 0]["Mean Wait"]
    MM_2 = df[df["Server"] == 1]["Mean Wait"]
    MM_4 = df[df["Server"] == 2]["Mean Wait"]

    for i in range(3):
        df[df["Server"] == 2]["Mean Wait"]
        print("E(W) for ")

    # Perform Welch t-tests for different combinations
    ttest_1_2 = stats.ttest_ind(MM_1, MM_2, equal_var = False)
    ttest_2_4 = stats.ttest_ind(MM_2, MM_4, equal_var = False)

    print("p value for t-test 1 and 2 servers:", ttest_1_2.pvalue)
    print("p value for t-test 2 and 4 servers:", ttest_2_4.pvalue)

    # Perform ANOVA test
    anova = stats.f_oneway(MM_1, MM_2, MM_4)
    print("p value for ANOVA 1, 2 and 4 servers:", anova.pvalue)

    shap_wilk_1 = stats.shapiro(MM_1)
    print("p value of Shapiro-Wilk test for 1 server", shap_wilk_1.pvalue)
    shap_wilk_2 = stats.shapiro(MM_2)
    print("p value of Shapiro-Wilk test for 2 servers", shap_wilk_2.pvalue)
    shap_wilk_4 = stats.shapiro(MM_4)
    print("p value of Shapiro-Wilk test for 4 servers", shap_wilk_4.pvalue)

# Bocplot function for all servers combined 
def boxplot_wait(df):
    """Create boxplot of mean time, according to csv"""
    plot = sns.boxplot(x = df["Server"], y = df["Mean Wait"])
    plot.set_xticklabels(N_servers)
    plot.tick_params(labelsize=14)
    plot.set_ylabel("Mean waiting time",fontsize=14)
    plot.set_xlabel("Server c =",fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/boxplot_92.png", dpi=300)
    # plt.title("Distributions of mean waiting time for different c")
    plt.show()

# Give three plots of the distributions found by the mean waiting time for 
# all simulations per server
def mean_waiting(df,single):
    for n in range(3):
        serverdata = df.loc[df["Server"] == n, "Mean Wait"]
        plot = sns.distplot(serverdata, label = "Server c="+str(N_servers[n]))
        plot.set_ylabel("Probability", fontsize=14)
        plot.set_xlabel("Mean waiting time",fontsize=14)
        # plt.title("PDF of mean waiting time for different c")
        plt.legend(fontsize=14)
        if single == True:
            plt.show()
    if single != True:
        plt.tight_layout()
        plt.tick_params(labelsize=14)
        plt.savefig("figures/mean_waiting_lt.png", dpi=300)
        plt.show()

# Plot to determine when the steady state starts
def waiting_pc(df):
    for n in range(3):
        waitingdata = df.loc[df["Server"] == n, "Waiting pc"]
        x = np.arange(waitingdata.shape[0])
        plot = sns.lineplot(x=x,y=waitingdata, label = "Server c="+str(N_servers[n]))
        plot.set(ylabel="Waiting time", xlabel="Customer number",\
            xlim = (0, waitingdata.shape[0]))
        # plt.title("Average waiting time per customer for different c")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.tick_params(labelsize=14)
    plt.savefig("figures/waiting_pc_lt.png", dpi=300)
    plt.show()

# Plot mean waiting time, with and without SJF for server n
def compare_sjf(df,df_sjf,n):
    serverdata = df.loc[df["Server"] == n, "Mean Wait"]
    plot = sns.distplot(serverdata, label = "Server")
    serverdata_sjf = df_sjf.loc[df["Server"] == n, "Mean Wait"]
    plot = sns.distplot(serverdata_sjf, label = "Server SJF")
    plot.set(ylabel="Probability",xlabel="Mean waiting time")
    # plt.title("PDF of mean waiting time with and without SJF for c=1")
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("figures/compare_sjf_92.png", dpi=300)
    plt.show()

def std_plot(dfs):
    labels = ["0.92", "0.90", "0.80", "0.50"]
    i = -1
    for df in dfs:
        i += 1
        df = df.transpose()
        df = df.drop(df.index[0])
        df.head()
        std = df.std(axis=1)

        x = np.arange(10000)
        plt.plot(x,std, label = "$\\rho$ ="+labels[i])
    plt.ylabel("Standard deviation $\sigma$",fontsize=14)
    plt.xlabel("Customer number",fontsize=14)
    plt.xlim(0,10000)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("figures/std_compare.png", dpi=300)
    plt.show()

def different_N():
    for exp in range(2,5):
        df = pd.read_csv("data/data_"+str(10**exp)+".csv")
        serverdata = df.loc[df["Server"] == 0, "Mean Wait"]
        plot = sns.distplot(serverdata, label = "N = "+str(10**exp),hist=False)
    df = pd.read_csv("data/data_"+str(20000)+".csv")
    serverdata = df.loc[df["Server"] == 0, "Mean Wait"]
    plot = sns.distplot(serverdata, label = "N = "+str(20000),hist=False)
    df = pd.read_csv("data/data_"+str(40000)+".csv")
    serverdata = df.loc[df["Server"] == 0, "Mean Wait"]
    plot = sns.distplot(serverdata, label = "N = "+str(40000),hist=False)
    plot.tick_params(labelsize=14)
    plot.set_ylabel("Probability",fontsize = 14)
    plot.set_xlabel("Mean waiting time",fontsize=14)
    plt.ylim(0,0.4)
    plt.xlim(0,20)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/PDF_compare_N.png", dpi=300)
    plt.show()

"""Choose a df dataframe"""
# boxplot_wait(df_92)
# statistics(df_92)
# mean_waiting(df_92, single = False)
# mean_waiting(df_det, single = False)
# mean_waiting(df_lt, single = False)
# compare_sjf(df_92, df_sjf_92,0)

"""Choose a df_pc dataframe"""
# waiting_pc(df_pc_sjf_92) 
# waiting_pc(df_pc_92) 
# waiting_pc(df_pc_det) 
waiting_pc(df_pc_lt) 

"""Choose matrix dataframe"""
# std_plot([df_matrix_92,df_matrix_90,df_matrix_80,df_matrix_50])

# different_N()
