import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simple_3d_plot(x,y,z, save_path, xlabel = 'Tau', ylabel = 'Moneyness', zlabel = 'Fitted IV', zlim_min = 0, zlim_max = 4):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.xlim(0, 1)
    plt.ylim(0, 2)
    ax.set_zlim(zlim_min, zlim_max)
    plt.savefig(save_path)

def plot_performance(performance_overview, time_var_name):
    """
    time_var_name = 'rounded_tau' or 'nweeks'
    plot_fname = 'plots/' + 'zero_beta_straddle_tau=' + tau_label + '.png'
    """
    performance_overview['Date'] = pd.to_datetime(performance_overview['Date'])
    for tau in performance_overview[time_var_name].unique():

        tau_sub = performance_overview.loc[performance_overview[time_var_name] == tau]
        tau_sub.sort_values('Date', inplace = True)
        tau_label = str(tau)

        # Group per Week
        # @Todo: Now relate this plot to the IV over Realized Vola premium!!
        fig = plt.figure(figsize = (10,7))
            
        plt.subplot(2, 1, 1)
        plt.plot(tau_sub['Date'], tau_sub['combined_payoff'], label = tau_label)
        plt.ylim(-5000, 5000)
        
        plt.subplot(2,1,2)
        plt.plot(tau_sub['Date'], tau_sub['combined_ret'], label = tau_label)
        plt.ylim(-2, 2)

        plt.legend()
        plt.savefig('plots/zero_beta_straddle_' + time_var_name + '=' + tau_label + '.png')