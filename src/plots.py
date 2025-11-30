import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

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
    plt.show()
    #plt.savefig(save_path, transparent = True)

def plot_performance(performance_overview, time_var_name, crash_resistant, _dir):
    """
    time_var_name = 'rounded_tau' or 'nweeks'
    plot_fname = 'plots/' + 'zero_beta_straddle_tau=' + tau_label + '.png'
    """
    performance_overview['day'] = pd.to_datetime(performance_overview['day'])
    for tau in performance_overview[time_var_name].unique():

        tau_sub = performance_overview.loc[performance_overview[time_var_name] == tau]
        tau_sub.sort_values('day', inplace = True)
        tau_label = str(tau)

        # Group per Week
        # @Todo: Now relate this plot to the IV over Realized Vola premium!!
        fig = plt.figure(figsize = (20,14))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            
        plt.subplot(2, 1, 1)
        plt.plot(tau_sub['day'], tau_sub['combined_payoff'], label = tau_label)
        plt.ylim(-5000, 5000)
        
        plt.subplot(2,1,2)
        plt.plot(tau_sub['day'], tau_sub['combined_ret'], label = tau_label)
        plt.ylim(-2, 2)

        plt.gcf().autofmt_xdate()
        plt.legend()
        if crash_resistant:
            fname = _dir + 'zero_beta_straddle_' + time_var_name + '=' + tau_label + '.png'
        else:
            fname = _dir + 'zero_beta_straddle_' + time_var_name + '=' + tau_label + '.png'
        plt.savefig(fname, transparent = True)

def grouped_boxplot(performance_overview, target_var_name, group_var_name, ylim_min = None, ylim_max = None, _dir = '', file_name_addition = '', crash_resistant = False, nth_label = 10, show_gridlines = False):
    """

    """
    fig, ax = plt.subplots(figsize=(20,16))
    performance_overview.boxplot(column=[target_var_name], by=group_var_name, ax=ax, showmeans=True)
    if ylim_min is not None and ylim_max is not None:
        ax.set_ylim(ylim_min, ylim_max)
    fig.tight_layout()
    if crash_resistant:
        file_name = _dir + 'boxplot_' + target_var_name + '_over_' + group_var_name + '_' + file_name_addition + '.png'
    else:
        file_name = _dir + 'boxplot_' + target_var_name + '_over_' + group_var_name + '_' + file_name_addition + '.png'
    
    # Deactivate auto Title
    plt.suptitle('')
    plt.title('')
    ax.get_figure().suptitle('')

    # Hide Grid Lines
    ax.grid(show_gridlines)

    # Adjust labels
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % nth_label != 0]
    #plt.show()
    plt.savefig(file_name, transparent = True)

def group_plot(sub, x_col_str, y_col_str, group_col_str, save_file_str):
    """
    Grouped Line Plot
    """
    fig, ax = plt.subplots(figsize=(10,4))
    for key, grp in sub.groupby([group_col_str]):
        ax.plot(grp[x_col_str], grp[y_col_str], label=key)
    #ax.legend()
    plt.savefig(save_file_str)