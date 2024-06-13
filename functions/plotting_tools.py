import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np
from scipy import stats
import matplotlib.ticker as mticker

# color mapping
fsed_colors = pl.cm.viridis
logg_colors = pl.cm.plasma

fsed_num = [1, 2, 3, 4, 8, 10]
fsed_ticks = ["Cloudy", '2', '3', '4', '8', 'No \n Clouds']
fsed_bounds = [0.5, 1.5, 2.5, 3.5, 6, 9, 11]

logg_num = [3.5, 4, 4.5, 5, 5.5]
logg_ticks = ['Less dense', '4', '4.5', '5', 'More dense']
logg_bounds = [3.25, 3.75, 4.25, 4.75, 5.25, 5.75]

norm_f = mpl.colors.BoundaryNorm(fsed_bounds, fsed_colors.N, extend='neither')
norm_g = mpl.colors.BoundaryNorm(logg_bounds, logg_colors.N, extend='max')



def fsed_colorbar(fig, cax = None, ax = None, orientation='vertical',
                   shrink=1.0, aspect=20, pad=.14):

    return fig.colorbar(pl.cm.ScalarMappable(norm=norm_f, cmap=fsed_colors),
                cax=cax, ax = ax, orientation= orientation, 
                ticks=fsed_num, format=mticker.FixedFormatter(fsed_ticks),
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    
def logg_colorbar(fig, cax = None, ax = None, orientation='vertical',
                  shrink=1.0, aspect=20, pad=.14):

    return fig.colorbar(pl.cm.ScalarMappable(norm=norm_g, cmap=logg_colors),
                cax=cax, ax = ax, orientation= orientation,
                ticks=logg_num,  format=mticker.FixedFormatter(logg_ticks),
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)

def plot_parameter_vs_logg_fsed(ax1, ax2, parameter, logg, fsed, param_name, lines = False):
    """
    Plots a given parameter against two variables, `logg` and `fsed`, on separate matplotlib axes.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first axis to plot `parameter` against `logg`.
    ax2 (matplotlib.axes._subplots.AxesSubplot): The second axis to plot `parameter` against `fsed`.
    parameter (array-like): The parameter values to be plotted.
    logg (array-like): The log(g) values to be plotted.
    fsed (array-like): The log(g) values to be plotted.
    param_name (str): The name of the parameter to be used as the y-axis label.
    lines (bool, False): If True a line will be plotted for each individual fsed and logg grouping.

    This function performs the following tasks:
    1. Plots scatter plots of `parameter` against `logg` on `ax1` and `parameter` against `fsed` on `ax2`.
    2. Colors the points on `ax1` based on `fsed` values and on `ax2` based on `logg` values.
    3. Fits and plots linear regression lines for the data on both axes.
    4. Calculates and annotates Pearson correlation coefficients on both axes.
    5. Sets custom x-ticks and x-tick labels for both axes.
    6. Sets the y-axis label of `ax1` to `param_name`.

    Example usage:
    ```
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_parameter(ax1, ax2, df['parameter'], 'Parameter Name')
    plt.show()
    ```
    """

    logg_values = list(set(logg))
    fsed_values = list(set(fsed))

    # plotting each point individually to assign the correct color
    for i in range(len(logg)):
        ax1.scatter(logg[i], parameter[i],
                    color=fsed_colors(norm_f(fsed[i])), marker='x')
        ax2.scatter(fsed[i], parameter[i],
                    color=logg_colors(norm_g(logg[i])), marker='x')
        
    # creating line on the first plot
    m, b = np.polyfit(logg, parameter, 1)
    ax1.plot(logg, logg*m + b, color='k',
             linewidth=.7, alpha=1, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r1 = stats.pearsonr(logg, parameter)[0]
    ax1.annotate(f"r = {r1:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')

    # creating line on the second plot
    m, b = np.polyfit(fsed, parameter, 1)
    ax2.plot(fsed, fsed*m + b, color='k',
             linewidth=.7, alpha=0.5, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r2 = stats.pearsonr(fsed, parameter)[0]
    ax2.annotate(f"r = {r2:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')
    
    # plotting the lines for each logg and fsed value grouping 
    if lines:
        for grav in logg_values:
            m, b = np.polyfit(fsed[logg == grav], parameter[logg == grav] , 1)
            ax2.plot(fsed, fsed*m + b, color=logg_colors(norm_g(grav)),
                        linewidth=.7, alpha=.5)
        for f in fsed_values:
            m, b = np.polyfit(logg[fsed == f], parameter[fsed == f] , 1)
            ax1.plot(logg, logg*m + b, color=fsed_colors(norm_f(f)),
                        linewidth=.7, alpha=.5)

    # creating custom x-ticks 
    ax1.set_xticks(logg_num)
    ax1.set_xticklabels([None for i in logg_num])

    ax2.set_xticks(fsed_num)
    ax2.set_xticklabels([None for i in fsed_num])

    ax1.set_ylabel(param_name)



