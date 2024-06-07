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


logg_num = [3.5, 4, 4.5, 5, 5.5]
logg_ticks = ['Less dense', '4', '4.5', '5', 'More dense']


norm_g = mpl.colors.Normalize(vmin=min(logg_num)-.15,
                              vmax=max(logg_num)+.12,  clip=True)

norm_f = mpl.colors.SymLogNorm(linthresh=4, vmin=min(fsed_num)-.15,
                               vmax=max(fsed_num)+1, clip=True)



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



def long_plot(df_sources, convolve_data_dict, x_min = 1.16, x_max = 1.185, 
                  x_increment = 0.005, const_spacing = 1.5, norm_scaling = 7e10,
                  title = "Potassium Doublets", color_by_logg = True):
    """
    Plots normalized and convolved spectral data with annotations and color-coding.

    Parameters:
    df_sources (pandas.DataFrame): DataFrame containing the sources and their parameters.
                                   It should have columns 'name', 'logg', and 'clouds'.
    convolve_data_dict (dict): Dictionary containing convolved spectral data arrays. 
                               Keys are source names, and values are numpy arrays with shape (3, length),
                               where the first row is wavelength data and the second row is flux data.
    x_min (float, optional): Minimum x-axis value for the plot. Default is 1.16.
    x_max (float, optional): Maximum x-axis value for the plot. Default is 1.185.
    x_increment (float, optional): Increment for x-axis ticks. Default is 0.005.
    const_spacing (float, optional): Constant spacing added to each spectrum for separation in the plot. Default is 1.5.
    norm_scaling (float, optional): Scaling factor for normalizing the flux data. Default is 7e10.
    title (str, optional): Title of the plot. Default is "Potassium Doublets".
    color_by_logg (bool, optional): If True, spectra are color-coded by logg. If False, spectra are color-coded by clouds. Default is True.

    This function performs the following tasks:
    1. Sorts the sources based on 'logg' and 'clouds' and determines the plotting order.
    2. Sets up constants for vertical spacing of the spectra.
    3. Initializes a matplotlib figure and axis for plotting.
    4. Iterates over the sorted sources to:
        a. Normalize the convolved flux data.
        b. Plot each spectrum with appropriate color-coding and vertical spacing.
        c. Annotate the plot with fitted parameters.
    5. Configures plot aesthetics including x and y limits, ticks, grid lines, and axis labels.
    6. Adds potassium doublet vertical lines and annotations.
    7. Optionally adds horizontal lines and labels to indicate different clouds or logg values.
    8. Adds a color bar indicating the parameter used for color-coding.
    """
    if color_by_logg:
        index = df_sources.sort_values(
            by=['clouds', 'logg']).index.to_numpy()  # order of spectra plotted
    else:
        index = df_sources.sort_values(by=['logg', 'clouds']).index.to_numpy()

    label_loc_l = []  # where left label located on y axis
    label_loc_r = []  # where right label located on y axis

    if color_by_logg:
        const = [(i*const_spacing) + const_spacing*(i//5)
                for i in range(30)]  # constants to be added to flux
    else:
        const = [(i*const_spacing) + 1.5*const_spacing*(i//6) for i in range(30)]

    fig, ax = plt.subplots(figsize=(6, 16))

    # plotting each spectra
    for n, i in enumerate(index):
        # setting constants, labels, color needed
        if color_by_logg:
            color = logg_colors(norm_g(df_sources.logg[i]))
        else:
            color = fsed_colors(norm_f(df_sources.clouds[i]))
        name = df_sources.name[i]
        c = const[n]

        # normalizing
        norm = (convolve_data_dict[name][1, :]) / norm_scaling

        # plotting
        ax.plot(convolve_data_dict[name][0, :], norm + c, alpha=1, color=color)

        # adding label for avg(A) and avg(FWHM)
        l_index = np.where(np.isclose(convolve_data_dict[name][0, :], x_min))[
            0][0]  # x index in spectra of left side
        r_index = np.where(np.isclose(convolve_data_dict[name][0, :], x_max))[
            0][0]  # x index in spectra of right side
        label_loc_r.append(norm[r_index] + c)   # y axis location for labels
        label_loc_l.append(norm[l_index] + c)

        A_label = f' {-(df_sources.A1[i] + df_sources.A2[i]) / 1e11:.2f}'  # string of A
        ax.annotate(A_label,
                    xy=(x_max, label_loc_r[n]), xycoords='data', color=color)
        FWHM_label = f'{df_sources.FWHM1[i]*1e3:.2f} '  # string for FWHM
        ax.annotate(FWHM_label, horizontalalignment='right',
                    xy=(x_min, label_loc_l[n]), xycoords='data', color=color)


    # setting limits
    y_max = max(label_loc_r)+2  # right side generally larger of 2
    y_min = min(label_loc_l)-2
    # limits in x and y
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # setting ticks
    ax.set_yticks([n for n in range(int(y_min), int(y_max) + 1)])  # y-ticks
    # creating blank label in y
    ax.set_yticklabels(['' for n in range(int(y_min), int(y_max) + 1)])
    # setting x-ticks with x-increment set at top
    ax.set_xticks(ticks=[round(x_min + n*x_increment, 3)
                for n in range(0, int((x_max-x_min)/x_increment)+1)])
    # creating grid
    ax.tick_params(which='major', length=5, width=0,
                direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=2, width=0,
                direction='in', top=True, right=True)
    # grid lines
    ax.minorticks_on()
    ax.grid(visible=True, which='major', alpha=.5)
    ax.grid(visible=True, which='minor', alpha=.1)


    # adding axis titles/labels
    ax.set_xlabel('Wavelength (μm)', fontsize=13)
    ax.set_ylabel("Normalized Flux + Constant\n\n", fontsize=13)
    ax.set_title(title, fontsize=13)

    # setting parameter labels
    ax.annotate(r"$avg(A)$  " "\n Max Depth  \n", xy=(x_min, y_max),
                ha='right',
                va='top',
                xycoords='data', color='k')

    ax.annotate(r"  $avg(\sigma)$" "\n  FWHM \n", xy=(1.190, y_max),
                ha='left',
                va='top',
                xycoords='data', color='k')

    # potassium vertical lines
    ax.vlines([1.16935, 1.1775], ymin=y_min, ymax=y_max-1,
            linestyle='dotted', color='k', linewidth=1.2, alpha=.8)
    ax.annotate("K I ", xy=(1.16901, y_max-1),
                xycoords='data', color='k', fontsize=12)
    ax.annotate("K I ", xy=(1.177062, y_max-1),
                xycoords='data', color='k', fontsize=12)


    # horizontal lines
    if color_by_logg:
        hline_y = [label_loc_r[(i*5)-1]+.5 for i in range(1, 7)]
        pretty_fsed = [' Cloudy \n $f_{sed} = 1$',
                    r' $f_{sed} = 2$', r' $f_{sed} = 3$',
                    r' $f_{sed} = 4$', r' $f_{sed} = 8$',
                    'No Clouds']
        ax.hlines(hline_y, xmin=x_min + 0.005, xmax=x_max,  color='k')

        for i, y in enumerate(hline_y):
            ax.annotate(pretty_fsed[i], xy=(x_min + 0.0001, y - .1),
                        xycoords='data', color='k', fontsize=9)
    else:
        hline_y = [label_loc_r[(i*6)-1]+.5 for i in range(1, 6)]
        pretty_logg = [' least dense\n'+r' $\log(g) = 3.5$',
                    r' $\log(g) = 4.0$', r' $\log(g) = 4.5$',
                    r' $\log(g) = 5.0$',
                    ' most dense \n' + r' $\log(g) = 5.5$']
        ax.hlines(hline_y, xmin=x_min + 0.006, xmax=x_max,  color='k')

        for i, y in enumerate(hline_y):
            ax.annotate(pretty_logg[i], xy=(x_min + 0.0001, y - .1),
                        xycoords='data', color='k', fontsize=9)


    # color bar
    if color_by_logg:
        axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_g, cmap=logg_colors),
                            ticks=logg_num, shrink=1, format=mticker.FixedFormatter(logg_ticks),
                            aspect=50,  pad=.14)
        axcb.set_label(' ', fontsize=12)  # empty label
        ax.annotate(r'$\log(g)$', xy=(.9, .5), xycoords='figure fraction',
                    rotation=270, fontsize=13)  # actual color bar label

    else:
        axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_f, cmap=fsed_colors),
                            ticks=fsed_num, shrink=1, format=mticker.FixedFormatter(fsed_ticks),
                            aspect=50, pad=.14)
        axcb.set_label(' ', fontsize=12)  # empty label
        ax.annotate(r'$f_{sed}$', xy=(.9, .5), xycoords='figure fraction',
                    rotation=270, fontsize=13)  # actual color bar label
