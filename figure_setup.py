"""figure_setup.py

Collection of simple functions setting up the plotting canvas in matplotlib
"""
__version__ = 0.0
__author__ = 'Joanna Piotrowska'

# ==================================================================
# imports
# ==================================================================
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ==================================================================
# python function definitions
# ==================================================================
def configure_plots():
    """
    Sets global Matplotlib settings to include the following features:
        - inward-facing ticks on all axis spines
        - latex rendering enabled
        - serif fonts in text
        - no frame around the legend
        - dotted gridlines (once matplotlib.pyplot.axes.grid(True)  
        - figure DPI set to 300 for PDF renders
        - default saving format as PDF
    """
    # line settings
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 3
    rcParams['errorbar.capsize'] = 3

    # tick settings
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['xtick.major.size'] = 7
    rcParams['xtick.minor.size'] = 4
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.major.size'] = 7
    rcParams['ytick.minor.size'] = 4
    rcParams['ytick.direction'] = 'in'

    # text settings
    rcParams['mathtext.rm'] = 'serif'
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['text.usetex'] = True
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 15
    rcParams['axes.ymargin'] = 0.5

    # legend
    rcParams['legend.fontsize'] = 12
    rcParams['legend.frameon'] = False

    # grid in plots
    rcParams['grid.linestyle'] = ':'

    # figure settings
    rcParams['figure.figsize'] = 4,3
    rcParams['figure.dpi'] = 300
    rcParams['savefig.format'] = 'png'

def configure_black():
    """
    Sets spine, tick and axes facecolor for generating plots in black.
    Applies the following:
        - white spines
        - white ticks
        - white text colour
        - black axis facecolour
        - black figure facecolor (NOTE: in order to save the figure instance 
        'fig' with black figure facecolor, you need to pass a keyword in the 
        savefig function, i.e. fig.savefig('figure-name', facecolor='k')
    """
    # figure
    rcParams['figure.facecolor'] = 'k'

    # ticks
    rcParams['xtick.color'] = 'w'
    rcParams['ytick.color'] = 'w'

    # axes
    rcParams['axes.edgecolor'] = 'w'
    rcParams['axes.facecolor'] = 'k'
    rcParams['axes.labelcolor'] = 'w'

    # text
    rcParams['text.color'] = 'w'
    rcParams['font.family'] = 'serif'

def configure_dark():
    """
    Sets spine, tick and axes facecolor for generating plots in 'dark'.
    Applies the following:
        - light yellow spines
        - light yellow ticks
        - light yellow text colour
        - dark grey axis facecolour
        - dark grey figure facecolor (NOTE: in order to save the figure instance 
        'fig' with dark grey figure facecolor, you need to pass a keyword in the 
        savefig function, i.e. fig.savefig('figure-name', facecolor='#222222')
    """
    # figure
    rcParams['figure.facecolor'] = '#222222'

    # ticks
    rcParams['xtick.color'] = 'lightyellow'
    rcParams['ytick.color'] = 'lightyellow'

    # axes
    rcParams['axes.edgecolor'] = 'lightyellow'
    rcParams['axes.facecolor'] = '#222222'
    rcParams['axes.labelcolor'] = 'lightyellow'

    # text
    rcParams['text.color'] = 'lightyellow'
    rcParams['font.family'] = 'serif'

def set_linear_ticks(ax, xmaj, ymaj, axis='both', side=None, color='k', nsub=5):
    """
    Places and formats ticks at appropriate locations in a linear scale.
    The default number of minor subdivisions is set to 5.

    Args:
        -- ax: matplotlib.pyplot.axis instance, axis to modify
        -- xmaj (float): xaxis major locator argument
        -- ymaj (float): yaxis major locator argument
        -- axis (str, optional): which axes to modify. Accepts 'x', 'y' or 'both.
            Defaults to 'both'.
        -- side (str, optional): whether to modify only one side in the figure. 
            Accepts 'right', 'left' or None. Defaults to None.
        -- color (str, optional): tick colour. Defaults to 'k', i.e. black. 
        -- nsub (int, optional): number of minor tick subdivisions. Defaults
            to 5.
    """
    # setting ticks on one side only
    if side is not None:
        if side == 'right':
            left, labelleft = False, False
            right, labelright = True, True
        elif side == 'left':
            right, labelright = False, False
            left, labelleft = True, True
        else:
            print('Ooops, looks like you need to choose your side again!')
            return
        
        for i, (which, length) in enumerate(zip(['major', 'minor'], [7,4])):
            ax.tick_params(which=which, axis=axis, direction='in', length=length,
                            right=right, left=left, labelleft=labelleft, 
                            labelright=labelright, color=color)
    
    # setting ticks on both right & left
    else:
        left, right, top, bottom = True, True, True, True
        labelleft, labelbottom = True, True
        
        for i, (which, length) in enumerate(zip(['major', 'minor'], [7,4])):
            ax.tick_params(which=which, axis=axis, direction='in', length=length,
                           right=right, top=top, left=left, bottom=bottom,
                           labelleft=labelleft, labelbottom=labelbottom, 
                           color=color)

    # setting tick locator
    ax.xaxis.set_major_locator(MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=nsub))
    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=nsub))
