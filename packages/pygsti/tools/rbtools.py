from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from matplotlib import pyplot as _plt


def rb_decay(x,a,b):
    """
    Return the Randomized Benchmarking (RB) decay function:

    decay(x,a,b) = (1 + (2a-1) exp(-bx))/2
    """
    return (1+(2*a-1)*_np.exp(-b * x))/2.

def rb_decay_rate(dataset,showPlot=False,xlim=None,ylim=None,saveFigPath=None):
    """
    Compute the Randomized Benchmarking (RB) decay rate given an data set
    containing counts for RB gate strings.  Note: currently this function
    only works for 1-qubit dataset having SPAM labels 'plus' and 'minus'.

    Parameters
    ----------
    dataset : DataSet
      The RB data set.

    showPlot : bool, optional
       Whether to show a plot of the fit to the RB data.

    xlim : (xmin,xmax), optional
       Specify x-axis limits for plot

    ylim : (ymin,ymax), optional
       Specify y-axis limits for plot

    saveFigPath : string, optional
       Pathname to save a plot of the fit to the RB data.

    Returns
    -------
    a,b : float
       The best-fit decay curve parameters a and b, as defined in
       the rb_decay function.
    """
    RBlengths = []
    RBsuccesses = []
    for key in list(dataset.keys()):
        dataLine = dataset[key]
        plus = dataLine['plus']
        minus = dataLine['minus']
        N = plus + minus
        RBlengths.append(len(key))
        RBsuccesses.append(1 - dataLine['plus']/float(N))
        if dataLine['plus']/float(N) > 1:
            print(key)
    a,b = _curve_fit(rb_decay,RBlengths,RBsuccesses)[0]
    if saveFigPath or showPlot:
        newplot = _plt.figure()
        newplotgca = newplot.gca()
        newplotgca.plot(RBlengths,RBsuccesses,'.')
        newplotgca.plot(range(max(RBlengths)),
                        rb_decay(_np.arange(max(RBlengths)),a,b),'+')
        newplotgca.set_xlabel('RB sequence length (non-Clifford)')
        newplotgca.set_ylabel('Success rate')
        newplotgca.set_title('RB success')
        if xlim:
            _plt.xlim(xlim)
        if ylim:
            _plt.ylim(ylim)
    if saveFigPath:
        newplot.savefig(saveFigPath)
    return a,b
