import numpy as np
import GST
from scipy.optimize import curve_fit
import matplotlib

def decay(x,a,b):
    return (1+(2*a-1)*np.exp(-b * x))/2.
    
def get_decay_rate(RBDS,showPlot=False,xlim=None,ylim=None,saveFigPath=None):
    RBlengths = []
    RBsuccesses = []
    for key in RBDS.keys():
        dataLine = RBDS[key]
        plus = dataLine['plus']
        minus = dataLine['minus']
        N = plus + minus
        RBlengths.append(len(key))
        RBsuccesses.append(1 - dataLine['plus']/float(N))
        if dataLine['plus']/float(N) > 1:
            print key
    a,b = curve_fit(decay,RBlengths,RBsuccesses)[0]
    if saveFigPath or showPlot:
        newplot = matplotlib.pylab.figure()
        newplotgca = newplot.gca()
        newplotgca.plot(RBlengths,RBsuccesses,'.')
        newplotgca.plot(xrange(max(RBlengths)),decay(np.arange(max(RBlengths)),a,b),'+')
        newplotgca.set_xlabel('RB sequence length (non-Clifford)')
        newplotgca.set_ylabel('Success rate')
        newplotgca.set_title('RB success')
        if xlim:
            matplotlib.pyplot.xlim(xlim)
        if ylim:
            matplotlib.pyplot.ylim(ylim)
    if saveFigPath:
        newplot.savefig(saveFigPath)            
    return a,b

def make_sim_RB_data(gs,ExpRBData,seed = None):
    if seed != None:
        np.random.seed(seed)
    DS = GST.DataSet(spamLabels=['plus','minus'])
    RBKeys = ExpRBData.keys()
    for key in RBKeys:
        N = sum(ExpRBData[key].values())
        pp = gs.pr('plus',key,clipTo=(0,1))
        Np = np.random.binomial(N,pp)
        Nm = N - Np
        DS.add_count_dict(key,{'plus':Np,'minus':Nm})
    DS.done_adding_data()
    return DS

def make_sim_rb_data_perfect(gs,ExpRBData,N=1e6):
    RBKeys = ExpRBData.keys()
    DS = GST.generate_fake_data(gs,RBKeys,N,sampleError='none')
    return DS