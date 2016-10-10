from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines canonical 1-qubit quantities used in Randomized Benchmarking"""

from ... import construction as _cnst
from . import rbobjs as _rbobjs

import numpy as _np
from collections import OrderedDict as _OrderedDict


gs_cliff_canonical = _cnst.build_gateset(
    [2],[('Q0',)], ['Gi','Gxp2','Gxp','Gxmp2','Gyp2','Gyp','Gymp2'], 
    [ "I(Q0)","X(pi/2,Q0)", "X(pi,Q0)", "X(-pi/2,Q0)",
      "Y(pi/2,Q0)", "Y(pi,Q0)", "Y(-pi/2,Q0)"],
    prepLabels=["rho0"], prepExpressions=["0"],
    effectLabels=["E0"], effectExpressions=["1"], 
    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )


#CliffD = {}
#Definitions taken from arXiv:1508.06676v1
#0-indexing used instead of 1-indexing

#CliffD[0] = ['Gi',]
#CliffD[1] = ['Gyp2','Gxp2']
#CliffD[2] = ['Gxmp2','Gymp2']
#CliffD[3] = ['Gxp',]
#CliffD[4] = ['Gymp2','Gxmp2']
#CliffD[5] = ['Gxp2','Gymp2']
#CliffD[6] = ['Gyp',]
#CliffD[7] = ['Gymp2','Gxp2']
#CliffD[8] = ['Gxp2','Gyp2']
#CliffD[9] = ['Gxp','Gyp']
#CliffD[10] = ['Gyp2','Gxmp2']
#CliffD[11] = ['Gxmp2','Gyp2']
#CliffD[12] = ['Gyp2','Gxp']
#CliffD[13] = ['Gxmp2']
#CliffD[14] = ['Gxp2','Gymp2','Gxmp2']
#CliffD[15] = ['Gymp2']
#CliffD[16] = ['Gxp2']
#CliffD[17] = ['Gxp2','Gyp2','Gxp2']
#CliffD[18] = ['Gymp2','Gxp']
#CliffD[19] = ['Gxp2','Gyp']
#CliffD[20] = ['Gxp2','Gymp2','Gxp2']
#CliffD[21] = ['Gyp2']
#CliffD[22] = ['Gxmp2','Gyp']
#CliffD[23] = ['Gxp2','Gyp2','Gxmp2']

clifford_to_canonical = _OrderedDict()
clifford_to_canonical["Gc0"] = ['Gi',]
clifford_to_canonical["Gc1"] = ['Gyp2','Gxp2']
clifford_to_canonical["Gc2"] = ['Gxmp2','Gymp2']
clifford_to_canonical["Gc3"] = ['Gxp',]
clifford_to_canonical["Gc4"] = ['Gymp2','Gxmp2']
clifford_to_canonical["Gc5"] = ['Gxp2','Gymp2']
clifford_to_canonical["Gc6"] = ['Gyp',]
clifford_to_canonical["Gc7"] = ['Gymp2','Gxp2']
clifford_to_canonical["Gc8"] = ['Gxp2','Gyp2']
clifford_to_canonical["Gc9"] = ['Gxp','Gyp']
clifford_to_canonical["Gc10"] = ['Gyp2','Gxmp2']
clifford_to_canonical["Gc11"] = ['Gxmp2','Gyp2']
clifford_to_canonical["Gc12"] = ['Gyp2','Gxp']
clifford_to_canonical["Gc13"] = ['Gxmp2']
clifford_to_canonical["Gc14"] = ['Gxp2','Gymp2','Gxmp2']
clifford_to_canonical["Gc15"] = ['Gymp2']
clifford_to_canonical["Gc16"] = ['Gxp2']
clifford_to_canonical["Gc17"] = ['Gxp2','Gyp2','Gxp2']
clifford_to_canonical["Gc18"] = ['Gymp2','Gxp']
clifford_to_canonical["Gc19"] = ['Gxp2','Gyp']
clifford_to_canonical["Gc20"] = ['Gxp2','Gymp2','Gxp2']
clifford_to_canonical["Gc21"] = ['Gyp2']
clifford_to_canonical["Gc22"] = ['Gxmp2','Gyp']
clifford_to_canonical["Gc23"] = ['Gxp2','Gyp2','Gxmp2']

gs_clifford_target = _cnst.build_alias_gateset(gs_cliff_canonical,
                                             clifford_to_canonical)

#cliff_group = MatrixGroup([ gs_cliff_generic["Gc%d"%i] for i in range(24)])
clifford_group = _rbobjs.MatrixGroup(gs_clifford_target.gates.values(),
                                  gs_clifford_target.gates.keys() )

#gs_cliff_generic = _cnst.build_gateset(
#    [2],[('Q0',)], [], [],
#    prepLabels=["rho0"], prepExpressions=["0"],
#    effectLabels=["E0"], effectExpressions=["1"],
#    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
#
#for i in list(CliffMatD.keys()):
#    gate_lbl = 'Gc'+str(i)
#    gs_cliff_generic.gates[gate_lbl] = \
#        _objs.FullyParameterizedGate(CliffMatD[i])


        

        

#CliffMatD = {}
#CliffMatInvD = {}
#for i in range(24):
#    CliffMatD[i] = gs_cliff_canonical.product(CliffD[i])
#    CliffMatInvD[i] = _np.linalg.matrix_power(CliffMatD[i],-1)

#gs_cliff_generic = _cnst.build_gateset(
#    [2],[('Q0',)], [], [],
#    prepLabels=["rho0"], prepExpressions=["0"],
#    effectLabels=["E0"], effectExpressions=["1"],
#    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
#
#for i in list(CliffMatD.keys()):
#    gate_lbl = 'Gc'+str(i)
#    gs_cliff_generic.gates[gate_lbl] = \
#        _objs.FullyParameterizedGate(CliffMatD[i])
#
#def makeCliffGroupTable():
#    CliffGroupTable = _np.zeros([24,24], dtype=int)
#    counter = 0
#    for i in range(24):
#        for j in range(24):
#            test = _np.dot(CliffMatD[j],CliffMatD[i])
#              #Dot in reverse order here for multiplication here because
#              # gates are applied left to right.
#            for k in range(24):
#                diff = _np.linalg.norm(test-CliffMatD[k])
#                if diff < 10**-10:
#                    CliffGroupTable[i,j]=k
#                    counter += 1
#                    break
#    assert counter==24**2, 'Logic Error!'
#    return CliffGroupTable
#
#CliffGroupTable = makeCliffGroupTable()
#
#CliffInvTable = {}
#for i in range(24):
#    for j in range(24):
#        if CliffGroupTable[i,j] == 0:
#            CliffInvTable[i] = j
            
######################################################
#Need to figure out what to do with these functions
######################################################

#def lookup_cliff_prod(i,j):
#    """
#    Auxiliary function for looking up the product of two Cliffords.
#    """
#    return CliffGroupTable[i,j]


    
# def rb_decay_WF_rate(dataset,avg_gates_per_cliff=None,showPlot=False,xlim=None,ylim=None,saveFigPath=None,printData=False,p0=[0.5,0.5,0.98]):
#     RBlengths = []
#     RBsuccesses = []
#     for key in list(dataset.keys()):
#         dataLine = dataset[key]
#         plus = dataLine['plus']
#         minus = dataLine['minus']
#         N = plus + minus
#         key_len = len(key)
#         RBlengths.append(key_len)
#         RBsuccessProb=1 - dataLine['plus']/float(N)
#         RBsuccesses.append(RBsuccessProb)
#         if dataLine['plus']/float(N) > 1:
#             print(key)
#         if printData:
#             print(key_len,RBsuccessProb)
#     results = _curve_fit(rb_decay_WF,RBlengths,RBsuccesses,p0=p0)
#     A,B,f = results[0]
#     cov = results[1]
#     if saveFigPath or showPlot:
#         newplot = _plt.figure()
#         newplotgca = newplot.gca()
#         newplotgca.plot(RBlengths,RBsuccesses,'.')
#         newplotgca.plot(range(max(RBlengths)),
#                         rb_decay_WF(_np.arange(max(RBlengths)),A,B,f),'+')
#         newplotgca.set_xlabel('RB sequence length (non-Clifford)')
#         newplotgca.set_ylabel('Success rate')
#         newplotgca.set_title('RB success')
#         if xlim:
#             _plt.xlim(xlim)
#         if ylim:
#             _plt.ylim(ylim)
#     if saveFigPath:
#         newplot.savefig(saveFigPath)
#     print "f (for gates) =",f
#     if avg_gates_per_cliff:
#         print "f (for Cliffords) = f^(avg. gates/Cliffords) =",f**avg_gates_per_cliff
#     return A,B,f,cov
