from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for generating data with which to evaluate germ sets"""

import numpy as _np

import pygsti
import pygsti.algorithms as alg
import pygsti.construction as constr
import pygsti.objects as obj


def simulate_convergence(germs, prepFiducials, effectFiducials, targetGS,
                         randStr=1e-2, numPertGS=5, maxLengthsPower=8,
                         clickNums=32, numRuns=7, seed=None, randState=None,
                         gaugeOptRatio=1e-3):
    if not isinstance(clickNums, list):
        clickNums = [clickNums]
    if randState is None:
        randState = _np.random.RandomState(seed)

    perturbedGatesets = [targetGS.randomize_with_unitary(scale=randStr,
                                                         randState=randState)
                         for n in range(numPertGS)]

    maxLengths = [0] + [2**n for n in range(maxLengthsPower + 1)]

    expList = constr.make_lsgst_experiment_list(targetGS.gates.keys(),
                                                prepFiducials, effectFiducials,
                                                germs, maxLengths)

    errorDict = {}
    resultDict = {}

    for trueGatesetNum, trueGateset in enumerate(perturbedGatesets):
        for numClicks in clickNums:
            for run in range(numRuns):
                success = False
                failCount = 0
                while not success and failCount < 10:
                    try:
                        ds = constr.generate_fake_data(trueGateset, expList,
                                                       nSamples=numClicks,
                                                       sampleError="binomial",
                                                       randState=randState)

                        result = pygsti.do_long_sequence_gst(
                            ds, targetGS, prepFiducials, effectFiducials,
                            germs, maxLengths, gaugeOptRatio=gaugeOptRatio)

                        errors = [(trueGateset
                                   .frobeniusdist(
                                       alg.gaugeopt_to_target(
                                           estimate, trueGateset,
                                           itemWeights={'spam': 0.0}),
                                       spamWeight=0.0), L)
                                  for estimate, L in 
                                  zip(result.gatesets['iteration estimates'][1:],
                                      result.parameters['max length list'][1:])]

                        resultDict[trueGatesetNum, numClicks, run] = result
                        errorDict[trueGatesetNum, numClicks, run] = errors

                        success = True
                    except Exception as e:
                        failCount += 1
                        if failCount == 10:
                            raise e
                        print(e)

    return obj.GermSetEval(germset=germs, gatesets=perturbedGatesets,
                           resultDict=resultDict, errorDict=errorDict)
