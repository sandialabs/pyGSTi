import GST
import numpy as np
import numpy.random as _rndm
import matplotlib
from matplotlib import pyplot
from GSTCommons import Std1Q_XYI
import GSTCommons.Analyze_WholeGermPowers as Analyze

def makeBootstrapDataset(inputDataSet,generationMethod,inputGateSet=None,seed=None,spamLabels=['plus','minus']):#,GSTMethod="MLE",constrainToTP=False,)
    if generationMethod not in ['nonparametric', 'parametric']:
        raise ValueError("generationMethod must be 'parametric' or 'nonparametric'!")
    if seed is not None:
        np.random.seed(seed)
    if inputGateSet is None:
        if generationMethod == 'nonparametric':
            print "Generating non-parametric dataset."
        elif generationMethod == 'parametric':
            raise ValueError("Cannot generate parametric data without input gateset!")
    else:
        if generationMethod == 'parametric':
            print "Generating parametric dataset."
        elif generationMethod == 'nonparametric':
            raise ValueError("Input gateset provided, but nonparametric data demanded; must reconcile inconsistency!")
#    if generationMethod == 'nonparametric':
#        return GST.generateFakeData(inputDataSet,inputDataSet.keys())
    simDS = GST.DataSet(spamLabels=spamLabels)
    gateStringList = inputDataSet.keys()
    for s in gateStringList:
        nSamples = np.sum([inputDataSet[s][key] for key in spamLabels])
        if generationMethod == 'parametric':
            ps = inputGateSet.Probs(s) # New version of old line.
        elif generationMethod == 'nonparametric':
            fs = inputDataSet[s].asDict()
            ps = {key:fs[key] / float(nSamples) for key in fs.keys()}
        counts = {}
        pList = np.array([np.clip(ps[spamLabel],0,1) for spamLabel in spamLabels])#Truncate before normalization; bad extremal values shouldn't screw up not-bad values, yes?
        pList = pList / sum(pList)
        countsArray = np.random.multinomial(nSamples, pList, 1)
        for i,spamLabel in enumerate(spamLabels):
            counts[spamLabel] = countsArray[0,i]
        simDS.addCountDict(s, counts)
    simDS.doneAddingData()
    return simDS

def makeBootstrapGatesets(numIterations,inputDataSet,generationMethod,inputGateSet=None,gs_target=None,startSeed=0,
                          spamLabels=['plus','minus'],GSTMethod="MLE",constrainToTP=False,returnData=False,
                          fiducials=Std1Q_XYI.fiducials,germs=Std1Q_XYI.germs,maxLengths=None,**kwargs):
    if maxLengths == None:
        print "No maxLengths value specified; using [0,1,24,...,1024]"
        maxLengths = [0]+[2**k for k in range(10)]
    datasetList = []
    if (inputGateSet is None and gs_target is None) or (inputGateSet is not None and gs_target is not None):
        raise ValueError("Cannot supply both inputGateSet and gs_target!")
    if generationMethod == 'parametric':
        gs_target = inputGateSet
    for run in xrange(numIterations):
        print run,
        datasetList.append(
            makeBootstrapDataset(inputDataSet,generationMethod,inputGateSet=inputGateSet,seed=startSeed+run,spamLabels=spamLabels)
            )
    gatesetList = []
    for run in xrange(numIterations):
        print "Run", run
        result = Analyze.doMLEAnalysis(datasetList[run], gs_target, 
                                fiducials, fiducials, germs, maxLengths, makeReport=False, 
                                appendices=False, constrainToTP=constrainToTP, confidenceLevel=95)
        gatesetList.append(result['MLEGST gatesets'][-1])
    if not returnData:
        return gatesetList
    else:
        return gatesetList, datasetList

def gaugeOptimizeGSList(gsList,targetGateset,constrainToTP=True,gateMetric = 'frobenius', spamMetric = 'frobenius', plot=True):
    listOfBootStrapEstsNoOpt = list(gsList)
    numResamples = len(listOfBootStrapEstsNoOpt)
    ddof = 1
    SPAMMin = []
    SPAMMax = []
    SPAMMean = []

    gateMin = []
    gateMax = []
    gateMean = []
    for spWind, spW in enumerate(np.logspace(-4,0,13)):
        print spWind
        listOfBootStrapEstsNoOptG0toTargetVarSpam = []
        for gs in listOfBootStrapEstsNoOpt:
            listOfBootStrapEstsNoOptG0toTargetVarSpam.append(GST.optimizeGauge(gs,"target",returnAll=True,targetGateset=targetGateset,spamWeight=spW,constrainToTP=constrainToTP,targetGatesMetric=gateMetric,targetSpamMetric=spamMetric))

        GateSetGOtoTargetVarSpamVecArray = np.zeros([numResamples],dtype='object')
        for i in xrange(numResamples):
            GateSetGOtoTargetVarSpamVecArray[i] = listOfBootStrapEstsNoOptG0toTargetVarSpam[i][-1].toVector()

    #    gsAverageGOtoGeneratorVarSpam = gsGST.copy()
    #    gsStdevGOtoGeneratorVarSpam = gsGST.copy()

    #    gsAverageGOtoGeneratorVarSpam.fromVector(np.mean(GateSetGOtoGeneratorVarSpamVecArray))
    #    gsStdevGOtoGeneratorVarSpam.fromVector(
        gsStdevVec = np.std(GateSetGOtoTargetVarSpamVecArray,ddof=ddof)
        gsStdevVecSPAM = gsStdevVec[:8]
        gsStdevVecGates = gsStdevVec[8:]

        SPAMMin.append(np.min(gsStdevVecSPAM))
        SPAMMax.append(np.max(gsStdevVecSPAM))    
        SPAMMean.append(np.mean(gsStdevVecSPAM))

        gateMin.append(np.min(gsStdevVecGates))
        gateMax.append(np.max(gsStdevVecGates))
        gateMean.append(np.mean(gsStdevVecGates))    

    if plot:
        matplotlib.pyplot.loglog(np.logspace(-4,0,13),SPAMMean,'b-o')
        matplotlib.pyplot.loglog(np.logspace(-4,0,13),SPAMMin,'b--+')
        matplotlib.pyplot.loglog(np.logspace(-4,0,13),SPAMMax,'b--x')

        matplotlib.pyplot.loglog(np.logspace(-4,0,13),gateMean,'r-o')
        matplotlib.pyplot.loglog(np.logspace(-4,0,13),gateMin,'r--+')
        matplotlib.pyplot.loglog(np.logspace(-4,0,13),gateMax,'r--x')

        matplotlib.pyplot.xlabel('SPAM weight in gauge optimization')
        matplotlib.pyplot.ylabel('Per element error bar size')
        matplotlib.pyplot.title('Per element error bar size vs. ${\\tt spamWeight}$')
        matplotlib.pyplot.xlim(1e-4,1)
        matplotlib.pyplot.legend(['SPAM-mean','SPAM-min','SPAM-max','gates-mean','gates-min','gates-max'],bbox_to_anchor=(1.4, 1.))

    gateTimesSPAMMean = np.array(SPAMMean) * np.array(gateMean)

    bestSPAMWeight = np.logspace(-4,0,13)[np.argmin(np.array(SPAMMean)*np.array(gateMean))]
    print "Best SPAM weight is", bestSPAMWeight

    listOfBootStrapEstsG0toTargetSmallSpam = []
    for gs in listOfBootStrapEstsNoOpt:
        listOfBootStrapEstsG0toTargetSmallSpam.append(GST.optimizeGauge(gs,"target",returnAll=True,targetGateset=targetGateset,spamWeight=bestSPAMWeight,constrainToTP=constrainToTP,targetGatesMetric=gateMetric,targetSpamMetric=spamMetric)[2])
    return listOfBootStrapEstsG0toTargetSmallSpam
    
#For metrics that evaluate gateset with single scalar:
def gsStdev(gsFunc, gsEnsemble, ddof=1, **kwargs):
    return np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],ddof=ddof)

def gsMean(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return np.mean([gsFunc(gs, **kwargs) for gs in gsEnsemble])

#For metrics that evaluate gateset with scalar for each gate
def gsStdev1(gsFunc, gsEnsemble, ddof=1,axis=0, **kwargs):
    return np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],axis=axis,ddof=ddof)

def gsMean1(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return np.mean([gsFunc(gs,**kwargs) for gs in gsEnsemble],axis=axis)

def toVector(gs):
    return gs.toVector()

def toMeanGateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = gsList[i].toVector()
    output_gs = target_gs.copy()
    output_gs.fromVector(np.mean(gsVecArray))
    return output_gs

def toStdGateset(gsList,target_gs,ddof=1):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = gsList[i].toVector()
    output_gs = target_gs.copy()
    output_gs.fromVector(np.std(gsVecArray,ddof=ddof))
    return output_gs

def toRMSGateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = np.sqrt(gsList[i].toVector()**2)
    output_gs = target_gs.copy()
    output_gs.fromVector(np.mean(gsVecArray))
    return output_gs

def gatesetJTraceDistance(gs,gs_target,mxBasis="gm"):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.JTraceDistance(gs[gate],gs_target[gate],mxBasis=mxBasis)
#    print output
    return output

def gatesetProcessFidelity(gs,gs_target):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.JamiolkowskiOps.ProcessFidelity(gs[gate],gs_target[gate])
    return output

def gatesetDecomp_angle(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decomposeGateMatrix(gs[gate]).get('pi rotations',0)
    return output

def gatesetDecomp_decay_diag(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decomposeGateMatrix(gs[gate]).get('decay of diagonal rotation terms',0)
    return output

def gatesetDecomp_decay_offdiag(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decomposeGateMatrix(gs[gate]).get('decay of off diagonal rotation terms',0)
    return output

#def gatesetFidelity(gs,gs_target,mxBasis="gm"):
#    output = np.zeros(3,dtype=float)
#    for i, gate in enumerate(gs_target.keys()):
#        output[i] = GST.Fidelity(gs[gate],gs_target[gate])
#    return output

def gateSetDiamondNorm(gs,gs_target,mxBasis="gm"):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.GateOps.DiamondNorm(gs[gate],gs_target[gate],mxBasis=mxBasis)
    return output
    
def spamrameter(gs):
    return np.dot(gs.rhoVecs[0].T,gs.EVecs[0])[0,0]

