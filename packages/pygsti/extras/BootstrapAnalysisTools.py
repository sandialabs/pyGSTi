import GST
import numpy as np
import matplotlib
from matplotlib import pyplot
from GSTCommons import Std1Q_XYI
import GSTCommons.Analyze_WholeGermPowers as Analyze

def make_bootstrap_dataset(inputDataSet,generationMethod,inputGateSet=None,seed=None,spamLabels=['plus','minus']):#,GSTMethod="MLE",constrainToTP=False,)
    if generationMethod not in ['nonparametric', 'parametric']:
        raise ValueError("generationMethod must be 'parametric' or 'nonparametric'!")
    rndm = _np.random.RandomState(seed)
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
#        return GST.generate_fake_data(inputDataSet,inputDataSet.keys())
    simDS = GST.DataSet(spamLabels=spamLabels)
    gatestring_list = inputDataSet.keys()
    for s in gatestring_list:
        nSamples = np.sum([inputDataSet[s][key] for key in spamLabels])
        if generationMethod == 'parametric':
            ps = inputGateSet.probs(s) # New version of old line.
        elif generationMethod == 'nonparametric':
            fs = inputDataSet[s].as_dict()
            ps = {key:fs[key] / float(nSamples) for key in fs.keys()}
        counts = {}
        pList = np.array([np.clip(ps[spamLabel],0,1) for spamLabel in spamLabels])#Truncate before normalization; bad extremal values shouldn't screw up not-bad values, yes?
        pList = pList / sum(pList)
        countsArray = rndm.multinomial(nSamples, pList, 1)
        for i,spamLabel in enumerate(spamLabels):
            counts[spamLabel] = countsArray[0,i]
        simDS.add_count_dict(s, counts)
    simDS.done_adding_data()
    return simDS

def make_bootstrap_gatesets(numIterations,inputDataSet,generationMethod,inputGateSet=None,gs_target=None,startSeed=0,
                          spamLabels=['plus','minus'],GSTMethod="MLE",constrainToTP=True,lsgstLists=None,returnData=False,
                          fiducialPrep=Std1Q_XYI.fiducials,fiducialMeasure=Std1Q_XYI.fiducials,germs=Std1Q_XYI.germs,maxLengths=None,verbosity=2,**kwargs):
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
            make_bootstrap_dataset(inputDataSet,generationMethod,inputGateSet=inputGateSet,seed=startSeed+run,spamLabels=spamLabels)
            )
    gatesetList = []
    for run in xrange(numIterations):
        print "Run", run
        result = Analyze.doMLEAnalysis(datasetList[run],gs_target,fiducialPrep,fiducialMeasure,germs,maxLengths,constrainToTP=constrainToTP,lsgstLists=lsgstLists,advancedOptions={'verbosity':verbosity})
#        result = Analyze.doMLEAnalysis(datasetList[run], gs_target, 
#                                fiducials, fiducials, germs, maxLengths, makeReport=False, 
#                                appendices=False, constrainToTP=constrainToTP, confidenceLevel=95)
        gatesetList.append(result.gatesets[-1])
    if not returnData:
        return gatesetList
    else:
        return gatesetList, datasetList

def gauge_optimize_gs_list(gsList,targetGateset,constrainToTP=True,gateMetric = 'frobenius', spamMetric = 'frobenius', plot=True):
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
            listOfBootStrapEstsNoOptG0toTargetVarSpam.append(GST.optimize_gauge(gs,"target",returnAll=True,targetGateset=targetGateset,spamWeight=spW,constrainToTP=constrainToTP,targetGatesMetric=gateMetric,targetSpamMetric=spamMetric))

        GateSetGOtoTargetVarSpamVecArray = np.zeros([numResamples],dtype='object')
        for i in xrange(numResamples):
            GateSetGOtoTargetVarSpamVecArray[i] = listOfBootStrapEstsNoOptG0toTargetVarSpam[i][-1].to_vector()

    #    gsAverageGOtoGeneratorVarSpam = gsGST.copy()
    #    gsStdevGOtoGeneratorVarSpam = gsGST.copy()

    #    gsAverageGOtoGeneratorVarSpam.from_vector(np.mean(GateSetGOtoGeneratorVarSpamVecArray))
    #    gsStdevGOtoGeneratorVarSpam.from_vector(
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
        listOfBootStrapEstsG0toTargetSmallSpam.append(GST.optimize_gauge(gs,"target",returnAll=True,targetGateset=targetGateset,spamWeight=bestSPAMWeight,constrainToTP=constrainToTP,targetGatesMetric=gateMetric,targetSpamMetric=spamMetric)[2])
    return listOfBootStrapEstsG0toTargetSmallSpam
    
#For metrics that evaluate gateset with single scalar:
def gs_stdev(gsFunc, gsEnsemble, ddof=1, **kwargs):
    return np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],ddof=ddof)

def gs_mean(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return np.mean([gsFunc(gs, **kwargs) for gs in gsEnsemble])

#For metrics that evaluate gateset with scalar for each gate
def gs_stdev1(gsFunc, gsEnsemble, ddof=1,axis=0, **kwargs):
    return np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],axis=axis,ddof=ddof)

def gs_mean1(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return np.mean([gsFunc(gs,**kwargs) for gs in gsEnsemble],axis=axis)

def to_vector(gs):
    return gs.to_vector()

def to_mean_gateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(np.mean(gsVecArray))
    return output_gs

def to_std_gateset(gsList,target_gs,ddof=1):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(np.std(gsVecArray,ddof=ddof))
    return output_gs

def to_rms_gateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = np.zeros([numResamples],dtype='object')
    for i in xrange(numResamples):
        gsVecArray[i] = np.sqrt(gsList[i].to_vector()**2)
    output_gs = target_gs.copy()
    output_gs.from_vector(np.mean(gsVecArray))
    return output_gs

def gateset_jtracedist(gs,gs_target,mxBasis="gm"):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.jtracedist(gs[gate],gs_target[gate],mxBasis=mxBasis)
#    print output
    return output

def gateset_process_fidelity(gs,gs_target):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.JamiolkowskiOps.process_fidelity(gs[gate],gs_target[gate])
    return output

def gateset_decomp_angle(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decompose_gate_matrix(gs[gate]).get('pi rotations',0)
    return output

def gateset_decomp_decay_diag(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decompose_gate_matrix(gs[gate]).get('decay of diagonal rotation terms',0)
    return output

def gateset_decomp_decay_offdiag(gs):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.keys()):
        output[i] = GST.GateOps.decompose_gate_matrix(gs[gate]).get('decay of off diagonal rotation terms',0)
    return output

#def gateset_fidelity(gs,gs_target,mxBasis="gm"):
#    output = np.zeros(3,dtype=float)
#    for i, gate in enumerate(gs_target.keys()):
#        output[i] = GST.fidelity(gs[gate],gs_target[gate])
#    return output

def gate_set_diamond_norm(gs,gs_target,mxBasis="gm"):
    output = np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.keys()):
        output[i] = GST.GateOps.diamonddist(gs[gate],gs_target[gate],mxBasis=mxBasis)
    return output
    
def spamrameter(gs):
    return np.dot(gs.rhoVecs[0].T,gs.EVecs[0])[0,0]

