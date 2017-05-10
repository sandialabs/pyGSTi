#Import relevant namespaces
from __future__ import division, print_function
import numpy as np
from scipy import stats
from matplotlib import pyplot
import scipy
from . import plotting
from .. import objects
from .. import construction

#Define auxiliary functions.
def xlogy(x,y):
    if x == 0:
        return 0
    else:
        return x * np.log(y)

def likelihood(pList,nList):
    output = 1.
    for i, pVal in enumerate(pList):
        output *= pVal**nList[i]
    return output

def loglikelihood(pList,nList):
    output = 0.
    for i, pVal in enumerate(pList):
        output += xlogy(nList[i],pVal)
    return output

def loglikelihoodRatioTestObj(alpha,nListList,dof):
    return np.abs(dof - loglikelihoodRatioTest(alpha*nListList))

def loglikelihoodRatioTest(nListList):
    nListC = np.sum(nListList,axis=0)
    pListC = nListC / np.float(np.sum(nListC))
    lC = loglikelihood(pListC,nListC)
    li_list = []
    for nList in nListList:
        pList = np.array(nList) / np.float(np.sum(nList))
        li_list.append(loglikelihood(pList,nList))
    lS = np.sum(li_list)
    return -2 * (lC - lS)

def pval(llrval,dof):
    return 1-stats.chi2.cdf(llrval,dof)

def is_gatestring_allowed_by_exclusion(gate_exclusions,gatestring):
    for gate in gate_exclusions:
        if gate in gatestring:
            return False
    return True

def is_gatestring_allowed_by_inclusion(gate_inclusions,gatestring):
    for gate in gate_inclusions:
        if gate not in gatestring:
            return False
    return True

def find_thresh(confidence_level,strings,dof):
    return scipy.stats.chi2.isf((1-confidence_level)/strings,dof)

#Define the data_comparator class.  This object will store the p-values and log-likelihood ratio values for a pair
#of datasets.  It also contains methods for plotting both p-value histograms and log-likelihood ratio box plots.
class data_comparator():
    def __init__(self,dataset_list_or_multidataset,pVals_and_strings,llrVals_and_strings,gate_exclusions,gate_inclusions,dof,DS_names=['DS0','DS1']):
        self.dataset_list_or_multidataset = dataset_list_or_multidataset
        self.pVals_and_strings = pVals_and_strings
        self.llrVals_and_strings = llrVals_and_strings
        self.gate_exclusions = gate_exclusions
        self.gate_inclusions = gate_inclusions
        self.pVals = np.array(pVals_and_strings,dtype=object)[:,1]
        self.llrVals = np.array(llrVals_and_strings,dtype=object)[:,1]
        self.pVals0 = str(len(self.pVals)-np.count_nonzero(self.pVals))
        self.dof = dof
        self.num_strs = len(self.pVals)
    def worst_strings(self,number):
        worst_strings = np.array(self.pVals_and_strings,dtype='object')
        worst_strings = sorted(worst_strings, key=lambda x: x[1])[:number]
        return worst_strings
    def hist_p_plot(self,bins=np.logspace(-10,0,50),frequency=True,log=True,datasetnames=['DS0','DS1'],filename=''):
        if frequency:
            weights = np.ones(self.pVals.shape) / len(self.pVals)
            hist_data = pyplot.hist(self.pVals,bins=bins,log=log,weights = weights)
            pyplot.plot(bins,scipy.stats.chi2.pdf(scipy.stats.chi2.isf(bins,self.dof),self.dof),'--',linewidth=4,label='No-change prediction')
            pyplot.ylabel('Relative frequency')
        else:
            hist_data = pyplot.hist(self.pVals,bins=bins,log=log)
            pyplot.plot(bins,hist_data[0][-1]*scipy.stats.chi2.pdf(scipy.stats.chi2.isf(bins,self.dof),self.dof),'--',linewidth=4,label='No-change prediction')
            pyplot.ylabel('Number of occurrences')            
        pyplot.legend()
        if log:
            pyplot.gca().set_xscale('log')
        pyplot.xlabel('p-value')
        title = 'p-value histogram for experimental coins;'
        if self.gate_exclusions:
            title += ' '+str(self.gate_exclusions)+' excluded'
            if self.gate_inclusions:
                title += ';'
        if self.gate_inclusions:
            title += ' '+str(self.gate_inclusions)+' included'
        title += '\nComparing datasets '+str(datasetnames)
        title += ' p=0 '+str(self.pVals0)+' times; '+str(len(self.pVals))+' total sequences'
        pyplot.title(title)
        if filename:
            pyplot.savefig(filename)
    def hist_logl_plot(self,bins=None,ylim=None,log=False,datasetnames=['DS0','DS1'],filename=''):
        if bins is None:
            bins = len(self.llrVals)
        pyplot.hist(self.llrVals,bins=bins,log=log,cumulative=True,normed=True,histtype='step')
        if log:
            pyplot.gca().set_xscale('log')
        pyplot.xlabel('log-likelihood')
        pyplot.ylabel('Cumulative frequency')
        title = 'Cumulative log-likelihood ratio histogram for experimental coins;'
        if self.gate_exclusions:
            title += ' '+str(self.gate_exclusions)+' excluded'
            if self.gate_inclusions:
                title += ';'
        if self.gate_inclusions:
            title += ' '+str(self.gate_inclusions)+' included'
        title += '\nComparing datasets '+str(datasetnames)
        title += ' p=0 '+str(self.pVals0)+' times; '+str(len(self.pVals))+' total sequences'
        pyplot.title(title)
        if filename:
            pyplot.savefig(filename)
    def box_plot(self,germs,prep_fids,effect_fids,max_lengths,linlg_pcntle,prec,title_text,save_to=None): 
        prepStrs, effectStrs = prep_fids, effect_fids
        xvals = max_lengths
        yvals = germs
        xy_gatestring_dict = {}
        for x in xvals:
            for y in yvals:
                xy_gatestring_dict[(x,y)] = construction.repeat_with_max_length(y,x)
        llrVals_and_strings_dict = dict(self.llrVals_and_strings)
        def mx_fn(gateStr,x,y):
            mx = np.empty((len(effectStrs),len(prepStrs)),'d')
            for i,f1 in enumerate(effectStrs):
                for j,f2 in enumerate(prepStrs):
                    seq = f2 + gateStr + f1
                    mx[i,j] = llrVals_and_strings_dict[seq]
            return mx
        xvals,yvals,subMxs,n_boxes,dof = plotting._computeSubMxs(xvals,yvals,xy_gatestring_dict,mx_fn,False)
        stdcmap = plotting.StdColormapFactory('linlog', n_boxes=n_boxes, linlg_pcntle=linlg_pcntle, dof=dof)
        plotting.generate_boxplot( xvals, yvals, xy_gatestring_dict, subMxs, stdcmap, "L","germs", prec=prec, title=title_text, save_to=save_to)
    def report(self,confidence_level=0.95):
        single_string_thresh = find_thresh(confidence_level,self.num_strs,self.dof)
        number_of_single_thresh_violators = np.sum(np.where(self.llrVals>single_string_thresh,1,0))
        composite_thresh = find_thresh(confidence_level,1,self.num_strs*self.dof)
        composite_score = np.sum(self.llrVals)
        print("Consistency report- datasets are inconsistent at given confidence level if EITHER of the following scores report inconsistency.")
        print()
        print("Threshold for individual gatestring scores is {0}".format(single_string_thresh))
        if number_of_single_thresh_violators > 0:
            print("As measured by worst-performing gate strings, data sets are INCONSISTENT at the {0}% confidence level.".format(confidence_level*100))
            print("{0} gate string(s) have loglikelihood scores greater than the threshold.".format(number_of_single_thresh_violators))
        else:
            print("As measured by worst-performing gate strings, data sets are CONSISTENT at the {0}% confidence level.".format(confidence_level*100))
            print("{0} gate string(s) have loglikelihood scores greater than the threshold.".format(number_of_single_thresh_violators))
        print()
        print("Threshold for sum of gatestring scores is {0}.".format(composite_thresh))
        if composite_score > composite_thresh:
            print("As measured by sum of gatestring scores, data sets are INCONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        else:
            print("As measured by sum of gatestring scores, data sets are CONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        print("Total loglikelihood is {0}".format(composite_score))
    def rectify_datasets(self,confidence_level=0.95,target_score='dof',x0=0.5,method='Nelder-Mead'):
        if target_score == 'dof':
            target_score = self.dof
        single_string_thresh = find_thresh(confidence_level,self.num_strs,self.dof)
        single_thresh_violator_locs = np.nonzero(np.where(self.llrVals>single_string_thresh,1,0))[0]
        self.alpha_dict = {}
        if isinstance(self.dataset_list_or_multidataset,list):
            dsList = [DS.copy_nonstatic() for DS in self.dataset_list_or_multidataset]
        elif isinstance(self.dataset_list_or_multidataset,objects.multidataset.MultiDataSet):
            dsList = [self.dataset_list_or_multidataset[key].copy() for key in self.dataset_list_or_multidataset.keys()]
        for violator_loc in single_thresh_violator_locs:
            gatestring = self.llrVals_and_strings[violator_loc][0]
            llr = self.llrVals_and_strings[violator_loc][1]
            datalineList = [ds[gatestring] for ds in dsList]
            nListList = np.array([dataline.values() for dataline in datalineList])
            self.alpha_dict[gatestring] = target_score / llr
            print('Rescaling counts for string '+str(gatestring)+' by '+str(self.alpha_dict[gatestring]))
            print('|target score - new score| = '+str(loglikelihoodRatioTestObj(self.alpha_dict[gatestring],nListList,target_score)))
            for ds in dsList:
                for outcome in ds.slIndex.keys():
                    ds[gatestring][outcome] = self.alpha_dict[gatestring] * ds[gatestring][outcome]
        self.rectified_datasets = dsList

#Define the function to compare two datasets for change and store the results in a "data_comparator" object.

#def data_comparator_test(dataset_list_or_multidataset,gatestrings = 'all',gate_exclusions = None, gate_inclusions = None, DS0_name = None, DS1_name=None):
def make_data_comparator(dataset_list_or_multidataset,gatestrings = 'all',gate_exclusions = None, gate_inclusions = None, DS_names = None):
    if DS_names is not None:
        if len(DS_names) != len(dataset_list_or_multidataset):
            raise ValueError('Length of provided DS_names list must equal length of dataset_list_or_multidataset.')
    if isinstance(dataset_list_or_multidataset,list):
        dsList = dataset_list_or_multidataset    
        slIndex = dsList[0].slIndex
        slIndexListBool = [ds.slIndex==(slIndex) for ds in dsList]
        if not np.all(slIndexListBool):
            raise ValueError('SPAM labels and order must be the same across datasets.')
        if gatestrings == 'all':
            gatestringList = dsList[0].keys()
            gatestringsListBool = [ds.keys()==gatestringList for ds in dsList]
            if not np.all(gatestringsListBool):
                raise ValueError('If gatestrings="all" is used, then datasets must contain identical gatestrings.  (They do not.)')
            gatestrings = gatestringList
    elif isinstance(dataset_list_or_multidataset,objects.multidataset.MultiDataSet):
        dsList = [dataset_list_or_multidataset[key] for key in dataset_list_or_multidataset.keys()]
        gatestrings = dsList[0].keys()
        if DS_names is None:
            DS_names = dataset_list_or_multidataset.keys()
    if gate_exclusions is not None:
        gatestrings_exc_temp = []
        for gatestring in gatestrings:
            if is_gatestring_allowed_by_exclusion(gate_exclusions,gatestring):
                gatestrings_exc_temp.append(gatestring)
        gatestrings = list(gatestrings_exc_temp)
    if gate_inclusions is not None:
        gatestrings_inc_temp = []
        for gatestring in gatestrings:
            if is_gatestring_allowed_by_inclusion(gate_inclusions,gatestring):
                gatestrings_inc_temp.append(gatestring)
        gatestrings = list(gatestrings_inc_temp)
    llrVals_and_strings = []
    pVals_and_strings = []
    dof = (len(dsList) - 1) * (len(dsList[0].slIndex) - 1)
    for gatestring in gatestrings:
        datalineList = [ds[gatestring] for ds in dsList]
        nListList = np.array([dataline.values() for dataline in datalineList])
        llrVals_and_strings.append([gatestring,loglikelihoodRatioTest(nListList)])
        temp_pvalue = pval(llrVals_and_strings[-1][1],dof) 
        pVals_and_strings.append([gatestring, temp_pvalue])
    return data_comparator(dataset_list_or_multidataset,pVals_and_strings,llrVals_and_strings,gate_exclusions,gate_inclusions,dof,DS_names)