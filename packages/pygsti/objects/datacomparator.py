from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the DataComparator class used to compare multiple DataSets."""

import numpy as _np
import scipy as _scipy
from scipy import stats as _stats

from .multidataset import MultiDataSet as _MultiDataSet


#Define auxiliary functions.
def xlogy(x,y):
    if x == 0:
        return 0
    else:
        return x * _np.log(y)

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
    return _np.abs(dof - loglikelihoodRatioTest(alpha*nListList))

def loglikelihoodRatioTest(nListList):
    nListC = _np.sum(nListList,axis=0)
    pListC = nListC / _np.float(_np.sum(nListC))
    lC = loglikelihood(pListC,nListC)
    li_list = []
    for nList in nListList:
        pList = _np.array(nList) / _np.float(_np.sum(nList))
        li_list.append(loglikelihood(pList,nList))
    lS = _np.sum(li_list)
    return -2 * (lC - lS)

def pval(llrval,dof):
    return 1-_stats.chi2.cdf(llrval,dof)

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
    return _scipy.stats.chi2.isf((1-confidence_level)/strings,dof)

#Define the data_comparator class.  This object will store the p-values and log-likelihood ratio values for a pair
#of datasets.  It also contains methods for plotting both p-value histograms and log-likelihood ratio box plots.
class DataComparator():

    def __init__(self, dataset_list_or_multidataset,gatestrings = 'all',
                 gate_exclusions = None, gate_inclusions = None,
                 DS_names = None):
        
        if DS_names is not None:
            if len(DS_names) != len(dataset_list_or_multidataset):
                raise ValueError('Length of provided DS_names list must equal length of dataset_list_or_multidataset.')
            
        if isinstance(dataset_list_or_multidataset,list):
            dsList = dataset_list_or_multidataset    
            slIndex = dsList[0].slIndex
            slIndexListBool = [ds.slIndex==(slIndex) for ds in dsList]
            if not _np.all(slIndexListBool):
                raise ValueError('SPAM labels and order must be the same across datasets.')
            if gatestrings == 'all':
                gatestringList = dsList[0].keys()
                gatestringsListBool = [ds.keys()==gatestringList for ds in dsList]
                if not _np.all(gatestringsListBool):
                    raise ValueError('If gatestrings="all" is used, then datasets must contain identical gatestrings.  (They do not.)')
                gatestrings = gatestringList
        elif isinstance(dataset_list_or_multidataset,_MultiDataSet):
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
            nListList = _np.array([dataline.values() for dataline in datalineList])
            llrVals_and_strings.append([gatestring,loglikelihoodRatioTest(nListList)])
            temp_pvalue = pval(llrVals_and_strings[-1][1],dof) 
            pVals_and_strings.append([gatestring, temp_pvalue])

        #Set members (was a separate __init_function:
        #     def __init__(self,dataset_list_or_multidataset,pVals_and_strings,
        #                  llrVals_and_strings,gate_exclusions,gate_inclusions,
        #                  dof,DS_names=['DS0','DS1']):
        self.dataset_list_or_multidataset = dataset_list_or_multidataset
        self.pVals_and_strings = pVals_and_strings
        self.llrVals_and_strings = llrVals_and_strings
        self.gate_exclusions = gate_exclusions
        self.gate_inclusions = gate_inclusions
        self.pVals = _np.array(pVals_and_strings,dtype=object)[:,1]
        self.llrVals = _np.array(llrVals_and_strings,dtype=object)[:,1]
        self.pVals0 = str(len(self.pVals)-_np.count_nonzero(self.pVals))
        self.dof = dof
        self.num_strs = len(self.pVals)
        self.DS_names = DS_names

    def worst_strings(self,number):
        worst_strings = _np.array(self.pVals_and_strings,dtype='object')
        worst_strings = sorted(worst_strings, key=lambda x: x[1])[:number]
        return worst_strings
            
    def report(self,confidence_level=0.95):
        single_string_thresh = find_thresh(confidence_level,self.num_strs,self.dof)
        number_of_single_thresh_violators = _np.sum(_np.where(self.llrVals>single_string_thresh,1,0))
        composite_thresh = find_thresh(confidence_level,1,self.num_strs*self.dof)
        composite_score = _np.sum(self.llrVals)
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
        single_thresh_violator_locs = _np.nonzero(_np.where(self.llrVals>single_string_thresh,1,0))[0]
        self.alpha_dict = {}
        if isinstance(self.dataset_list_or_multidataset,list):
            dsList = [DS.copy_nonstatic() for DS in self.dataset_list_or_multidataset]
        elif isinstance(self.dataset_list_or_multidataset,_MultiDataSet):
            dsList = [self.dataset_list_or_multidataset[key].copy() for key in self.dataset_list_or_multidataset.keys()]
        for violator_loc in single_thresh_violator_locs:
            gatestring = self.llrVals_and_strings[violator_loc][0]
            llr = self.llrVals_and_strings[violator_loc][1]
            datalineList = [ds[gatestring] for ds in dsList]
            nListList = _np.array([dataline.values() for dataline in datalineList])
            self.alpha_dict[gatestring] = target_score / llr
            print('Rescaling counts for string '+str(gatestring)+' by '+str(self.alpha_dict[gatestring]))
            print('|target score - new score| = '+str(loglikelihoodRatioTestObj(self.alpha_dict[gatestring],nListList,target_score)))
            for ds in dsList:
                for outcome in ds.slIndex.keys():
                    ds[gatestring][outcome] = self.alpha_dict[gatestring] * ds[gatestring][outcome]
        self.rectified_datasets = dsList
