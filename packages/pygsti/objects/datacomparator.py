""" Defines the DataComparator class used to compare multiple DataSets."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy as _scipy
import copy as _copy
from scipy import stats as _stats
import collections as _collections
from .multidataset import MultiDataSet as _MultiDataSet
from .hypothesistest import HypothesisTest as _HypothesisTest

#Define auxiliary functions.
def xlogy(x,y):
    if x == 0:
        return 0
    else:
        return x * _np.log(y)

def likelihood(pList,nList):
    """
    Todo
    """
    output = 1.
    for i, pVal in enumerate(pList):
        output *= pVal**nList[i]
    return output

def loglikelihood(pList,nList):
    """
    Todo
    """
    output = 0.
    for i, pVal in enumerate(pList):
        output += xlogy(nList[i],pVal)
    return output

def loglikelihoodRatioTestObj(alpha,nListList,dof):
    """
    Todo
    """
    return _np.abs(dof - loglikelihoodRatioTest(alpha*nListList))

def loglikelihoodRatioTest(nListList):
    """
    Todo
    """
    nListC = _np.sum(nListList,axis=0)
    pListC = nListC / _np.float(_np.sum(nListC))
    lC = loglikelihood(pListC,nListC)
    li_list = []
    for nList in nListList:
        pList = _np.array(nList) / _np.float(_np.sum(nList))
        li_list.append(loglikelihood(pList,nList))
    lS = _np.sum(li_list)
    return -2 * (lC - lS)

def pval(llrval, dof):
    """
    Todo
    """
    return 1 - _stats.chi2.cdf(llrval, dof)

def is_gatestring_allowed_by_exclusion(gate_exclusions,gatestring):
    """
    Todo
    """
    for gate in gate_exclusions:
        if gate in gatestring:
            return False
    return True

def is_gatestring_allowed_by_inclusion(gate_inclusions,gatestring):
    """
    Todo
    """
    if len(gatestring) == 0: return True # always include the empty string
    for gate in gate_inclusions:
        if gate in gatestring:
            return True
    return False

def find_thresh(confidence_level, strings, dof):
    """
    Todo
    """
    return _scipy.stats.chi2.isf((1-confidence_level)/strings,dof)

def tvd(gatestring, ds0, ds1):
    """
    Todo
    """
#    assert set(ds0.slIndex.keys()) == set(ds1.slIndex.keys())
#    outcomes = ds0.slIndex.keys()
    assert set(ds0.olIndex.keys()) == set(ds1.olIndex.keys())
    outcomes = ds0.olIndex.keys()
    line0 = ds0[gatestring]
    line1 = ds1[gatestring]
    N0 = line0.total#()
    N1 = line1.total#()
    return 0.5 * _np.sum(_np.abs(line0[outcome]/N0 - line1[outcome]/N1) for outcome in outcomes)

#Define the data_comparator class. 
class DataComparator():
    """
    An object that stores the p-values and log-likelihood ratio values from a comparison of
    a pair of datasets, and methods to: perform a hypothesis test to decide which value are,
    and plotting of both p-value histograms, and log-likelihood ratio box plots.
    """
    def __init__(self, dataset_list_or_multidataset, gatestrings = 'all',
                 gate_exclusions = None, gate_inclusions = None, DS_names = None):
        """
        Todo
        """      
        if DS_names is not None:
            if len(DS_names) != len(dataset_list_or_multidataset):
                raise ValueError('Length of provided DS_names list must equal length of dataset_list_or_multidataset.')
            
        if isinstance(dataset_list_or_multidataset,list):
            dsList = dataset_list_or_multidataset    
            olIndex = dsList[0].olIndex
            olIndexListBool = [ds.olIndex==(olIndex) for ds in dsList]
            if not _np.all(olIndexListBool):
                raise ValueError('Outcomes labels and order must be the same across datasets.')
            if gatestrings == 'all':
                gatestringList = dsList[0].keys()
                gatestringsListBool = [ds.keys()==gatestringList for ds in dsList]
                if not _np.all(gatestringsListBool):
                    raise ValueError('If gatestrings="all" is used, then datasets must contain identical gatestrings. (They do not.)')
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
            
        llrs = {}
        pVals = {}
        dof = (len(dsList) - 1) * (len(dsList[0].olIndex) - 1)
        
        for gatestring in gatestrings:
            datalineList = [ds[gatestring] for ds in dsList]
            nListList = _np.array([list(dataline.allcounts.values()) for dataline in datalineList])
            llrs[gatestring] = loglikelihoodRatioTest(nListList)
            pVals[gatestring] =  pval(llrs[gatestring],dof) 

        self.dataset_list_or_multidataset = dataset_list_or_multidataset
        self.pVals = pVals
        self.llrs = llrs
        self.gate_exclusions = gate_exclusions
        self.gate_inclusions = gate_inclusions
        self.pVals0 = str(len(self.pVals)-_np.count_nonzero(list(self.pVals.values())))
        self.dof = dof
        self.num_strs = len(self.pVals)
        self.DS_names = DS_names

        self.composite_llr = _np.sum(list(self.llrs.values())  )     
        self.composite_pVal = pval(self.composite_llr, self.num_strs*self.dof) 
        
        # Convert the composite LLR to a signed standard deviations.
        k = self.num_strs*self.dof
        self.composite_nsigma = (self.composite_llr - k) / _np.sqrt(2*k)

    # Todo : fix this
    # def worst_strings(self, number):
    #     """
    #     Returns the `number` strings with the smallest p-values.
    #     """
    #     worst_strings = _np.array(self.pVals_and_strings,dtype='object')
    #     worst_strings = sorted(worst_strings, key=lambda x: x[1])[:number]
    #     return worst_strings

    def implement(self, significance=0.05, per_sequence_correction='Hochberg', 
                  composite_test_weighting=0.5,  pass_alpha=True, verbosity=1):

        self.significance = significance
        assert(composite_test_weighting <= 1. or composite_test_weighting >= 0.), "The weighting on the composite test must be between 0 and 1!"
        

        if verbosity >= 2:
            print("Implementing {}% significance statistical hypothesis testing...".format(self.significance*100),end='')

       

        gatestrings = tuple(self.pVals.keys())
        hypotheses = ('composite', gatestrings)
        weighting = {}
        weighting['composite'] = composite_test_weighting
        weighting[gatestrings] = 1 - composite_test_weighting
        
        if pass_alpha: passing_graph = 'Holms'
        else: passing_graph = 'none'

        hypotest = _HypothesisTest(hypotheses, significance=significance, weighting=weighting, 
                                   passing_graph=passing_graph, local_corrections=per_sequence_correction)
        extended_pVals_dict = _copy.copy(self.pVals)
        extended_pVals_dict['composite'] = self.composite_pVal
        hypotest.add_pvalues(extended_pVals_dict)
        hypotest.implement()
        self.results = hypotest

        if composite_test_weighting == 0:
            self.composite_thresh = _np.inf
        else:
            # Tim doesn't really get this function, and will change it to something more transparent at
            # some point.
            self.composite_thresh = find_thresh(1-significance, 1./composite_test_weighting, self.num_strs*self.dof)

        temp_hypothesis_rejected_dict = _copy.copy(hypotest.hypothesis_rejected)
        self.inconsistent_datasets_detected = any(list(temp_hypothesis_rejected_dict.values()))
        del temp_hypothesis_rejected_dict['composite']            
        self.number_of_significant_sequences = _np.sum(list(temp_hypothesis_rejected_dict.values()))

        if verbosity >= 2:
            print("complete.")

        if len(self.dataset_list_or_multidataset) == 2:
            self.compute_TVDs(verbosity=verbosity)
        
        if verbosity >= 2:
            print("\n--- Results ---\n")

        if verbosity >= 1:
            if self.inconsistent_datasets_detected:
                print("The datasets are INCONSISTENT at {0}% significance.\n".format(self.significance*100))
                print("  Details:")
                print("  - The number of sequences with data that is inconsistent is {0}".format(self.number_of_significant_sequences))
                print("  - Using the composite log-likelihood ratio test, the significance of the inconsistency between datasets is {} standard deviations.".format(self.composite_nsigma))  
            else:
                print("Statistical hypothesis tests did NOT find inconsistency between the datasets at {0}% significance.\n".format(self.significance*100))           
                #print("  Details:")
                #print("  - Using the composite log-likelihood ratio test, the signed N-sigma between the datasets is {}".format(self.composite_nsigma))  
       
        # print("Consistency report- datasets are inconsistent at given confidence level if EITHER of the following scores report inconsistency.")
        # print()
        # print("Threshold for individual gatestring scores is {0}".format(single_string_thresh))
        # if number_of_significant_sequences > 0:
        #     print("As measured by worst-performing gate strings, data sets are INCONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        #     print("{0} gate string(s) have loglikelihood scores greater than the threshold.".format(number_of_significant_sequences))
        # else:
        #     print("As measured by worst-performing gate strings, data sets are CONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        #     print("{0} gate string(s) have loglikelihood scores greater than the threshold.".format(number_of_significant_sequences))
        # print()
        # print("Threshold for sum of gatestring scores is {0}.".format(composite_thresh))
        # if composite_score > composite_thresh:
        #     print("As measured by sum of gatestring scores, data sets are INCONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        # else:
        #     print("As measured by sum of gatestring scores, data sets are CONSISTENT at the {0}% confidence level.".format(confidence_level*100))
        # print("Total loglikelihood is {0}".format(composite_score))
        # print("Total number of standard deviations (N_sigma) of model violation is {0}.".format(N_sigma))
    
    def compute_TVDs(self, verbosity=2):
        """
        Todo
        """
        if verbosity >= 2:
            print("Computing {}% statistically significant TVDs...".format(self.significance*100),end='')
        try:
            assert len(self.dataset_list_or_multidataset) == 2
        except:
            raise ValueError("Can only compute TVD between two datasets.")    

        self.tvds = _collections.OrderedDict({})
        self.sstvds = _collections.OrderedDict({})
        for key in self.llrs.keys():
            tvd_val = tvd(key,self.dataset_list_or_multidataset[0],self.dataset_list_or_multidataset[1])
            self.tvds[key] = tvd_val
            if self.results.hypothesis_rejected[key]:               
                self.sstvds[key] = tvd_val
            else:
                self.sstvds[key] = None
        if verbosity >= 2:
            print("complete.")
                
    def rectify_datasets(self,confidence_level=0.95,target_score='dof'):
        assert(False), "This method needs to be fixed by Tim!"
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
            nListList = _np.array([list(dataline.allcounts.values()) for dataline in datalineList],'d')
            self.alpha_dict[gatestring] = target_score / llr
            print('Rescaling counts for string '+str(gatestring)+' by '+str(self.alpha_dict[gatestring]))
            print('|target score - new score| = '+str(loglikelihoodRatioTestObj(self.alpha_dict[gatestring],nListList,target_score)))
            for ds in dsList:
                ds[gatestring].scale(self.alpha_dict[gatestring])
        self.rectified_datasets = dsList
