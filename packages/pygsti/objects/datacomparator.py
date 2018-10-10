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

def xlogy(x,y):
    """
    Returns x*log(y).
    """
    if x == 0:
        return 0
    else:
        return x * _np.log(y)

def likelihood(pList,nList):
    """
    The likelihood for probabilities `pList` of a die,
    given `nList` counts for each outcome. 
    """
    output = 1.
    for i, pVal in enumerate(pList):
        output *= pVal**nList[i]
    return output

def loglikelihood(pList,nList):
    """
    The log of the likelihood for probabilities `pList` of a die,
    given `nList` counts for each outcome.
    """
    output = 0.
    for i, pVal in enumerate(pList):
        output += xlogy(nList[i],pVal)
    return output

# Only used by the rectify datasets function, which is commented out,
# so this is also commented out.
# def loglikelihoodRatioObj(alpha,nListList,dof):
#     return _np.abs(dof - loglikelihoodRatio(alpha*nListList))

def loglikelihoodRatio(nListList):
    """
    Calculates the log-likelood ratio between the null hypothesis
    that a die has *the same* probabilities in multiple "contexts" and
    that it has *different* probabilities in multiple "contexts".

    Parameters
    ----------
    nListList : List of lists of ints
        A list whereby element i is a list containing observed counts for
        all the different possible outcomes of the "die" in context i.

    Returns
    -------
    float
        The log-likehood ratio for this model comparison.
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

def JensenShannonDivergence(nListList):
    """
    Calculates the Jensen-Shannon divergence (JSD) between between different
    observed frequencies, obtained in different "contexts", for the different 
    outcomes of a "die" (i.e., coin with more than two outcomes).

    Parameters
    ----------
    nListList : List of lists of ints
        A list whereby element i is a list containing observed counts for
        all the different possible outcomes of the "die" in context i.

    Returns
    -------
    float
        The observed JSD for this data.
    """
    total_counts = _np.sum(_np.array(nListList))
    return loglikelihoodRatio(nListList)/(2*total_counts)

def pval(llrval, dof):
    """
    The p-value of a log-likelihood ratio (LLR), comparing a 
    nested null hypothsis and a larger alternative hypothesis.

    Parameters
    ----------
    llrval : float
        The log-likehood ratio

    dof : int
        The number of degrees of freedom associated with
        the LLR, given by the number of degrees of freedom
        of the full model space (the alternative hypothesis)
        minus the number of degrees of freedom of the restricted
        model space (the null hypothesis space).

    Returns
    -------
    float
        An approximation of the p-value for this LLR. This is
        calculated as 1 - F(llrval,dof) where F(x,k) is the
        cumulative distribution function, evaluated at x, for
        the chi^2_k distribution. The validity of this approximation
        is due to Wilks' theorem.
    """
    return 1 - _stats.chi2.cdf(llrval, dof)

def llr_to_signed_nsigma(llrval, dof):
    """
    Finds the signed number of standard deviations for the input
    log-likelihood ratio (LLR). This is given by

    (llrval - dof) / (sqrt(2*dof)).

    This is the number of standard deviations above the mean
    that `llrval` is for a chi^2_(dof) distribution.

    Parameters
    ----------
    llrval : float
        The log-likehood ratio

    dof : int
        The number of degrees of freedom associated with
        the LLR, given by the number of degrees of freedom
        of the full model space (the alternative hypothesis)
        minus the number of degrees of freedom of the restricted
        model space (the null hypothesis space), in the hypothesis
        test.

    Returns
    -------
    float
        The signed standard deviations.
    """
    return (llrval - dof) / _np.sqrt(2*dof)

def is_gatestring_allowed_by_exclusion(gate_exclusions, gatestring):
    """
    Returns True if `gatestring` does not contain any gates from `gate_exclusions`.
    Otherwise, returns False.
    """
    for gate in gate_exclusions:
        if gate in gatestring:
            return False
    return True

def is_gatestring_allowed_by_inclusion(gate_inclusions,gatestring):
    """
    Returns True if `gatestring` contains *any* of the gates from `gate_inclusions`.
    Otherwise, returns False. The exception is the empty gatestring, which always 
    returns True.
    """
    if len(gatestring) == 0: return True # always include the empty string
    for gate in gate_inclusions:
        if gate in gatestring:
            return True
    return False

def compute_llr_threshold(significance, dof):
    """
    Given a p-value threshold, *below* which a pvalue
    is considered statistically significant, it returns 
    the corresponding log-likelihood ratio threshold, *above* 
    which a LLR is considered statically significant. For a single
    hypothesis test, the input pvalue should be the desired "significance" 
    level of the test (as a value between 0 and 1). For multiple hypothesis
    tests, this will normally be smaller than the desired global significance.

    Parameters
    ----------
    pVal : float
        The p-value

    dof : int
        The number of degrees of freedom associated with
        the LLR , given by the number of degrees of freedom
        of the full model space (the alternative hypothesis)
        minus the number of degrees of freedom of the restricted
        model space (the null hypothesis space), in the hypothesis
        test.   

    Returns
    -------
    float
        The significance threshold for the LLR, given by
        1 - F^{-1}(pVal,dof) where F(x,k) is the cumulative distribution 
        function, evaluated at x, for the chi^2_k distribution. This
        formula is based on Wilks' theorem.   
    """
    return _scipy.stats.chi2.isf(significance,dof)

def tvd(nListList):
    """
    Calculates the total variation distance (TVD) between between different
    observed frequencies, obtained in different "contexts", for the *two* set of 
    outcomes for roles of a "die".

    Parameters
    ----------
    nListList : List of lists of ints
        A list whereby element i is a list counting counts for the
        different outcomes of the "die" in context i, for *two* contexts.

    Returns
    -------
    float
        The observed TVD between the two contexts
    """
    assert(len(nListList) == 2), "Can only compute the TVD between two sets of outcomes!"
    num_outcomes = len(nListList[0])
    assert(num_outcomes == len(nListList[1])), "The number of outcomes must be the same in both contexts!"

    N0 = _np.sum(nListList[0])
    N1 = _np.sum(nListList[1])

    return 0.5 * _np.sum(_np.abs(nListList[0][i]/N0 - nListList[1][i]/N1) for i in range(num_outcomes))

class DataComparator():
    """
    This object can be used to implement all of the "context dependence detection" methods described 
    in "Probing context-dependent errors in quantum processors", by Rudinger et al.
    (See that paper's supplemental material for explicit demonstrations of this object.)

    This object stores the p-values and log-likelihood ratio values from a consistency comparison between
    two or more datasets, and provides methods to:

        - Perform a hypothesis test to decide which sequences contain statistically significant variation.
        - Plot p-value histograms and log-likelihood ratio box plots.
        - Extract (1) the "statistically significant total variation distance" for a circuit, 
          (2) various other quantifications of the "amount" of context dependence, and (3) 
          the level of statistical significance at which any context dependence is detected.

    """
    def __init__(self, dataset_list_or_multidataset, gatestrings = 'all',
                 gate_exclusions = None, gate_inclusions = None, DS_names = None):
        """
        Initializes a DataComparator object.

        Parameters
        ----------
        dataset_list_multidataset : List of DataSets or MultiDataSet
            Either a list of DataSets, containing two or more sets of data to compare,
            or a MultiDataSet object, containing two or more sets of data to compare. Note
            that these DataSets should contain data for the same set of GateStrings (although
            if there are additional GateStrings these can be ignored using the parameters below).
            This object is then intended to be used test to see if the results are indicative 
            that the outcome probabilities for these GateStrings has changed between the "contexts" that 
            the data was obtained in.

        gatestrings : 'all' or list of GateStrings, optional (default is 'all')
            If 'all' the comparison is implemented for all GateStrings in the DataSets. Otherwise,
            this should be a list containing all the GateStrings to implement the comparison for (although
            note that some of these GateStrings may be ignored with non-default options for the next two
            inputs).

        gate_exclusions : None or list of gates, optional (default is None)
            If not None, all GateStrings containing *any* of the gates in this list are discarded,
            and no comparison will be made for those strings.

        gate_exclusions : None or list of gates, optional (default is None)
            If not None, a GateString will be dropped from the list to implement the comparisons for
            if it doesn't include *some* gate from this list (or is the empty gatestring). 

        DS_names : None or list, optional (default is None)
            If `dataset_list_multidataset` is a list of DataSets, this can be used to specify names
            for the DataSets in the list. E.g., ["Time 0", "Time 1", "Time 3"] or ["Driving","NoDriving"].
        
        Returns
        -------
        A DataComparator object.

        """      
        if DS_names is not None:
            if len(DS_names) != len(dataset_list_or_multidataset):
                raise ValueError('Length of provided DS_names list must equal length of dataset_list_or_multidataset.')
        
        if isinstance(gatestrings,str):
            assert(gatestrings == 'all'), "If gatestrings is a string it must be 'all'!"

        if isinstance(dataset_list_or_multidataset,list):
            dsList = dataset_list_or_multidataset    
            olIndex = dsList[0].olIndex
            olIndexListBool = [ds.olIndex==(olIndex) for ds in dsList]
            DS_names = list(range(len(dataset_list_or_multidataset)))
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
            if gatestrings == 'all':
                gatestrings = dsList[0].keys()
            if DS_names is None:
                DS_names = list(dataset_list_or_multidataset.keys())

        else:
            raise ValueError("The `dataset_list_or_multidataset` must be a list of DataSets of a MultiDataSet!")
                
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
        jsds = {}
        dof = (len(dsList) - 1) * (len(dsList[0].olIndex) - 1)
        total_counts = []

        if len(dataset_list_or_multidataset) == 2:
            tvds = {}
        
        for gatestring in gatestrings:
            datalineList = [ds[gatestring] for ds in dsList]
            nListList = _np.array([list(dataline.allcounts.values()) for dataline in datalineList])
            total_counts.append(_np.sum(nListList))
            llrs[gatestring] = loglikelihoodRatio(nListList)
            jsds[gatestring] = JensenShannonDivergence(nListList)
            pVals[gatestring] =  pval(llrs[gatestring],dof)
            if len(dataset_list_or_multidataset) == 2:
                tvds[gatestring] = tvd(nListList) 

        self.dataset_list_or_multidataset = dataset_list_or_multidataset
        self.pVals = pVals
        self.pVals_pseudothreshold = None
        self.llrs = llrs
        self.llrs_pseudothreshold = None
        self.jsds = jsds
        if len(dataset_list_or_multidataset) == 2:
            self.tvds = tvds
        self.gate_exclusions = gate_exclusions
        self.gate_inclusions = gate_inclusions
        self.pVals0 = str(len(self.pVals)-_np.count_nonzero(list(self.pVals.values())))
        self.dof = dof
        self.num_strs = len(self.pVals)
        self.DS_names = DS_names

        if _np.std(_np.array(total_counts)) > 10e-10:
            self.fixed_totalcount_data = False
            self.counts_per_sequence = None
        else:
            self.fixed_totalcount_data = True
            self.counts_per_sequence = int(total_counts[0])

        self.aggregate_llr = _np.sum(list(self.llrs.values())) 
        self.aggregate_llr_threshold = None    
        self.aggregate_pVal = pval(self.aggregate_llr, self.num_strs*self.dof) 
        self.aggregate_pVal_threshold = None 

        # Convert the aggregate LLR to a signed standard deviations.
        self.aggregate_nsigma = llr_to_signed_nsigma(self.aggregate_llr,self.num_strs*self.dof)
        self.aggregate_nsigma_threshold = None 
        
        # All attributes to be populated in methods that can be called from .get methods, so
        # we can raise a meaningful warning if they haven't been calculated yet.
        self.sstvds = None
        self.pVal_pseudothreshold = None
        self.llr_pseudothreshold = None
        self.pVal_pseudothreshold = None
        self.jsd_pseudothreshold = None

        self.aggregate_llr_threshold = None
        self.aggregate_nsigma_threshold = None
        self.aggregate_pVal_threshold = None

    def implement(self, significance=0.05, per_sequence_correction='Hochberg', 
                  aggregate_test_weighting=0.5,  pass_alpha=True, verbosity=1):
        """
        Implements statistical hypothesis testing, to detect whether there is statistically
        significant variation between the DateSets in this DataComparator. This performs
        hypothesis tests on the data from individual circuits, and a joint hypothesis test
        on all of the data. With the default settings, this is the method described and implemented
        in "Probing context-dependent errors in quantum processors", by Rudinger et al. With
        non-default settings, this is some minor variation on that method.

        Note that the default values of all the parameters are likely sufficient for most 
        purposes.

        Parameters
        ----------
        significance : float in (0,1), optional (default is 0.05)
            The "global" statistical significance to implement the tests at. I.e, with
            the standard `per_sequence_correction` value (and some other values for this parameter)
            the probability that a sequence that has been flagged up as context dependent
            is actually from a context-independent circuit is no more than `significance`.
            Precisely, `significance` is what the "family-wise error rate" (FWER) of the full set
            of hypothesis tests (1 "aggregate test", and 1 test per sequence) is controlled to, 
            as long as `per_sequence_correction` is set to the default value, or another option 
            that controls the FWER of the per-sequence comparion (see below).
        
        per_sequence_correction : string, optional (default is 'Hochberg')
            The multi-hypothesis test correction used for the per-circuit/sequence comparisons.
            (See "Probing context-dependent errors in quantum processors", by Rudinger et al. for
            the details of what the per-circuit comparison is). This can be any string that is an allowed 
            value for the `localcorrections` input parameter of the HypothesisTest object. This includes:

                - 'Hochberg'. This implements the Hochberg multi-test compensation technique. This
                is strictly the best method available in the code, if you wish to control the FWER, 
                and it is the method described in "Probing context-dependent errors in quantum processors", 
                by Rudinger et al.

                - 'Holms'. This implements the Holms multi-test compensation technique. This
                controls the FWER, and it results in a strictly less powerful test than the Hochberg 
                correction.

                - 'Bonferroni'. This implements the well-known Bonferroni multi-test compensation 
                technique. This controls the FWER, and it results in a strictly less powerful test than 
                the Hochberg correction.

                - 'none'. This implements no multi-test compensation for the per-sequence comparsions,
                so they are all implemented at a "local" signifincance level that is altered from `significance`
                only by the (inbuilt) Bonferroni-like correction between the "aggregate" test and the per-sequence
                tests. This option does *not* control the FWER, and many sequences may be flagged up as context 
                dependent even if none are.

                -'Benjamini-Hochberg'. This implements the Benjamini-Hockberg multi-test compensation 
                technique. This does *not* control the FWER, and instead controls the "False Detection Rate"
                (FDR); see, for example, https://en.wikipedia.org/wiki/False_discovery_rate. That means that 
                the global significance is maintained for the test of "Is there any context dependence?". I.e., 
                one or more tests will trigger when there is no context 
                dependence with at most a probability of `significance`. But, if one or more per-sequence tests 
                trigger then we are only guaranteed that (in expectation) no more than a fraction of 
                "local-signifiance" of the circuits that have been flagged up as context dependent actually aren't. 
                Here, "local-significance" is the  significance at which the per-sequence tests are, together, 
                implemented, which is `significance`*(1 - `aggregate_test_weighting`) if the aggregate test doesn't 
                detect context dependence and `significance` if it does (as long as `pass_alpha` is True). This
                method is strictly more powerful than the Hochberg correction, but it controls a different, weaker
                quantity.
        
        aggregate_test_weighting : float in [0,1], optional (default is 0.5)
            The weighting, in a generalized Bonferroni correction, to put on the "aggregate test", that jointly
            tests all of the data for context dependence (in contrast to the per-sequence tests). If this is 0 then 
            the aggreate test is not implemented, and if it is 1 only the aggregate test is implemented (unless it 
            triggers and `pass_alpha` is True).

        pass_alpha : Bool, optional (default is True)

            The aggregate test is implemented first, at the "local" significance defined by `aggregate_test_weighting`
            and `significance` (see above). If `pass_alpha` is True, then when the aggregate test triggers all the 
            local significance for this test is passed on to the per-sequence tests (which are then jointly implemented 
            with significance `significance`, that is then locally corrected for the multi-test correction as specified
            above), and when the aggregate test doesn't trigger this local significance isn't passed on. If `pass_alpha` 
            is False then local significance of the aggregate test is never passed on from the aggregate test. See 
            "Probing context-dependent errors in quantum processors", by Rudinger et al. (or hypothesis testing literature) 
            for discussions of why this "significance passing" still maintains a (global) FWER of `significance`.
            Note that The default value of True always results in a strictly more powerful test.

        verbosity : int, optional (default is 1)
            If > 0 then a summary of the results of the tests is printed to screen. Otherwise, the
            various .get_...() methods need to be queried to obtain the results of the 
            hypothesis tests.

        Returns
        -------
        None

        """
        self.significance = significance
        assert(aggregate_test_weighting <= 1. or aggregate_test_weighting >= 0.), "The weighting on the aggregate test must be between 0 and 1!"
        
        if verbosity >= 2:
            print("Implementing {0:.2f}% significance statistical hypothesis testing...".format(self.significance*100),end='')

        gatestrings = tuple(self.pVals.keys())
        hypotheses = ('aggregate', gatestrings)
        weighting = {}
        weighting['aggregate'] = aggregate_test_weighting
        weighting[gatestrings] = 1 - aggregate_test_weighting
        
        if pass_alpha: passing_graph = 'Holms'
        else: passing_graph = 'none'

        hypotest = _HypothesisTest(hypotheses, significance=significance, weighting=weighting, 
                                   passing_graph=passing_graph, local_corrections=per_sequence_correction)
        extended_pVals_dict = _copy.copy(self.pVals)
        extended_pVals_dict['aggregate'] = self.aggregate_pVal
        hypotest.add_pvalues(extended_pVals_dict)
        hypotest.implement()
        self.results = hypotest

        if aggregate_test_weighting == 0:
            self.aggregate_llr_threshold = _np.inf
            self.aggregate_nsigma_threshold = _np.inf
            self.aggregate_pVal_threshold = 0.
        else:
            self.aggregate_llr_threshold = compute_llr_threshold(aggregate_test_weighting*significance, self.num_strs*self.dof)
            self.aggregate_nsigma_threshold = llr_to_signed_nsigma(self.aggregate_llr_threshold, self.num_strs*self.dof)
            self.aggregate_pVal_threshold = aggregate_test_weighting*significance

        self.pVal_pseudothreshold = hypotest.pvalue_pseudothreshold[gatestrings]
        self.llr_pseudothreshold = compute_llr_threshold(self.pVal_pseudothreshold,self.dof)

        if self.fixed_totalcount_data:
            self.jsd_pseudothreshold = self.llr_pseudothreshold/self.counts_per_sequence

        temp_hypothesis_rejected_dict = _copy.copy(hypotest.hypothesis_rejected)
        self.inconsistent_datasets_detected = any(list(temp_hypothesis_rejected_dict.values()))
        del temp_hypothesis_rejected_dict['aggregate']            
        self.number_of_significant_sequences = _np.sum(list(temp_hypothesis_rejected_dict.values()))

        if len(self.dataset_list_or_multidataset) == 2:
            sstvds = {}
            for gs in list(self.llrs.keys()):
                if self.results.hypothesis_rejected[gs]:               
                    sstvds[gs] = self.tvds[gs]
            self.sstvds = sstvds

        if verbosity >= 2:
            print("complete.")

        if verbosity >= 2:
            print("\n--- Results ---\n")

        if verbosity >= 1:
            if self.inconsistent_datasets_detected:
                print("The datasets are INCONSISTENT at {0:.2f}% significance.".format(self.significance*100))
                print("  - Details:")
                print("    - The aggregate log-likelihood ratio test is significant at {0:.2f} standard deviations.".format(self.aggregate_nsigma))
                print("    - The aggregate log-likelihood ratio test standard deviations signficance threshold is {0:.2f}".format(self.aggregate_nsigma_threshold)) 
                print("    - The number of sequences with data that is inconsistent is {0}".format(self.number_of_significant_sequences))
                if len(self.dataset_list_or_multidataset) == 2 and self.number_of_significant_sequences>0:
                    max_SSTVD_gs, max_SSTVD = self.get_maximum_SSTVD()
                    print("    - The maximum SSTVD over all sequences is {0:.2f}".format(max_SSTVD)) 
                    print("    - The maximum SSTVD was observed for {}".format(max_SSTVD_gs))                    
            else:
                print("Statistical hypothesis tests did NOT find inconsistency between the datasets at {0:.2f}% significance.".format(self.significance*100))           
        
        return

    def get_TVD(self, gatestring):
        """
        Returns the observed total variation distacnce (TVD) for the specified gatestring.
        This is only possible if the comparison is between two sets of data. See Eq. (19) 
        in "Probing context-dependent errors in quantum processors", by Rudinger et al. for the 
        definition of this observed TVD.

        This is a quantification for the "amount" of context dependence for this gatestring (see also,
        get_JSD(), get_SSTVD() and get_SSJSD()).

        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the TVD of.

        Returns
        -------
        float
            The TVD for the specified gatestring.
        """
        try: assert len(self.dataset_list_or_multidataset) == 2
        except: raise ValueError("The TVD is only defined for comparisons between two datasets!")  

        return self.tvds[gatestring]

    def get_SSTVD(self, gatestring):
        """
        Returns the "statistically significant total variation distacnce" (SSTVD) for the specified 
        gatestring. This is only possible if the comparison is between two sets of data. The SSTVD
        is None if the gatestring has not been found to have statistically significant variation.
        Otherwise it is equal to the observed TVD. See Eq. (20) and surrounding discussion in 
        "Probing context-dependent errors in quantum processors", by Rudinger et al., for more information.
        
        This is a quantification for the "amount" of context dependence for this gatestring (see also,
        get_JSD(), get_TVD() and get_SSJSD()).

        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the SSTVD of.

        Returns
        -------
        float
            The SSTVD for the specified gatestring.
        """
        try: assert len(self.dataset_list_or_multidataset) == 2
        except: raise ValueError("Can only compute TVD between two datasets.")  
        assert(self.sstvds is not None), "The SSTVDS have not been calculated! Run the .implement() method first!"

        return self.sstvds.get(gatestring, None)

    def get_maximum_SSTVD(self):
        """
        Returns the maximum, over gatestrings, of the "statistically significant total variation distance" 
        (SSTVD). This is only possible if the comparison is between two sets of data. See the .get_SSTVD()
        method for information on SSTVD.

        Returns
        -------
        float
            The gatestring associated with the maximum SSTVD, and the SSTVD of that gatestring.
        """
        try: assert len(self.dataset_list_or_multidataset) == 2
        except: raise ValueError("Can only compute TVD between two datasets.")  
        assert(self.sstvds is not None), "The SSTVDS have not been calculated! Run the .implement() method first!"

        if len(self.sstvds) == 0:
            return None, None
        else: 
            index = _np.argmax(list(self.sstvds.values()))
            max_sstvd_gs = list(self.sstvds.keys())[index]
            max_sstvd = self.sstvds[max_sstvd_gs]
            
            return max_sstvd_gs, max_sstvd

    def get_pvalue(self, gatestring):
        """
        Returns the pvalue for the log-likelihood ratio test for the specified gatestring.
  
        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the p-value of.

       Returns
        -------
        float
            The p-value of the specified gatestring.        
        """
        return self.pVals[gatestring]

    def get_pvalue_pseudothreshold(self):
        """
        Returns the (multi-test-adjusted) statistical significance pseudo-threshold for the per-sequence 
        p-values (obtained from the log-likehood ratio test). This is a "pseudo-threshold", because it 
        is data-dependent in general, but all the per-sequence p-values below this value are statistically 
        significant. This quantity is given by Eq. (9) in  "Probing context-dependent errors in quantum 
        processors", by Rudinger et al.

        Returns
        -------
        float
            The statistical significance pseudo-threshold for the per-sequence p-value.
        """
        assert(self.pVal_pseudothreshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.pVal_pseudothreshold

    def get_LLR(self, gatestring):
        """
        Returns the log-likelihood ratio (LLR) for the input gatestring.
        This is the quantity defined in Eq (4) of "Probing context-dependent 
        errors in quantum processors", by Rudinger et al.
        
        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the LLR of.

        Returns
        -------
        float
            The LLR of the specified gatestring.
        """
        return self.llrs[gatestring]

    def get_LLR_pseudothreshold(self):
        """
        Returns the (multi-test-adjusted) statistical significance pseudo-threshold for the per-sequence 
        log-likelihood ratio (LLR). This is a "pseudo-threshold", because it is data-dependent in 
        general, but all LLRs above this value are statistically significant. This quantity is given 
        by Eq (10) in  "Probing context-dependent errors in quantum processors", by Rudinger et al.

        Returns
        -------
        float
            The statistical significance pseudo-threshold for per-sequence LLR.
        """
        assert(self.llr_pseudothreshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.llr_pseudothreshold

    def get_JSD(self, gatestring):
        """
        Returns the observed Jensen-Shannon divergence (JSD) between "contexts" for
        the specified gatestring. The JSD is a rescaling of the LLR, given by dividing
        the LLR by 2*N where N is the total number of counts (summed over contexts) for 
        this gatestring. This quantity is given by Eq (15) in  "Probing context-dependent 
        errors in quantum processors", Rudinger et al.

        This is a quantification for the "amount" of context dependence for this gatestring (see also,
        get_TVD(), get_SSTVD() and get_SSJSD()).

        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the JSD of

        Returns
        -------
        float
            The JSD of the specified gatestring.
        """
        return self.jsds[gatestring]

    def get_JSD_pseudothreshold(self):
        """
        Returns the statistical significance pseudo-threshold for the Jensen-Shannon divergence (JSD) 
        between "contexts". This is a rescaling of the pseudo-threshold for the LLR, returned by the
        method .get_LLR_pseudothreshold(); see that method for more details. This threshold is also given by
        Eq (17) in  "Probing context-dependent errors in quantum processors", by Rudinger et al.

        Note that this pseudo-threshold is not defined if the total number of counts (summed over 
        contexts) for a sequence varies between sequences.

        Returns
        -------
        float
            The pseudo-threshold for the JSD of a gatestring, if well-defined.
        """
        assert(self.fixed_totalcount_data), "The JSD only has a pseudo-threshold when there is the same number of total counts per sequence!"
        assert(self.jsd_pseudothreshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.jsd_pseudothreshold

    def get_SSJSD(self, gatestring):
        """
        Returns the "statistically significanet Jensen-Shannon divergence" (SSJSD) between "contexts" for
        the specified gatestring. This is the JSD of the gatestring (see .get_JSD()), if the gatestring 
        has been found to be context dependent, and otherwise it is None. This quantity is the JSD version 
        of the SSTVD given in Eq. (20) of "Probing context-dependent errors in quantum processors", by Rudinger 
        et al.

        This is a quantification for the "amount" of context dependence for this gatestring (see also,
        get_TVD(), get_SSTVD() and get_SSJSD()).

        Parameters
        ----------
        gatestring : GateString
            The gatestring to return the JSD of

        Returns
        -------
        float
            The JSD of the specified gatestring.
        """
        assert(self.llr_pseudothreshold is not None), "The hypothsis testing has not been implemented yet! Run the .implement() method first!"
        if self.results.hypothesis_rejected[gatestring]:               
            return self.jsds[gatestring]
        else:
            return None

    def get_aggregate_LLR(self):
        """
        Returns the "aggregate" log-likelihood ratio (LLR), comparing the null
        hypothesis of no context dependence in *any* sequence with the full model
        of arbitrary context dependence. This is the sum of the per-sequence LLRs, and
        it is defined in Eq (11) of "Probing context-dependent  errors in 
        quantum processors", by Rudinger et al.
  
        Returns
        -------
        float
            The aggregate LLR.
        """
        return self.aggregate_llr

    def get_aggregate_LLR_threshold(self):
        """
        Returns the (multi-test-adjusted) statistical significance threshold for the 
        "aggregate" log-likelihood ratio (LLR), above which this LLR is significant. 
        See .get_aggregate_LLR() for more details. This quantity is the LLR version 
        of the quantity defined in Eq (14) of "Probing context-dependent errors in 
        quantum processors", by Rudinger et al.

        Returns
        -------
        float
            The threshold above which the aggregate LLR is statistically significant.
        """
        assert(self.aggregate_llr_threshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.aggregate_llr_threshold

    def get_aggregate_pvalue(self):
        """
        Returns the p-value for the "aggregate" log-likelihood ratio (LLR), comparing the null
        hypothesis of no context dependence in any sequence with the full model of arbitrary 
        dependence. This LLR is defined in Eq (11) in "Probing context-dependent errors in 
        quantum processors", by Rudinger et al., and it is converted to a p-value via Wilks' 
        theorem (see discussion therein).

        Note that this p-value is often zero to machine precision, when there is context dependence,
        so a more useful number is often returned by get_aggregate_nsigma() (that quantity is equivalent to 
        this p-value but expressed on a different scale).
  
        Returns
        -------
        float
            The p-value of the aggregate LLR.
        """
        return self.aggregate_pVal

    def get_aggregate_pvalue_threshold(self):
        """
        Returns the (multi-test-adjusted) statistical significance threshold for the p-value of 
        the "aggregate" log-likelihood ratio (LLR), below which this p-value is significant. 
        See the .get_aggregate_pvalue() method for more details.
  
        Returns
        -------
        float
            The statistical significance threshold for the p-value of the "aggregate" LLR.
        """
        assert(self.aggregate_pVal_threshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.aggregate_pVal_threshold

    def get_aggregate_nsigma(self):
        """
        Returns the number of standard deviations above the context-independent mean that the "aggregate" 
        log-likelihood ratio (LLR) is. This quantity is defined in Eq (13) of "Probing context-dependent 
        errors in quantum processors", by Rudinger et al.
  
        Returns
        -------
        float
            The number of signed standard deviations of the aggregate LLR .
        """
        return self.aggregate_nsigma

    def get_aggregate_nsigma_threshold(self):
        """
        Returns the (multi-test-adjusted) statistical significance threshold for the signed standard
        deviations of the the "aggregate" log-likelihood ratio (LLR). See the .get_aggregate_nsigma() 
        method for more details. This quantity is defined in Eq (14) of "Probing context-dependent errors 
        in quantum processors", by Rudinger et al.
  
        Returns
        -------
        float
            The statistical significance threshold above which the signed standard deviations 
            of the aggregate LLR is significant.
        """
        assert(self.aggregate_nsigma_threshold is not None), "This has not yet been calculated! Run the .implement() method first!"
        return self.aggregate_nsigma_threshold

    def get_worst_gatestrings(self, number):
        """
        Returns the "worst" gatestrings that have the smallest p-values.

        Parmeters
        ---------
        number : int
            The number of gatestrings to return.

        Returns
        -------
        List
            A list of tuples containing the worst `number` gatestrings along
            with the correpsonding p-values.
        """
        worst_strings = sorted(self.pVals.items(), key=lambda kv: kv[1])[:number]
        return worst_strings

    # Commented out, as it doesn't work with Tim's updated DataComparator, and it
    # seems not to be being used currently.
    # def rectify_datasets(self,confidence_level=0.95,target_score='dof'):
    #     """
    #     Todo
    #     """
    #     assert(False), "This method needs to be fixed by Tim!"
    #     if target_score == 'dof':
    #         target_score = self.dof
    #     single_string_thresh = find_thresh(confidence_level,self.num_strs,self.dof)
    #     single_thresh_violator_locs = _np.nonzero(_np.where(self.llrVals>single_string_thresh,1,0))[0]
    #     self.alpha_dict = {}
    #     if isinstance(self.dataset_list_or_multidataset,list):
    #         dsList = [DS.copy_nonstatic() for DS in self.dataset_list_or_multidataset]
    #     elif isinstance(self.dataset_list_or_multidataset,_MultiDataSet):
    #         dsList = [self.dataset_list_or_multidataset[key].copy() for key in self.dataset_list_or_multidataset.keys()]
    #     for violator_loc in single_thresh_violator_locs:
    #         gatestring = self.llrVals_and_strings[violator_loc][0]
    #         llr = self.llrVals_and_strings[violator_loc][1]
    #         datalineList = [ds[gatestring] for ds in dsList]
    #         nListList = _np.array([list(dataline.allcounts.values()) for dataline in datalineList],'d')
    #         self.alpha_dict[gatestring] = target_score / llr
    #         print('Rescaling counts for string '+str(gatestring)+' by '+str(self.alpha_dict[gatestring]))
    #         print('|target score - new score| = '+str(loglikelihoodRatioObj(self.alpha_dict[gatestring],nListList,target_score)))
    #         for ds in dsList:
    #             ds[gatestring].scale(self.alpha_dict[gatestring])
    #     self.rectified_datasets = dsList
