"""
Defines HypothesisTest object and supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy

import numpy as _np


class HypothesisTest(object):
    """
    A set of statistical hypothesis tests on a set of null hypotheses.

    This object has *not* been carefully tested.

    Parameters
    ----------
    hypotheses : list or tuple
        Specifies the set of null hypotheses. This should be a list containing elements
        that are either

        * A "label" for a hypothesis, which is just some hashable object such
          as a string.
        * A tuple of "nested hypotheses", which are also just labels for some
          null hypotheses.

        The elements of this list are then subject to multi-test correction of the "closed test
        procedure" type, with the exact correction method specified by `passing_graph`. For each element that
        is itself a tuple of hypotheses, these hypotheses are then further corrected using the method
        specified by `local_corrections`.

    significance : float in (0,1), optional
        The global significance level. If either there are no "nested hypotheses" or the
        correction used for the nested hypotheses will locally control the family-wise error rate
        (FWER) (such as if `local_correction`='Holms') then when the hypothesis test encoded by
        this object will control the FWER to `significance`.

    weighting : string or dict.
        Specifies what proportion of `significance` is initially allocated to each element
        of `hypotheses`. If a string, must be 'equal'. In this case, the local significance
        allocated to each element of `hypotheses` is `significance`/len(`hypotheses`). If
        not a string, a dictionary whereby each key is an element of `hypotheses` and each value
        is a non-negative integer (which will be normalized to one inside the function).

    passing_graph : string or numpy.array
        Specifies where the local significance from each test in `hypotheses` that triggers is
        passed to. If a string, then must be 'Holms'. In this case a test that triggers passes
        it's local significance to all the remaining hypotheses that have not yet triggered, split
        evenly over these hypotheses. If it is an array then its value for [i,j] is the proportion
        of the "local significance" that is passed from hypothesis with index i (in the tuple
        `hypotheses`) to the hypothesis with index j if the hypothesis with index i is rejected (and
        if j hasn't yet been rejected; otherwise that proportion is re-distributed other the other
        hypothesis that i is to pass it's significance to). The only restriction on restriction on
        this array is that a row must sum to <= 1 (and it is sub-optimal for a row to sum to less
        than 1).

        Note that a nested hypothesis is not allowed to pass significance out of it, so any rows
        that request doing this will be ignored. This is because a nested hypothesis represents a
        set of hypotheses that are to be jointly tested using some multi-test correction, and so
        this can only pass significance out if *all* of the hypotheses in that nested hypothesis
        are rejected. As this is unlikely in most use-cases, this has not been allowed for.

    local_corrections : str, optional
        The type of multi-test correction used for testing any nested hypotheses. After all
        of the "top level" testing as been implemented on all non-nested hypotheses, whatever
        the "local" significance is for each of the "nested hypotheses" is multi-test corrected
        using this procedure. Must be one of:

        * 'Holms'. This implements the Holms multi-test compensation technique. This
        controls the FWER for each set of nested hypotheses (and so controls the global FWER, in
        combination with the "top level" corrections). This requires no assumptions about the
        null hypotheses.
        * 'Bonferroni'. This implements the well-known Bonferroni multi-test compensation technique. 
        This is strictly less powerful test than the Hochberg correction. Note that neither 
        'Holms' nor 'Bonferronni' gained any advantage from being implemented
        using "nesting", as if all the hypotheses were put into the "top level" the same corrections
        could be achieved.
        * 'Hochberg'. This implements the Hockberg multi-test compensation technique. It is
        not a "closed test procedure", so it is not something that can be implemented in the
        top level. To be provably valid, it is necessary for the p-values of the nested
        hypotheses to be non-negatively dependent. When that is true, this is strictly better
        than the Holms and Bonferroni corrections whilst still controlling the FWER.
        * 'none'. This implements no multi-test compensation. This option does *not* control the
        FWER of the nested hypotheses. So it will generally not control the global FWER as specified.
        * 'Benjamini-Hochberg'. This implements the Benjamini-Hockberg multi-test compensation
        technique. This does *not* control the FWER of the nested hypotheses, and instead controls
        the "False Detection Rate" (FDR); see wikipedia. That means that the global significance is
        maintained in the sense that the probability of one or more tests triggering is at most `significance`.
        But, if one or more tests are triggered in a particular nested hypothesis test we are only guaranteed
        that (in expectation) no more than a fraction of  "local signifiance" of tests are false alarms.This
        method is strictly more powerful than the Hochberg correction, but it controls a different, weaker
        quantity.
        
    """

    def __init__(self, hypotheses, significance=0.05, weighting='equal',
                 passing_graph='Holms', local_corrections='Holms'):
        """
        Initializes a HypothesisTest object. This specifies the set of null hypotheses,
        and the tests to be implemented, it does *not* implement the tests. Methods are used
        to add the data (.add_pvalues) and run the tests (.run).

        Parameters
        ----------
        hypotheses : list or tuple
            Specifies the set of null hypotheses. This should be a list containing elements
            that are either

            * A "label" for a hypothesis, which is just some hashable object such
              as a string.
            * A tuple of "nested hypotheses", which are also just labels for some
              null hypotheses.

            The elements of this list are then subject to multi-test correction of the "closed test
            procedure" type, with the exact correction method specified by `passing_graph`. For each element that
            is itself a tuple of hypotheses, these hypotheses are then further corrected using the method
            specified by `local_corrections`.

        significance : float in (0,1), optional
            The global significance level. If either there are no "nested hypotheses" or the
            correction used for the nested hypotheses will locally control the family-wise error rate
            (FWER) (such as if `local_correction`='Holms') then when the hypothesis test encoded by
            this object will control the FWER to `significance`.

        weighting : string or dict.
            Specifies what proportion of `significance` is initially allocated to each element
            of `hypotheses`. If a string, must be 'equal'. In this case, the local significance
            allocated to each element of `hypotheses` is `significance`/len(`hypotheses`). If
            not a string, a dictionary whereby each key is an element of `hypotheses` and each value
            is a non-negative integer (which will be normalized to one inside the function).

        passing_graph : string or numpy.array
            Specifies where the local significance from each test in `hypotheses` that triggers is
            passed to. If a string, then must be 'Holms'. In this case a test that triggers passes
            it's local significance to all the remaining hypotheses that have not yet triggered, split
            evenly over these hypotheses. If it is an array then its value for [i,j] is the proportion
            of the "local significance" that is passed from hypothesis with index i (in the tuple
            `hypotheses`) to the hypothesis with index j if the hypothesis with index i is rejected (and
            if j hasn't yet been rejected; otherwise that proportion is re-distributed other the other
            hypothesis that i is to pass it's significance to). The only restriction on restriction on
            this array is that a row must sum to <= 1 (and it is sub-optimal for a row to sum to less
            than 1).

            Note that a nested hypothesis is not allowed to pass significance out of it, so any rows
            that request doing this will be ignored. This is because a nested hypothesis represents a
            set of hypotheses that are to be jointly tested using some multi-test correction, and so
            this can only pass significance out if *all* of the hypotheses in that nested hypothesis
            are rejected. As this is unlikely in most use-cases, this has not been allowed for.

        local_corrections : str, optional
            The type of multi-test correction used for testing any nested hypotheses. After all
            of the "top level" testing as been implemented on all non-nested hypotheses, whatever
            the "local" significance is for each of the "nested hypotheses" is multi-test corrected
            using this procedure. Must be one of:

            * 'Holms'. This implements the Holms multi-test compensation technique. This
            controls the FWER for each set of nested hypotheses (and so controls the global FWER, in
            combination with the "top level" corrections). This requires no assumptions about the
            null hypotheses.
            * 'Bonferroni'. This implements the well-known Bonferroni multi-test compensation
            technique. This is strictly less powerful test than the Hochberg correction.
            Note that neither 'Holms' nor 'Bonferronni' gained any advantage from being implemented
            using "nesting", as if all the hypotheses were put into the "top level" the same corrections
            could be achieved.
            * 'Hochberg'. This implements the Hockberg multi-test compensation technique. It is
            not a "closed test procedure", so it is not something that can be implemented in the
            top level. To be provably valid, it is necessary for the p-values of the nested
            hypotheses to be non-negatively dependent. When that is true, this is strictly better
            than the Holms and Bonferroni corrections whilst still controlling the FWER.
            * 'none'. This implements no multi-test compensation. This option does *not* control the
            FWER of the nested hypotheses. So it will generally not control the global FWER as specified.
            * 'Benjamini-Hochberg'. This implements the Benjamini-Hockberg multi-test compensation
            technique. This does *not* control the FWER of the nested hypotheses, and instead controls
            the "False Detection Rate" (FDR); see wikipedia. That means that the global significance is
            maintained in the sense that the probability of one or more tests triggering is at most `significance`.
            But, if one or more tests are triggered in a particular nested hypothesis test we are only guaranteed
            that (in expectation) no more than a fraction of  "local signifiance" of tests are false alarms.This
            method is strictly more powerful than the Hochberg correction, but it controls a different, weaker
            quantity.

        Returns
        -------
        A HypothesisTest object.
        """
        assert(0. < significance and significance < 1.), \
            'The significance level in a hypotheses test must be > 0 and < 1!'

        self.hypotheses = hypotheses
        self.significance = significance
        self.pvalues = None
        self.hypothesis_rejected = None
        self.significance_tested_at = None
        self.pvalue_pseudothreshold = None

        self.nested_hypotheses = {}
        for h in self.hypotheses:
            if not (isinstance(h, tuple) or isinstance(h, list)):
                self.nested_hypotheses[h] = False
            else:
                self.nested_hypotheses[h] = True

        if isinstance(passing_graph, str):
            assert(passing_graph == 'Holms')
            self._initialize_to_weighted_holms_test()

        self.local_significance = {}
        if isinstance(weighting, str):
            assert(weighting == 'equal')
            for h in self.hypotheses:
                self.local_significance[h] = self.significance / len(self.hypotheses)
        else:
            totalweight = 0.
            for h in self.hypotheses:
                totalweight += weighting[h]
            for h in self.hypotheses:
                self.local_significance[h] = significance * weighting[h] / totalweight

        if isinstance(local_corrections, str):
            assert(local_corrections in ('Holms', 'Hochberg', 'Bonferroni', 'none', 'Benjamini-Hochberg')
                   ), "A local correction of `{}` is not a valid choice".format(local_corrections)
            self.local_corrections = {}
            for h in self.hypotheses:
                if self.nested_hypotheses[h]:
                    self.local_corrections[h] = local_corrections
        else:
            self.local_corrections = local_corrections

        #self._check_permissible()

        # if is not isinstance(threshold_function, str):
        #     raise ValueError ("Data that is not p-values is currently not supported!")
        # else:
        #     if threshold_function is not 'pvalue':
        #         raise ValueError ("Data that is not p-values is currently not supported!")

        return

    def _initialize_to_weighted_holms_test(self):
        """
        Initializes the passing graph to the weighted Holms test.
        """
        self.passing_graph = _np.zeros((len(self.hypotheses), len(self.hypotheses)), float)
        for hind, h in enumerate(self.hypotheses):
            if not self.nested_hypotheses[h]:
                self.passing_graph[hind, :] = _np.ones(len(self.hypotheses), float) / (len(self.hypotheses) - 1)
                self.passing_graph[hind, hind] = 0.

    # def _check_permissible(self):
    #     """
    #     Todo
    #     """
    #     # Todo : test that the graph is acceptable.
    #     return True

    def add_pvalues(self, pvalues):
        """
        Insert the p-values for the hypotheses.

        Parameters
        ----------
        pvalues : dict
            A dictionary specifying the p-value for each hypothesis.

        Returns
        -------
        None
        """
        # Testing this is slow, so we'll just leave it out.
        #for h in self.hypotheses:
        #    if self.nested_hypotheses[h]:
        #        for hh in h:
        #           assert(hh in list(pvalues.keys())), \
        #               "Some hypothese do not have a pvalue in this pvalue dictionary!"
        #            assert(pvalues[hh] >= 0. and pvalues[hh] <= 1.), "Invalid p-value!"
        #    else:
        #        assert(h in list(pvalues.keys())), "Some hypothese do not have a pvalue in this pvalue dictionary!"
        #        assert(pvalues[h] >= 0. and pvalues[h] <= 1.), "Invalid p-value!"
        self.pvalues = _copy.copy(pvalues)

        return

    def run(self):
        """
        Implements the multiple hypothesis testing routine encoded by this object.

        This populates the self.hypothesis_rejected dictionary, that shows which
        hypotheses can be rejected using the procedure specified.

        Returns
        -------
        None
        """
        assert(self.pvalues is not None), "Data must be input before the test can be implemented!"

        self.pvalue_pseudothreshold = {}
        self.hypothesis_rejected = {}
        for h in self.hypotheses:
            self.pvalue_pseudothreshold[h] = 0.
            if not self.nested_hypotheses[h]:
                self.hypothesis_rejected[h] = False

        dynamic_local_significance = _copy.copy(self.local_significance)
        dynamic_null_hypothesis = list(_copy.copy(self.hypotheses))
        dynamic_passing_graph = self.passing_graph.copy()
        self.significance_tested_at = {}
        for h in self.hypotheses:
            if not self.nested_hypotheses[h]:
                self.significance_tested_at[h] = 0.
        # Test the non-nested hypotheses. This can potentially pass significance on
        # to the nested hypotheses, so these are tested after this (the nested
        # hypotheses never pass significance out of them so can be tested last in any
        # order).
        stop = False
        while not stop:
            stop = True
            for h in dynamic_null_hypothesis:
                if not self.nested_hypotheses[h]:
                    if dynamic_local_significance[h] > self.significance_tested_at[h]:
                        self.significance_tested_at[h] = dynamic_local_significance[h]
                        self.pvalue_pseudothreshold[h] = dynamic_local_significance[h]
                    if self.pvalues[h] <= dynamic_local_significance[h]:
                        hind = self.hypotheses.index(h)
                        stop = False
                        self.hypothesis_rejected[h] = True
                        del dynamic_null_hypothesis[dynamic_null_hypothesis.index(h)]

                        # Update the local significance, and the significance passing graph.
                        new_passing_passing_graph = _np.zeros(_np.shape(self.passing_graph), float)
                        for l in dynamic_null_hypothesis:
                            lind = self.hypotheses.index(l)

                            dynamic_local_significance[l] = dynamic_local_significance[l] + \
                                dynamic_local_significance[h] * dynamic_passing_graph[hind, lind]
                            for k in dynamic_null_hypothesis:
                                kind = self.hypotheses.index(k)
                                if lind != kind:
                                    a = dynamic_passing_graph[lind, kind] + \
                                        dynamic_passing_graph[lind, hind] * dynamic_passing_graph[hind, kind]
                                    b = 1. - dynamic_passing_graph[lind, hind] * dynamic_passing_graph[hind, lind]
                                    new_passing_passing_graph[lind, kind] = a / b

                        del dynamic_local_significance[h]
                        dynamic_passing_graph = new_passing_passing_graph.copy()

        # Test the nested hypotheses
        for h in self.hypotheses:
            if self.nested_hypotheses[h]:
                self._implement_nested_hypothesis_test(h, dynamic_local_significance[h], self.local_corrections[h])

        return

    def _implement_nested_hypothesis_test(self, hypotheses, significance, correction='Holms'):
        """
        Todo
        """
        for h in hypotheses:
            self.hypothesis_rejected[h] = False

        for h in hypotheses:
            self.significance_tested_at[h] = 0.

        if correction == 'Bonferroni':
            self.pvalue_pseudothreshold[hypotheses] = significance / len(hypotheses)
            for h in hypotheses:
                self.significance_tested_at[h] = significance / len(hypotheses)
                if self.pvalues[h] <= significance / len(hypotheses):
                    self.hypothesis_rejected[h] = True

        elif correction == 'Holms':
            dynamic_hypotheses = list(_copy.copy(hypotheses))
            stop = False
            while not stop:
                stop = True
                for h in dynamic_hypotheses:
                    test_significance = significance / len(dynamic_hypotheses)
                    if self.pvalues[h] <= test_significance:
                        stop = False
                        self.hypothesis_rejected[h] = True
                        del dynamic_hypotheses[dynamic_hypotheses.index(h)]
                    if test_significance > self.significance_tested_at[h]:
                        self.significance_tested_at[h] = test_significance

            self.pvalue_pseudothreshold[hypotheses] = significance / len(dynamic_hypotheses)

        elif correction == 'Hochberg':

            dynamic_hypotheses = list(_copy.copy(hypotheses))
            pvalues = [self.pvalues[h] for h in dynamic_hypotheses]
            pvalues, dynamic_hypotheses = zip(*sorted(zip(pvalues, dynamic_hypotheses)))
            pvalues = list(pvalues)
            dynamic_hypotheses = list(dynamic_hypotheses)
            pvalues.reverse()
            dynamic_hypotheses.reverse()

            #print(pvalues)
            #print(dynamic_hypotheses)

            num_hypotheses = len(pvalues)
            for i in range(num_hypotheses):
                #print(dynamic_hypotheses[0],pvalues[0])
                if pvalues[0] <= significance / (i + 1):
                    for h in dynamic_hypotheses:
                        #print(h)
                        self.hypothesis_rejected[h] = True
                        self.significance_tested_at[h] = significance / (i + 1)

                    self.pvalue_pseudothreshold[hypotheses] = significance / (i + 1)
                    return
                else:
                    self.significance_tested_at[dynamic_hypotheses[0]] = significance / (i + 1)
                    del pvalues[0]
                    del dynamic_hypotheses[0]

            # If no nulls rejected, the threshold is the Bonferroni threshold
            self.pvalue_pseudothreshold[hypotheses] = significance / num_hypotheses

        elif correction == 'Benjamini-Hochberg':
            # print("Warning: the family-wise error rate is not being controlled! "
            #       "Instead the False discovery rate is being controlled")
            dynamic_hypotheses = list(_copy.copy(hypotheses))
            pvalues = [self.pvalues[h] for h in dynamic_hypotheses]
            pvalues, dynamic_hypotheses = zip(*sorted(zip(pvalues, dynamic_hypotheses)))
            pvalues = list(pvalues)
            dynamic_hypotheses = list(dynamic_hypotheses)
            pvalues.reverse()
            dynamic_hypotheses.reverse()

            num_hypotheses = len(pvalues)
            for i in range(num_hypotheses):
                #print(dynamic_hypotheses[0],pvalues[0])
                if pvalues[0] <= significance * (num_hypotheses - i) / num_hypotheses:
                    for h in dynamic_hypotheses:
                        #print(h)
                        self.hypothesis_rejected[h] = True
                        self.significance_tested_at[h] = significance * (num_hypotheses - i) / num_hypotheses

                    self.pvalue_pseudothreshold[hypotheses] = significance * (num_hypotheses - i) / num_hypotheses
                    return
                else:
                    self.significance_tested_at[dynamic_hypotheses[0]] = significance * \
                        (num_hypotheses - i) / num_hypotheses
                    del pvalues[0]
                    del dynamic_hypotheses[0]

            # If no nulls rejected, the threshold is the Bonferroni threshold
            self.pvalue_pseudothreshold[hypotheses] = significance / num_hypotheses

        elif correction == 'none':
            # print("Warning: the family-wise error rate is not being controlled, "
            #       "as the correction specified for this nested hypothesis is 'none'!")
            self.pvalue_pseudothreshold[hypotheses] = significance
            for h in hypotheses:
                self.significance_tested_at[h] = significance
                if self.pvalues[h] <= significance:
                    self.hypothesis_rejected[h] = True

        else:
            raise ValueError("The choice of `{}` for the `correction` parameter is invalid.".format(correction))
