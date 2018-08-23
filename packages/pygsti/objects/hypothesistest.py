""" Defines HypothesisTest object and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy

# -- To be put back in when we allow for non-p-value pvalues --
# def pvalue_threshold_function(hypothesisname, p, significance):
# 	"""
# 	Todo
# 	"""
# 	if p < significance: 
# 		return True
# 	else: 
# 		return False

#class NestedHypothesisTest(object)
##
#	 def __init__(self, hypotheses, correction='Holms', significance=0.05):

class HypothesisTest(object):
    """
    An object to define a set of statistical hypothesis tests on a 
    set of null hypotheses.
    """   
    def __init__(self, hypotheses, significance=0.05, weighting='equal', 
    			 passing_graph='Holms', local_corrections='Holms'):
    	"""
    	Todo :

		Parameters
		----------
		nullhypotheses : ?
			?

		significance : float, optional
			The significance level. (0.05 is 95% confidence, using the standard convention not the
			current convention in pyGSTi / our papers).

		control : str, optional
			The general form of the multi-test false-detection-probability control. The options are:

				- "FWER": Corresponds to 'Family-wise error rate', meaning ...
				- "none": Corresponds to no corrections

    	"""
    	assert(0. < significance and significance < 1.), 'The significance level in a hypotheses test must be > 0 and < 1!'

    	self.hypotheses = hypotheses
    	self.significance = significance
    	self.pvalues = None
    	self.hypothesis_rejected = None
    	self.significance_tested_at = None
    	self.pvalue_pseudothreshold = None

    	self.nested_hypotheses = {}
    	for h in self.hypotheses:
    		if not (isinstance(h,tuple) or isinstance(h,list)):
    			self.nested_hypotheses[h]=False
    		else:
    			self.nested_hypotheses[h]=True

    	if isinstance(passing_graph,str):
    		assert(passing_graph == 'Holms')
    		self._initialize_to_weighted_holms_test()

    	self.local_significance = {}
    	if isinstance(weighting,str):
    		assert(weighting == 'equal')
    		for h in self.hypotheses:
    			self.local_significance[h] = self.significance/len(self.hypotheses)
    	else:
    		totalweight = 0.
    		for h in self.hypotheses:
    			totalweight += weighting[h]
    		for h in self.hypotheses:
    			self.local_significance[h] = significance*weighting[h]/totalweight

    	if isinstance(local_corrections,str): 
    		assert(local_corrections in ('Holms','Hochberg','Bonferroni','none')), "A local correction of `{}` is not a valid choice".format(local_corrections)
    		self.local_corrections = {}
    		for h in self.hypotheses:
    			if self.nested_hypotheses[h]:
    				self.local_corrections[h] = local_corrections
    	else:
    		self.local_corrections = local_corrections

    	self._check_permissible()

    	# if is not isinstance(threshold_function,str):
    	# 	raise ValueError ("Data that is not p-values is currently not supported!")
    	# else:
    	# 	if threshold_function is not 'pvalue':
    	# 		raise ValueError ("Data that is not p-values is currently not supported!")

    	return

    def _initialize_to_weighted_holms_test(self):
    	"""
		Todo
    	"""
    	self.passing_graph = _np.zeros((len(self.hypotheses),len(self.hypotheses)),float)
    	for hind, h in enumerate(self.hypotheses):
    		if not self.nested_hypotheses[h]:
    			self.passing_graph[hind,:] = _np.ones(len(self.hypotheses),float)/(len(self.hypotheses)-1)
    			self.passing_graph[hind,hind] = 0.

    def _check_permissible(self):
    	"""
		Todo
    	"""
    	# Todo : test that the graph is acceptable.
    	return True

    def add_pvalues(self, pvalues):
    	"""
		Insert the p-values for the hypotheses.

		Parameters
		----------
		pvalues : dict
			todo

		Returns
		-------
		None
    	"""
    	
    	# Testing this is slow, so we'll just leave it out.
    	#for h in self.hypotheses:
    	#	if self.nested_hypotheses[h]:
    	#		for hh in h:
    	#			assert(hh in list(pvalues.keys())), "Some hypothese do not have a pvalue in this pvalue dictionary!"
	    #			assert(pvalues[hh] >= 0. and pvalues[hh] <= 1.), "Invalid p-value!" 
    	#	else:
	    #		assert(h in list(pvalues.keys())), "Some hypothese do not have a pvalue in this pvalue dictionary!"
	    #		assert(pvalues[h] >= 0. and pvalues[h] <= 1.), "Invalid p-value!"

    	self.pvalues = _copy.copy(pvalues)

    	return

    def implement(self):
    	"""
		Implements the multiple hypothesis testing routine encoded by this object. This populates
		the self.hypothesis_rejected dictionary, that shows which hypotheses can be rejected using
		the procedure specified.
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
    		most_significant_index = []
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
    					new_passing_passing_graph = _np.zeros(_np.shape(self.passing_graph),float)
    					for l in dynamic_null_hypothesis:
    						lind = self.hypotheses.index(l)

    						dynamic_local_significance[l] = dynamic_local_significance[l] + dynamic_local_significance[h]*dynamic_passing_graph[hind,lind]
    						for k in dynamic_null_hypothesis:
    							kind = self.hypotheses.index(k)
    							if lind != kind:
    								a = dynamic_passing_graph[lind,kind] + dynamic_passing_graph[lind,hind]*dynamic_passing_graph[hind,kind]
    								b = 1. - dynamic_passing_graph[lind,hind]*dynamic_passing_graph[hind,lind]			
    								new_passing_passing_graph[lind,kind] =  a/b

    					del dynamic_local_significance[h]
    					dynamic_passing_graph =	new_passing_passing_graph.copy()
 	
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
    		self.pvalue_pseudothreshold[hypotheses] = significance/len(hypotheses)
	    	for h in hypotheses:
	    		self.significance_tested_at[h] = significance/len(hypotheses)
	    		if self.pvalues[h] <= significance/len(hypotheses):
	    			self.hypothesis_rejected[h] = True

    	elif correction == 'Holms':
    		dynamic_hypotheses = list(_copy.copy(hypotheses))
    		stop = False
    		while not stop:
    			stop = True
    			for h in dynamic_hypotheses:
    				test_significance = significance/len(dynamic_hypotheses)
    				if self.pvalues[h] <= test_significance:
    					stop = False
    					self.hypothesis_rejected[h] = True
    					del dynamic_hypotheses[dynamic_hypotheses.index(h)]
    				if test_significance > self.significance_tested_at[h]:
    					self.significance_tested_at[h] = test_significance

    		self.pvalue_pseudothreshold[hypotheses] = significance/len(dynamic_hypotheses)

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
    			if pvalues[0] <= significance/(i + 1):
    				for h in dynamic_hypotheses:
    					#print(h)
    					self.hypothesis_rejected[h] = True
    					self.significance_tested_at[h] = significance/(i + 1)

    				self.pvalue_pseudothreshold[hypotheses] = significance/(i + 1)
    				return
    			else:
    				self.significance_tested_at[dynamic_hypotheses[0]] = significance/(i + 1)
    				del pvalues[0]
    				del dynamic_hypotheses[0]
 				
    	elif correction == 'none':
    		print("Warning: the family-wise error rate is not being controlled, as the correction specified for this nested hypothesis is 'none'!")
    		self.pvalue_pseudothreshold[hypotheses] = significance
    		for h in hypotheses:
	    		self.significance_tested_at[h] = significance
	    		if self.pvalues[h] <= significance:
	    			self.hypothesis_rejected[h] = True
   				
    	else: 
    		raise ValueError("The choice of `{}` for the `correction` parameter is invalid.".format(correction))
    #def any_hypotheses_rejected():
    #	assert(self.results is not None), "Test must be implemented before results can be queried!"
    #	return