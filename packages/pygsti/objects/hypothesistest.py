""" Defines HypothesisTest object and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy

# -- To be put back in when we allow for non-p-value data --
# def pvalue_threshold_function(hypothesisname, p, alpha):
# 	"""
# 	Todo
# 	"""
# 	if p < alpha: 
# 		return True
# 	else: 
# 		return False

#class NestedHypothesisTest(object)
##
#	 def __init__(self, null_hypotheses, correction='Holms', alpha=0.05):

class HypothesisTest(object):
    """
    An object to define a set of statistical hypothesis tests on a 
    set of null hypotheses.
    """   
    def __init__(self, null_hypotheses, weighting='equal', 
    			 passing_graph='Holms', local_corrections='Holms', 
    			 alpha=0.05, control='FWER'):
    	"""
		Parameters
		----------
		nullhypotheses : ?
			?

		alpha : float, optional
			The significance level. (0.05 is 95% confidence, using the standard convention not the
			current convention in pyGSTi / our papers).

		control : str, optional
			The general form of the multi-test false-detection-probability control. The options are:

				- "FWER": Corresponds to 'Family-wise error rate', meaning ...
				- "none": Corresponds to no corrections

    	"""
    	self.null_hypotheses = null_hypotheses
    	self.alpha = alpha
    	self.control = control
    	self.data = None
    	self.hypothesis_rejected = None

    	self.nested_hypotheses = {}
    	for h in self.null_hypotheses:
    		if not (isinstance(h,tuple) or isinstance(h,list)):
    			self.nested_hypotheses[h]=False
    		else:
    			self.nested_hypotheses[h]=True

    	if isinstance(passing_graph,str):
    		assert(passing_graph == 'Holms')
    		self.initialize_to_weighted_holms_test()
    	self.local_corrections = local_corrections

    	self.local_alpha = {}
    	if isinstance(weighting,str):
    		assert(weighting == 'equal')
    		for h in self.null_hypotheses:
    			self.local_alpha[h] = self.alpha/len(self.null_hypotheses)
    	else:
    		totalweight = 0.
    		for h in self.null_hypotheses:
    			totalweight += weighting[h]
    		print(totalweight)
    		for h in self.null_hypotheses:
    			self.local_alpha[h] = alpha*weighting[h]/totalweight

    	if isinstance(local_corrections,str):
    		assert(local_corrections == 'Holms')
    		self.local_corrections = {}
    		for h in self.null_hypotheses:
    			if self.nested_hypotheses[h]:
    				self.local_corrections[h] = 'Holms'

    	self.check_permissible()

    	# if is not isinstance(threshold_function,str):
    	# 	raise ValueError ("Data that is not p-values is currently not supported!")
    	# else:
    	# 	if threshold_function is not 'pvalue':
    	# 		raise ValueError ("Data that is not p-values is currently not supported!")

    	return

    def initialize_to_weighted_holms_test(self):

    	self.passing_graph = _np.zeros((len(self.null_hypotheses),len(self.null_hypotheses)),float)
    	for hind, h in enumerate(self.null_hypotheses):
    		if not self.nested_hypotheses[h]:
    			self.passing_graph[hind,:] = _np.ones(len(self.null_hypotheses),float)/(len(self.null_hypotheses)-1)
    			self.passing_graph[hind,hind] = 0.

    def check_permissible(self):
    	"""
		Todo
    	"""
    	# Todo : test that the graph is acceptable.
    	return True

    def add_data(self, data):
    	"""
		Todo
    	"""
    	self.data = data

    def implement(self):
    	assert(self.data is not None), "Data must be input before the test can be implemented!"

    	self.hypothesis_rejected = {}
    	for h in self.null_hypotheses:
    		if not self.nested_hypotheses[h]:
    			self.hypothesis_rejected[h] = False

    	dynamic_local_alpha = _copy.copy(self.local_alpha)
    	dynamic_null_hypothesis = list(_copy.copy(self.null_hypotheses))
    	dynamic_passing_graph = self.passing_graph.copy()
    	self.highest_alpha_tested_at = {}
    	for h in self.null_hypotheses:
    		if not self.nested_hypotheses[h]:
    			self.highest_alpha_tested_at[h] = 0.
    	#print(dynamic_local_alpha)
    	#print(dynamic_passing_graph)
    	# Test the non-nested hypotheses
    	stop = False
    	while not stop:
    		stop = True
    		most_significant_index = []
    		for h in dynamic_null_hypothesis:
    			if not self.nested_hypotheses[h]:
    				#print(h)
    				if dynamic_local_alpha[h] > self.highest_alpha_tested_at[h]:
    					self.highest_alpha_tested_at[h] = dynamic_local_alpha[h]
    				if self.data[h] <= dynamic_local_alpha[h]:
    					#print("Rejected")
    					hind = self.null_hypotheses.index(h)
    					stop = False
    					self.hypothesis_rejected[h] = True
    					del dynamic_null_hypothesis[dynamic_null_hypothesis.index(h)]  					
    					# Update the local alpha, and the alpha passing graph.
    					new_passing_passing_graph = _np.zeros(_np.shape(self.passing_graph),float)
    					for l in dynamic_null_hypothesis:
    						lind = self.null_hypotheses.index(l)
    						#print(dynamic_local_alpha[h])
    						#print(hind,lind)
    						#print(dynamic_passing_graph[hind,lind])
    						dynamic_local_alpha[l] = dynamic_local_alpha[l] + dynamic_local_alpha[h]*dynamic_passing_graph[hind,lind]
    						for k in dynamic_null_hypothesis:
    							kind = self.null_hypotheses.index(k)
    							if lind != kind:
    								a = dynamic_passing_graph[lind,kind] + dynamic_passing_graph[lind,hind]*dynamic_passing_graph[hind,kind]
    								b = 1. - dynamic_passing_graph[lind,hind]*dynamic_passing_graph[hind,lind]			
    								new_passing_passing_graph[lind,kind] =  a/b

    					del dynamic_local_alpha[h]
    					dynamic_passing_graph =	new_passing_passing_graph.copy()
    				#print(dynamic_local_alpha)
    				#print(dynamic_passing_graph)
 	
    	# Test the nested hypotheses
    	for h in self.null_hypotheses:
    		if self.nested_hypotheses[h]:
    			self.implement_nested_hypothesis_test(h, dynamic_local_alpha[h], self.local_corrections[h])

    	return

    def implement_nested_hypothesis_test(self, hypotheses, alpha, correction='Holms'):

    	print(hypotheses)
    	print(alpha)
    	for h in hypotheses:
    		self.hypothesis_rejected[h] = False

    	for h in hypotheses:
    		self.highest_alpha_tested_at[h] = 0.

    	if correction == 'Bonferroni':
	    	for h in hypotheses:
	    		self.highest_alpha_tested_at[h] = alpha/len(hypotheses)
	    		if data[h] <= alpha/len(hypotheses):
	    			self.hypothesis_rejected[h] = True

    	elif correction == 'Holms':
    		dynamic_hypotheses = list(_copy.copy(hypotheses))
    		stop = False
    		while not stop:
    			stop = True
    			for h in dynamic_hypotheses:
    				test_alpha = alpha/len(dynamic_hypotheses)
    				if self.data[h] <= test_alpha:
    					stop = False
    					self.hypothesis_rejected[h] = True
    					del dynamic_hypotheses[dynamic_hypotheses.index(h)]
    				if test_alpha > self.highest_alpha_tested_at[h]:
    					self.highest_alpha_tested_at[h] = test_alpha

    	elif correction == 'Hockberg':
    		raise ValueError("Not implemented yet!")

    #def any_hypotheses_rejected():
    #	assert(self.results is not None), "Test must be implemented before results can be queried!"
    #	return