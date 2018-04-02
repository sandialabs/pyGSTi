""" Defines classes which represent terms in gate expansions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************



class RankOneTerm(object):
    def __init__(self, coeff, pre_op, post_op):
        self.coeff = coeff # potentially a Polynomial
        self.pre_op = pre_op
        self.post_op = post_op
        


    


    
