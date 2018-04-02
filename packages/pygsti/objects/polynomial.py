""" Defines the Polynomial class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************



class Polynomial(object):
    """ Encapsulates a polynomial """
    def __init__(self, nVariables):
        self.num_variables = nVariables # "named" by integers 0 to nVariables-1
        self.coeffs = {}

    def add(self, coeff, variable_exponents):
        pass # TODO

    def deriv(self):
        pass # TODO - returns another polynomial

    def evaluate(self, variable_values):
        pass # TODO
