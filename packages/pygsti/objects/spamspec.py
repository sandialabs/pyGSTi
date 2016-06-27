from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the SpamSpec class and supporting functions """
from . import gatestring as _gatestring

class SpamSpec(object):
    """
    Encapsulates a rho- or E-vector label paired with
    a gate string, which serves as either a "prep-specifier" or
    an "effect-specifier" respectively.  A prep-specifier means a state
    preparation followed by the gate string, while an effect-specifier
    means a gate string followed by a measurement (Note: gate
    strings are performed in left-to-right order!)
    """

    def __init__(self,label,gatestring):
        """
        Create a new SpamSpec object

        Parameters
        ----------
        label : str
          rho- or E-vector label

        gatestring : tuple or GateString
          gate string, evaluated in left-to-right order, which
          precedes or follows the E- or rho-vector indexed by
          index, respectively.
        """
        self.lbl = label
        self.str = _gatestring.GateString(gatestring)
          #this makes sure self.str is always a gatestring object

    def __str__(self):
        return str(self.str)+"("+str(self.lbl)+")"

    def __eq__(self,x):
        try:
            if self.str == x.str and self.lbl == x.lbl:
                return True
        except: pass
        return False
