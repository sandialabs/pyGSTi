from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for Python2 / Python3 compatibility """

import numbers as _numbers

#Define basestring in python3 so unicode
# strings can be tested for in python2 using
# python2's built-in basestring type.
# When removing __future__ imports, remove
# this and change basestring => str below.
try:  basestring
except NameError: basestring = str

def isint(x):
    return isinstance(x, _numbers.Integral)

def isstr(x):
    return isinstance(x, basestring)

#Worse way to do this
#import sys as _sys
#
#if _sys.version_info > (3, 0): # Python3?
#    longT = int      # define long and unicode
#    unicodeT = str   #  types to mimic Python2
#else:
#    longT = long
#    unicodeT = unicode
#
#def isint(x):
#    return isinstance(x,(int,longT))
#
#def isstr(x):
#    return isinstance(x,(str,unicodeT))
    
