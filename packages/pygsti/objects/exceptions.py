from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines GST exception classes """

class GSTRuntimeError(Exception):
    """Gate Set Tomography run-time exception class."""
    pass

class GSTValueError(Exception):
    """Gate Set Tomography value error exception class."""
    pass
