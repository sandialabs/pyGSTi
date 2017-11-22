""" Defines the ReportTable class """
from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from collections  import OrderedDict   as _OrderedDict
from .row         import Row
from .convert import convertDict as _convertDict

class ReportFigure(object):
    '''
    A report figure, encapsulating a plotly figure and related metadata that
    can be rendered in a variety of formats.
    '''
    def __init__(self, plotlyfig, colormap=None, pythonValue=None, **kwargs):
        '''
        Create a table object

        Parameters
        ----------
        plotlyfig : plotly.Figure
            The plotly figure to encapsulate
        
        colormap : ColorMap, optional
            A pygsti color map object used for this figure.

        pythonValue : object, optional
            A python object to be used as the Python-version of
            this figure (usually the data being plotted in some 
            convenient format).

        kwargs : dict
            Additional meta-data relevant to this figure
        '''
        self.plotlyfig = plotlyfig
        self.colormap = colormap
        self.pythonvalue = pythonValue
        self.metadata = dict(kwargs).copy()
        
