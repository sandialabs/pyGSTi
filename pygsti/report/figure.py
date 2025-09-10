"""
Defines the ReportTable class
"""

#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class ReportFigure(object):
    """
    A  plotly figure and related metadata that can be rendered in a variety of formats.

    For use in pyGSTi reports.

    Parameters
    ----------
    plotlyfig : plotly.Figure
        The plotly figure to encapsulate

    colormap : ColorMap, optional
        A pygsti color map object used for this figure.

    python_value : object, optional
        A python object to be used as the Python-version of
        this figure (usually the data being plotted in some
        convenient format).

    kwargs : dict
        Additional meta-data relevant to this figure
    """

    def __init__(self, plotlyfig, colormap=None, python_value=None, **kwargs):
        '''
        Create a table object

        Parameters
        ----------
        plotlyfig : plotly.Figure
            The plotly figure to encapsulate

        colormap : ColorMap, optional
            A pygsti color map object used for this figure.

        python_value : object, optional
            A python object to be used as the Python-version of
            this figure (usually the data being plotted in some
            convenient format).

        kwargs : dict
            Additional meta-data relevant to this figure
        '''
        self.plotlyfig = plotlyfig
        self.colormap = colormap
        self.pythonvalue = python_value
        self.metadata = dict(kwargs).copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self.plotlyfig, 'to_dict'):
            state['plotlyfig'] = {'__plotlydict__': self.plotlyfig.to_dict()}
        return state

    def __setstate__(self, state):
        if isinstance(state['plotlyfig'], dict) and '__plotlydict__' in state['plotlyfig']:
            import plotly.graph_objs as go
            state['plotlyfig'] = go.Figure(state['plotlyfig']['__plotlydict__'])
        self.__dict__.update(state)
