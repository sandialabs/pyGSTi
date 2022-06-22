"""
Defines the Cell class
"""

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from pygsti.report.convert import convert_dict as _convert_dict
from pygsti.report.formatters import format_dict as _format_dict


class Cell(object):
    """
    Representation of a table cell, containing formatting and labeling info

    Parameters
    ----------
    data : ReportableQty
        data to be reported

    formatter_name : string, optional
        name of the formatter to be used (ie 'Effect')

    label : string, optional
        label of the cell
    """

    def __init__(self, data=None, formatter_name=None, label=None):
        '''
        Creates Cell object

        Parameters
        ----------
        data : ReportableQty
            data to be reported
        formatter_name : string
            name of the formatter to be used (ie 'Effect')
        label : string
            label of the cell
        '''
        self.data = data
        self.formatterName = formatter_name
        self.label = label

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _render_data(self, fmt, spec):
        '''
        Render self.data as a string

        Parameters
        ----------
        fmt : string
            name of format to be used
        spec: dict
            dictionary of formatting options
        Returns
        -------
        string
        '''
        if self.formatterName is not None:
            formatter = _format_dict[self.formatterName]
            formatted_item = formatter[fmt](self.data, spec)
            assert formatted_item is not None, ("Formatter " + str(type(formatter[fmt]))
                                                + " returned None for item = " + str(self.data))
            return formatted_item
        else:
            if self.data.value is not None:
                return str(self.data)
            else:
                raise ValueError("Unformatted None in Cell")

    def render(self, fmt, spec):
        """
        Render full cell as a string

        Parameters
        ----------
        fmt : string
            name of format to be used

        spec : dict
            dictionary of formatting options

        Returns
        -------
        string
        """
        format_cell = _convert_dict[fmt]['cell']  # Function for rendering a cell in the format "fmt"
        formattedData = self._render_data(fmt, spec)

        # If we aren't given a label and the cell value is a number, use it with full precision as the label:
        if self.label is None and isinstance(self.data.value, (int, float)) or _np.isscalar(self.data.value):
            lbl = str(self.data)
        else:
            lbl = self.label

        return format_cell(formattedData, lbl, spec)
