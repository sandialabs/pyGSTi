from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from .formatters import formatDict  as _formatDict
from .convert    import convertDict as _convertDict

class Cell(object):
    def __init__(self, data=None, formatterName=None, label=None):
        self.data          = data
        self.formatterName = formatterName
        self.label         = label

    def _render_data(self, fmt, spec):
        if self.formatterName is not None:
            formatter = _formatDict[self.formatterName]
            formatted_item = formatter[fmt](self.data, spec)
            assert formatted_item is not None, ("Formatter " + str(type(formatter[fmt]))
                                              + " returned None for item = " + str(self.data))
            return formatted_item
        else:
            if self.data.get_value() is not None:
                return str(self.data)
            else:
                raise ValueError("Unformatted None in Cell")

    def render(self, fmt, spec):
        format_cell   = _convertDict[fmt]['cell'] # Function for rendering a cell in the format "fmt"
        formattedData = self._render_data(fmt, spec)

        return format_cell(formattedData, self.label, spec)
