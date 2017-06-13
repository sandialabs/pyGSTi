from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

class FormatSet():
    '''
    Attributes
    ---------
    formatDict: Static dictionary containing small formatter dictionaries
                Ex: { 'Rho' :  { 'html' : ... , 'text' : ... }, ... }
                (created below)

    Methods
    -------

    __init__: (specs): Specs is a dictionary of the form { 'setting'(kwarg) : value }
                       Ex: { 'precision' : 6, 'polarprecision' : 3, 'sciprecision' : 0 }
                       given to Formatters that need them

    formatList : Given a list of items and formatters and a target format, returns formatted items
    '''
    formatDict = {}

    def __init__(self, specs):
        self.specs = specs

    def formatList(self, items, formatterNames, fmt):
        assert(len(items) == len(formatterNames))
        formatted_items = []

        for item, formatterName in zip(items, formatterNames):
            if formatterName is not None:
                formatter = FormatSet.formatDict[formatterName]
                formatted_item = formatter[fmt](item, self.specs)
                if formatted_item is None:
                    raise ValueError("Formatter " + str(type(formatter[fmt]))
                                     + " returned None for item = " + str(item))
                if isinstance(formatted_item, list): #formatters can create multiple table cells by returning *lists* 
                    formatted_items.extend(formatted_item)
                else:
                    formatted_items.append(formatted_item)
            else:
                if item.get_value() is not None:
                    formatted_items.append(str(item))
                else:
                    raise ValueError("Unformatted None in formatList")

        return formatted_items

