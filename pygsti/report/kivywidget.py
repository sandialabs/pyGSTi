"""
Routines for converting python objects to Kivy display widgets.

Parallel rountines as html.py has for HTML conversion.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import cmath

import os as _os
import numpy as _np
import marshal as _marshal
#import tempfile as _tempfile

from pygsti.io.metadir import _class_for_name
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


'''
table() and cell() functions are used by table.py in table creation
everything else is used in creating formatters in formatters.py
'''


class KivyWidgetFactory(_NicelySerializable):
    pass

    #REMOVE (covered by NicelySerializable)
    #@classmethod
    #def from_json_dict(cls, json_dict):
    #    cls_to_create = _class_for_name(json_dict['factory_class'])
    #    return cls_to_create._from_json_dict(json_dict)
    #
    #def add_class_name(self, json_dict):
    #    json_dict['factory_class'] = _full_class_name(self)


class TableWidgetFactory(KivyWidgetFactory):
    '''
    can be 'rendered' to, and contains all the information to quickly and easily create
    a TableWidget on the front end AND can be serialized easily.

    Need to create similar kernel "shadows" of widgets that can be serialized and passed to front end for displaying.
    Similar for figures too -- maybe need to make classes for each type of plot (?)
    '''
    def __init__(self, formatted_headings, formatted_rows, spec):
        super().__init__()
        self.padding = 2
        self.spacing = 5
        self.row_factories = formatted_rows  # "formatted" with 'kivywidget' type means => factories
        self.heading_factories = formatted_headings
        self.spec = spec

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'padding': self.padding,
                      'spacing': self.spacing,
                      'heading_factories': [hf.to_nice_serialization() for hf in self.heading_factories],
                      'row_factories': [[cf.to_nice_serialization() for cf in row] for row in self.row_factories],
                      'spec': {}  # TODO - serialize spec?!
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        heading_factories = [KivyWidgetFactory.from_nice_serialization(d) for d in state['heading_factories']]
        row_factories = [[KivyWidgetFactory.from_nice_serialization(d) for d in row] for row in state['row_factories']]
        spec = state['spec']
        return cls(heading_factories, row_factories, spec)

    def create_widget(self):
        from .kivyfrontend import TableWidget  # do NOT import at top level
        heading_cell_widgets = [cell_factory.create_widget() for cell_factory in self.heading_factories]
        row_cell_widgets = [[cell_factory.create_widget() for cell_factory in row] for row in self.row_factories]
        return TableWidget(heading_cell_widgets, row_cell_widgets, self.spec)


class CellWidgetContainerFactory(KivyWidgetFactory):

    def __init__(self, widget_factory, hover_text=None):
        super().__init__()
        self.widget_factory = widget_factory
        self.hover_text = hover_text

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'widget_factory': self.widget_factory.to_nice_serialization(),
                      'hover_text': self.hover_text
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        widget_factory = KivyWidgetFactory.from_nice_serialization(state['widget_factory'])
        return cls(widget_factory, state['hover_text'])

    def create_widget(self):
        from .kivyfrontend import CellWidgetContainer  # do NOT import at top level
        widget = self.widget_factory.create_widget()
        return CellWidgetContainer(widget, self.hover_text)


class LatexWidgetFactory(KivyWidgetFactory):

    def __init__(self, latex_string):
        super().__init__()
        self.latex_string = latex_string

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'latex_string': self.latex_string})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(state['latex_string'])

    def create_widget(self):
        from .kivyfrontend import LatexWidget  # do NOT import at top level
        return LatexWidget(self.latex_string)


class AdjustingLabelFactory(KivyWidgetFactory):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'args': self._args,
                      'kwargs': self._kwargs
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(*state['args'], **state['kwargs'])

    def create_widget(self):
        from .kivyfrontend import AdjustingLabel  # do NOT import at top level
        PADDING = 35
        widget = AdjustingLabel(*self._args, **self._kwargs)
        widget.size = (widget.texture_size[0] + PADDING, widget.texture_size[1] + PADDING)
        print("AdjustingLabel widget => ", widget.size, ':', widget.text)
        return widget


# Used within workspaceplots.py to build Kivy plots
class KivyPlotFactory(KivyWidgetFactory):

    def __init__(self, class_to_construct=None, constructor_fn=None, numpy_args=None, native_args=None,
                 serializable_args=None, function_args=None, natural_size=None):
        super().__init__()
        self.class_to_construct = class_to_construct  # holds a *string* that names a class to construct
        self.constructor_fn = constructor_fn  # holds a *string* that names a function to execute
        #Note: hold strings in above attributes so that a KivyPlotFactory object can be created
        # and used in a backend environment where constructing kivy widgets is forbidden (because
        # importing kivy results in a graphics lockup).  Only when create_widget is called is a
        # kivy widget class instantiated, and we never call create_widget in the backend.

        self.numpy_args = numpy_args
        self.native_args = native_args
        self.serializable_args = serializable_args
        self.function_args = function_args
        self.natural_size = natural_size


    def _to_nice_serialization(self):
        if self.function_args is not None:
            function_args = {}  # Note: this sort of works, but doesn't for our functions that have weird closures...
            for k, fn in self.function_args.items():
                function_args[k] = (_marshal.dumps(fn.__code__).decode('latin-1'), fn.__name__)
        else:
            function_args = None

        if self.numpy_args is not None:
            def encode_numpy(x):
                if isinstance(x, _np.ndarray): return self._encodemx(x)
                elif isinstance(x, python_list): return {'numpy_encoded': 'list',
                                                         'elements': [encode_numpy(el) for el in x]}
                else: raise ValueError("Cannot encode numpy argument! (reached %s type)" % str(type(x)))
            numpy_args = {k: encode_numpy(v) for k, v in self.numpy_args.items()}
        else:
            numpy_args = None

        serializable_args = {} if (self.serializable_args is None) \
            else {k: v.to_nice_serialization() for k, v in self.serializable_args.items()}

        state = super()._to_nice_serialization()
        state.update({'class_to_construct': self.class_to_construct,
                      'constructor_fn': self.constructor_fn,
                      'numpy_args': numpy_args,
                      'native_args': self.native_args,
                      'serializable_args': serializable_args,
                      'function_args': function_args,
                      'natural_size': self.natural_size
                      })
        #OLD REMOVE - when these attributes were either a class or function, respectively
        # (now these attributes hold strings which are resolved in create_widget)
        #'class_to_construct': (self.class_to_construct.__module__ + '.' + self.class_to_construct.__name__
        #                       if (self.class_to_construct is not None) else None),
        #'constructor_fn': (self.constructor_fn.__module__ + '.' + self.constructor_fn.__name__
        #                   if (self.constructor_fn is not None) else None),
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        class_to_construct = state['class_to_construct']
        constructor_fn = state['constructor_fn']

        if state['numpy_args'] is not None:
            def decode_numpy(x):
                if isinstance(x, dict) and 'numpy_encoded' in x and x['numpy_encoded'] == 'list':
                    return [decode_numpy(el) for el in x['elements']]
                else:
                    return cls._decodemx(x)
            numpy_args = {k: decode_numpy(v) for k, v in state['numpy_args'].items()}
        else:
            numpy_args = None

        if state['serializable_args'] is not None:
            serializable_args = {k: _NicelySerializable.from_nice_serialization(v)
                                 for k, v in state['serializable_args'].items()}
        else:
            serializable_args = None

        if state['function_args'] is not None:
            import types as _types
            function_args = {k: _types.FunctionType(_marshal.loads(v[0].encode('latin-1')), globals(), v[1])
                             for k, v in state['function_args'].items()}
        else:
            function_args = None

        return cls(class_to_construct, constructor_fn, numpy_args, state['native_args'],
                   serializable_args, function_args, natural_size=state['natural_size'])

    def create_widget(self):
        kwargs = {}
        if self.numpy_args is not None: kwargs.update(self.numpy_args)
        if self.native_args is not None: kwargs.update(self.native_args)
        if self.serializable_args is not None: kwargs.update(self.serializable_args)
        if self.function_args is not None: kwargs.update(self.function_args)
        kwargs.update({'size_hint': (None, None)})

        if self.class_to_construct is not None:
            cls = _class_for_name(self.class_to_construct)
            widget = cls(**kwargs)
        elif self.constructor_fn is not None:
            fn = _class_for_name(self.constructor_fn)
            widget = fn(**kwargs)
        else:
            raise ValueError("Cannot create widget when 'class_to_construct' and 'constructor_fn' are both None!")

        widget.size = self.natural_size if (self.natural_size is not None) else (300, 300)
        return widget


def table(custom_headings, col_headings_formatted, rows, spec):
    """
    Create an HTML table

    Parameters
    ----------
    custom_headings : None, dict
        optional dictionary of custom table headings

    col_headings_formatted : list
        formatted column headings

    rows : list of lists of cell-strings
        Data in the table, pre-formatted

    spec : dict
        options for the formatter

    Returns
    -------
    dict : contains keys 'html' and 'js', which correspond to a html and js strings representing the table
    """

    #assert(col_headings_formatted is None), "Can't deal with custom column heading yet"
    headings = col_headings_formatted  # custom_headings
    widget_factory = TableWidgetFactory(headings, rows, spec)  # **spec['kivywidget_kwargs']
    return {'kivywidget': widget_factory}


def cell(data, label, spec):
    """
    Format the cell of an HTML table

    Parameters
    ----------
    data : string
        string representation of cell content

    label : string
        optional cell label, used for tooltips

    spec : dict
        options for the formatters

    Returns
    -------
    string
    """
    if isinstance(data, KivyWidgetFactory):
        #print("Other widget (%s) => %s" % (str(type(data)), str(data.size)))
        print("Other widget (%s)" % str(type(data)))
        return CellWidgetContainerFactory(data)  # assume widget size is already set
    elif isinstance(data, str) and ('$' in data):

        COL_SIZE = 100; LINE_SIZE = 80; CHAR_SIZE = 8
        if 'num_lines' in spec and 'num_cols' in spec:
            w, h = (COL_SIZE * spec['num_cols'], LINE_SIZE * spec['num_lines'])
            print("Latex widget w/ lines & cols: ", spec['num_lines'], spec['num_cols'])
        else:
            h = LINE_SIZE  # assume a single line
            w = len(data) * CHAR_SIZE  # rough estimate of width
            print("Latex widget w/ %d chars." % len(data))

        # render latex via LatexWidget
        widget = CellWidgetContainerFactory(LatexWidgetFactory(data), label) #, size_hint=(None, None), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #widget.content.size = (w, h)
        #print("Latex widget => ", widget.content.size, ':', widget.content.latex_string)
        return widget
    else:
        #PADDING = 35
        txt = str(data)
        if txt.startswith('**') and txt.endswith('**'):
            txt = txt[2:-2]; bold = True
        else: bold = False
        widget = AdjustingLabelFactory(text=txt, color=(0,0,0,1), bold=bold)  # use label arg?
        #widget = Label(text=str(data), color=(0,0,0,1))  # use label arg?
        #widget.texture_update()
        #widget.size = (widget.texture_size[0] + PADDING, widget.texture_size[1] + PADDING)
        #print("AdjustingLabel widget => ", widget.size, ':', widget.text)
        return CellWidgetContainerFactory(widget, label)


python_list = list  # so we can still access usual list object...


def list(l, specs):
    """
    Convert a list to html.

    Parameters
    ----------
    l : list
        list to convert into HTML. sub-items pre formatted

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for l.
    """

    return "<br>".join(l)


def vector(v, specs):
    """
    Convert a 1D numpy array to html.

    Parameters
    ----------
    v : numpy array
        1D array to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for v.
    """
    lines = []
    specs['num_lines'] = len(v)
    specs['num_cols'] = 1
    #import bpdb; bpdb.set_trace()
    for el in v:
        lines.append(value(el, specs, mathmode=True))
    if specs['brackets']:
        return "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"
    else:
        return "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"


def matrix(m, specs):
    """
    Convert a 2D numpy array to html.

    Parameters
    ----------
    m : numpy array
        2D array to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
        html string for m.
    """
    lines = []
    prefix = ""
    fontsize = specs['fontsize']
    specs['num_lines'] = m.shape[0]
    specs['num_cols'] = m.shape[1]

    if fontsize is not None:
        prefix += "\\fontsize{%f}{%f}\selectfont " % (fontsize, fontsize * 1.2)

    for r in range(m.shape[0]):
        lines.append(" & ".join(
            [value(el, specs, mathmode=True) for el in m[r, :]]))

    if specs['brackets']:
        return prefix + "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"
    else:
        return prefix + "$ \\begin{pmatrix}\n" + \
            " \\\\ \n".join(lines) + "\n \end{pmatrix} $\n"


def value(el, specs, mathmode=False):
    """
    Convert a floating point or complex value to html.

    Parameters
    ----------
    el : float or complex
        Value to convert into HTML.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    mathmode : bool, optional
        Whether this routine should generate HTML for use within a math-rendered
        HTML element (rather than a normal plain-HTML one).  So when set to True
        output is essentially the same as latex format.

    Returns
    -------
    string
        html string for el.
    """

    # ROUND = digits to round values to
    TOL = 1e-9  # tolerance for printing zero values

    precision = specs['precision']
    sciprecision = specs['sciprecision']
    polarprecision = specs['polarprecision']
    complexAsPolar = specs['complex_as_polar']

    def render(x):
        """Render a single float (can be real or imag part)"""
        if abs(x) < 5 * 10**(-(precision + 1)):
            s = "%.*e" % (sciprecision, x)
        elif abs(x) < 1:
            s = "%.*f" % (precision, x)
        elif abs(x) <= 10**precision:
            s = "%.*f" % (precision - int(_np.log10(abs(x))), x)  # round to get precision+1 digits when x is > 1
        else:
            s = "%.*e" % (sciprecision, x)

        #Fix scientific notition
        p = s.split('e')
        if len(p) == 2:
            ex = str(int(p[1]))  # exponent without extras (e.g. +04 => 4)
            if mathmode:  # don't use <sup> in math mode
                s = p[0] + "\\times 10^{" + ex + "}"
            else:
                s = p[0] + "&times;10<sup>" + ex + "</sup>"

        #Strip superfluous endings
        if "." in s:
            while s.endswith("0"): s = s[:-1]
            if s.endswith("."): s = s[:-1]
        return s

    if isinstance(el, str):
        return el
    if type(el) in (int, _np.int64):
        return "%d" % el
    if el is None or _np.isnan(el): return "--"

    try:
        if abs(el.real) > TOL:
            if abs(el.imag) > TOL:
                if complexAsPolar:
                    r, phi = cmath.polar(el)
                    ex = ("i%.*f" % (polarprecision, phi / _np.pi)) if phi >= 0 \
                        else ("-i%.*f" % (polarprecision, -phi / _np.pi))
                    if mathmode:  # don't use <sup> in math mode
                        s = "%se^{%s\\pi}" % (render(r), ex)
                    else:
                        s = "%se<sup>%s &pi;</sup>" % (render(r), ex)
                else:
                    s = "%s%s%si" % (render(el.real), '+' if el.imag > 0 else '-', render(abs(el.imag)))
            else:
                s = "%s" % render(el.real)
        else:
            if abs(el.imag) > TOL:
                s = "%si" % render(el.imag)
            else:
                s = "0"
    except:
        s = str(el)

    return s


def escaped(txt, specs):
    """
    Escape txt so it is html safe.

    Parameters
    ----------
    txt : string
        value to escape

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
    """
    return txt
