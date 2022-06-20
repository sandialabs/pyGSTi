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
#import tempfile as _tempfile
import xml.etree.ElementTree as _ET
import warnings as _warnings

try:
    import latextools as _latextools
except ImportError:
    _latextools = None

try:
    from kivy.uix.widget import Widget
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.anchorlayout import AnchorLayout
    from kivy.uix.relativelayout import RelativeLayout
    from kivy.uix.scatter import Scatter
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.image import Image

    from kivy.graphics import Color, Rectangle
    from kivy.graphics.svg import Svg
    from kivy.graphics.transformation import Matrix
    from kivy.graphics.context_instructions import Translate
except ImportError:
    GridLayout = object


'''
table() and cell() functions are used by table.py in table creation
everything else is used in creating formatters in formatters.py
'''


class TableWidget(GridLayout):
    def __init__(self, formatted_headings, formatted_rows, spec, **kwargs):
        super(TableWidget, self).__init__(**kwargs)
        self.rows = len(formatted_rows) + 1  # +1 for column headers
        self.cols = len(formatted_headings)
        self.padding = 2
        self.spacing = 5

        # Set size hints for CellWidgetContainers (direct children of
        # this table (a GridLayout) to proportion grid correctly, based
        # on overall row widths and heights relative to table width & height.
        # Set size hints of cell contents to proportion content size relative
        # to cell size.

        #Note: sizes (widths and heights) of cell contents are set BUT these widgets
        # were not created with size_hint=(None,None), so they *will* be resized by
        # their parent later on (this is desirable).  The current sizes just serve as
        # "natural sizes" in order to correctly proportion the table we're now building.

        print("TABLE SIZE COMPUTE:")

        # pass 1: get row and column widths and heights - don't add any widgets yet
        heading_row_width = sum([w.content.width for w in formatted_headings])
        heading_row_height = max([w.content.height for w in formatted_headings])

        row_widths = [sum([w.content.width for w in row]) for row in formatted_rows]
        row_heights = [max([w.content.height for w in row]) for row in formatted_rows]

        ncols = len(formatted_headings)
        column_widths = [max(max([r[j].content.width for r in formatted_rows]), formatted_headings[j].content.width)
                         for j in range(ncols)]
        column_heights = [(sum([r[j].content.height for r in formatted_rows]) + formatted_headings[j].content.height)
                          for j in range(ncols)]

        table_width = max(max(row_widths), heading_row_width)
        table_height = sum(row_heights) + heading_row_height
        assert(table_height >= max(column_heights))  # can have all columns less than table height b/c of row alighmt

        # pass 2: add widgets and set their size hints
        for colwidth, heading_cell_widget in zip(column_widths, formatted_headings):
            heading_cell_widget.size_hint_x = colwidth / table_width
            heading_cell_widget.size_hint_y = heading_row_height / table_height
            heading_cell_widget.content.size_hint_x = heading_cell_widget.content.width / colwidth
            heading_cell_widget.content.size_hint_y = heading_cell_widget.content.height / heading_row_height
            self.add_widget(heading_cell_widget)

        for rowheight, row in zip(row_heights, formatted_rows):
            assert(len(row) == self.cols)
            for colwidth, cell_widget in zip(column_widths, row):
                cell_widget.size_hint_x = colwidth / table_width
                cell_widget.size_hint_y = rowheight / table_height
                cell_widget.content.size_hint_x = cell_widget.content.width / colwidth
                cell_widget.content.size_hint_y = cell_widget.content.height / rowheight
                self.add_widget(cell_widget)

        self.size = (table_width, table_height)
        print("DB: TABLE Initial size = ", self.size)

    def _redraw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.5, 0.5, 0.5)  # table background
            Rectangle(pos=self.pos, size=self.size)

    def on_pos(self, *args):
        self._redraw()

    def on_size(self, *args):
        self._redraw()


class CellWidgetContainer(AnchorLayout):
    def __init__(self, widget, **kwargs):
        super().__init__(**kwargs)
        self.content = widget
        self.add_widget(self.content)

    #def on_size(self, *args):
    #    print("Cell container onsize: ", self.size)


class LatexWidget(Scatter):
    def __init__(self, latex_string, **kwargs):
        kwargs.update(dict(do_rotation=False, do_scale=False, do_translation=False))
        super(LatexWidget, self).__init__(**kwargs)
        if _latextools is None:
            raise ValueError("You must `pip install latextools` and `pip install drawSvg` to render latex within Kivy widgets.")
        pdf = _latextools.render_snippet(latex_string, commands=[_latextools.cmd.all_math])

        self.latex_string = latex_string
        svg_string = pdf.as_svg().content

        #This manual SVG simplification/manipulation should be unnecessary once Svg() in Kivy works properly
        svg_string = svg_string.replace(r'0%', '0')  # used, e.g. in 'rgb(0%,0%,0%)'
        svg_string = svg_string.replace(r'stroke:none', 'stroke-width:0')  # because 'stroke:none' doesn't work (I think because the alpha channel doesn't)
        etree = _ET.ElementTree(_ET.fromstring(svg_string))
        etree = self.simplify_svg_tree(etree)
        with self.canvas:
            self.svg_offset = Translate(0, 0)
            svg = Svg()
            svg.set_tree(etree)

        with self.canvas.before:
            Color(0.7, 0.7, 0.0)
            self.bgrect = Rectangle(pos=(0,0), size=self.size)

        #self.etree = etree
        self.svg_size = (svg.width, svg.height)
        #print("SVG size = ", svg.width, svg.height)
        self.size = svg.width, svg.height

        #REMOVE
        #desired_width = 200.0
        #desired_height = 200.0
        #scalew = desired_width / svg.width  # so scale * svg_width == desired_width
        #scaleh = desired_height / svg.height  # so scale * svg_width == desired_width
        #self.scale = 1.0  #min(scalew, scaleh)
        #print("Scatter size = ", self.size) #, " scale=", self.scale)


        # An alternative to SVG mode above -- very slow and less good -- REMOVE
        #if image_mode:
        #    with _tempfile.TemporaryDirectory() as tmp:
        #        #import bpdb; bpdb.set_trace()
        #        temp_filename = _os.path.join(tmp, 'kivy-latex-widget-image.png')
        #        pdf.rasterize(temp_filename, scale=10)  # returns a raster obj
        #        with self.canvas:
        #            self.img = Image(source=temp_filename)
        #    self.size = self.img.width, self.img.height

    def on_size(self, *args):
        #self.canvas.before.clear()
        #print("Latex onsize ", id(self), self.pos, self.size)
        scalew = self.size[0] / self.svg_size[0]  # so scale * svg_width == desired_width
        scaleh = self.size[1] / self.svg_size[1]  # so scale * svg_width == desired_width
        if scalew <= scaleh:  # scale so width of SVG takes up full width; center in y
            self.scale = max(scalew, 0.01)  # don't allow scale == 0 (causes error)
            self.svg_offset.x = 0
            self.svg_offset.y = (self.size[1] / self.scale - self.svg_size[1]) / 2.0
        else:  # scale so height of SVG takes up full height; center in x
            self.scale = max(scaleh, 0.01)  # don't allow scale == 0 (causes error)
            self.svg_offset.x = (self.size[0] / self.scale - self.svg_size[0]) / 2.0
            self.svg_offset.y = 0
        #print("  -> Latex scale = ",self.scale)

        self.bgrect.pos = (0, 0)  # self.pos
        #self.bgrect.size = self.svg_size  # self.size
        self.bgrect.size = (self.size[0]/self.scale, self.size[1]/self.scale)  # self.size
        #with self.canvas.before:
        #    Color(0.7, 0.7, 0.0)
        #    Rectangle(pos=self.pos, size=self.size)

    def simplify_svg_tree(self, svg_tree):
        """" Simplifies - mainly by resolving reference links within - a SVG file so that Kivy can digest it """
        definitions = {}

        def simplify_element(e, new_parent, definitions, in_defs):
            if e.tag.endswith('svg'):
                assert(new_parent is None), "<svg> element shouldn't have any parent (should be the root)!"
                if e.get('viewBox', None):
                    definitions['_viewbox'] = e.get('viewBox')  # perhaps for later use

                # Remove "pt" unit designations from width and height, as Kivy doesn't understand that
                # this sets the units for the entire file, and treats the rest of the file's number as
                # being in pixels -- so removing the "pt"s causes everything to be in pixels.
                attrib = e.attrib.copy()
                if 'width' in attrib and attrib['width'].endswith('pt'):
                    attrib['width'] = attrib['width'][:-2]
                if 'height' in attrib and attrib['height'].endswith('pt'):
                    attrib['height'] = attrib['height'][:-2]
                new_el = _ET.Element(e.tag, attrib)
                process_children = True
            elif in_defs and e.tag.endswith('symbol'):
                if e.get('id', None) is not None:
                    new_el = _ET.Element(e.tag, e.attrib)  # root a new "symbol tree" w/out parent
                    definitions[e.get('id')] = new_el
                    process_children = True
                else:  # no id, so ignore
                    _warnings.warn("SVG definition without id!")
                    new_el = None
                    process_children = False
            elif e.tag.endswith('clipPath'):  # ignore clipPath (Kivy can't process it)
                new_el = None
                process_children = False
            elif e.tag.endswith('defs'):
                in_defs = True
                new_el = None  # Ignore defs tag, but still process children
                process_children = True
            elif e.tag.endswith('use'):
                href = e.get('href', None)
                if href is None:  # then try to look for a {url}href attribute:
                    for k, v in e.attrib.items():
                        if k.endswith('href'):
                            href = v; break
                if href.startswith('#') and href[1:] in definitions:
                    href = href[1:]
                    new_el = _ET.SubElement(new_parent, 'g',
                                            {'transform': 'translate(%s,%s)' % (e.get('x', '0'), e.get('y', '0'))})
                    use_root = definitions[href]
                    for child_el in use_root:
                        simplify_element(child_el, new_el, definitions, False)
                else:
                    _warnings.warn("SVG id=%s not found in defintions" % str(href))
                    new_el = None  # id not found or not in defintions
                process_children = False
            # REMOvE: no need to perform any special processing here
            #elif e.tag.endswith('g'):
            #    if e.get('clip-path', None):
            #        new_el = new_parent  # ignore g elements with clip-path
            #    else:  # process normally
            #        new_el = 'copy'
            #    process_children = True
            else:
                new_el = 'copy'
                process_children = True

            if new_el == 'copy':
                new_el = _ET.Element(e.tag, e.attrib) if (new_parent is None) \
                    else _ET.SubElement(new_parent, e.tag, e.attrib)

            # DEBUG HACK - create red bounding box for debugging -- REMOVE LATER
            if e.tag.endswith('svg'):
                if e.get('viewBox', None):
                    x, y, w, h = e.get('viewBox').split()
                    #print("SVG bouding box dimensions: ",w, h)
                    _ET.SubElement(new_el, 'path',
                                   {'stroke': 'red', 'fill': 'none',
                                    'd': "M {x0} {y0} L {x1} {y0} L {x1} {y1} L {x0} {y1} Z".format(
                                        x0=x, y0=y, x1=str(float(x) + float(w)), y1=str(float(y) + float(h)))})
                
            if process_children:
                for child_el in e:
                    simplify_element(child_el, new_el, definitions, in_defs)

            return new_el

        root = svg_tree._root  # SVG root element
        new_root = simplify_element(root, None, definitions, False)
        # Note: could use definitions['_viewbox'] here if needed
        return _ET.ElementTree(new_root)


class WrappedLabel(Label):
    def __init__(self, *args, **kwargs):
        kwargs['size_hint_y'] = None
        super().__init__(*args, **kwargs)

    def on_size(self, *args):
        self.text_size = self.width, None
        self.height = self.texture_size[1]


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
    widget = TableWidget(headings, rows, spec, **spec['kivywidget_kwargs'])
    return {'kivywidget': widget}


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
    if isinstance(data, Widget):
        print("Other widget (%s) => %s" % (str(type(data)), str(data.size)))
        return CellWidgetContainer(data)  # assume widget size is already set
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
        widget = CellWidgetContainer(LatexWidget(data)) #, size_hint=(None, None), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        widget.content.size = (w, h)
        print("Latex widget => ", widget.content.size, ':', widget.content.latex_string)
        return widget
    else:
        PADDING = 10
        #widget = WrappedLabel(text=str(data))  # use label arg?
        widget = Label(text=str(data))  # use label arg?
        widget.texture_update()
        widget.size = (widget.texture_size[0] + PADDING, widget.texture_size[1] + PADDING)
        print("WrappedLabel widget => ", widget.size, ':', widget.text)
        return CellWidgetContainer(widget)


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
