#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Functions for generating report tables in different formats """

import latex as _lu
import html as _hu
import ppt as _pu
import cgi as _cgi
import numpy as _np

##############################################################################
#Formatting functions
##############################################################################

#create a 'rho' (state prep) symbol of given index
def _fmtRho_html(j):  return '&rho;<sub>%d</sub>' % j
def _fmtRho_latex(j): return '$\\rho_{%d}$' % j
def _fmtRho_py(j): return 'rho_%d' % j
def _fmtRho_ppt(j): return 'rho_%d' % j
Rho = { 'html': _fmtRho_html, 'latex': _fmtRho_latex, 'py': _fmtRho_py, 'ppt': _fmtRho_ppt }

#create an 'E' (POVM) symbol of given index
def _fmtE_html(j):  return 'E<sub>%d</sub>' % j
def _fmtE_latex(j): return '$E_{%d}$' % j
def _fmtE_py(j): return 'E_%d' % j
def _fmtE_ppt(j): return 'E_%d' % j
E = { 'html': _fmtE_html, 'latex': _fmtE_latex, 'py': _fmtE_py, 'ppt': _fmtE_ppt }

# 'normal' formatting
def _fmtNml_html(x):  return _hu.html(x)
def _fmtNml_latex(x): return _lu.latex(x)
def _fmtNml_py(x): return x
def _fmtNml_ppt(x): return _pu.ppt(x)
Nml = { 'html': _fmtNml_html, 'latex': _fmtNml_latex, 'py': _fmtNml_py, 'ppt': _fmtNml_ppt }

# 'normal' formatting but round to 2 decimal places
def _fmtNml2_html(x):  return _hu.html_value(x,ROUND=2)
def _fmtNml2_latex(x): return _lu.latex_value(x,ROUND=2)
def _fmtNml2_py(x): return x
def _fmtNml2_ppt(x): return _pu.ppt_value(x,ROUND=2)
Nml2 = { 'html': _fmtNml2_html, 'latex': _fmtNml2_latex, 'py': _fmtNml2_py, 'ppt': _fmtNml2_ppt }

# 'small' formating - make text smaller
def _fmtSml_html(x):  return _hu.html(x)
def _fmtSml_latex(x): return "\\small" + _lu.latex(x)
def _fmtSml_py(x): return x
def _fmtSml_ppt(x): return _pu.ppt(x)
Sml = { 'html': _fmtSml_html, 'latex': _fmtSml_latex, 'py': _fmtSml_py, 'ppt': _fmtSml_ppt }

# 'pi' formatting: add pi symbol/text after given quantity
def _fmtPi_html(x):
    if x == "" or x == "--": return x
    else: return _hu.html(x) + "&pi;"
def _fmtPi_latex(x):
    if x == "" or x == "--": return x
    else: return _lu.latex(x) + "$\\pi$"
def _fmtPi_py(x):
    if x == "" or x == "--": return ""
    else: 
        try: return x * _np.pi #but sometimes can't take product b/c x isn't a number
        except: return None
def _fmtPi_ppt(x):
    if x == "" or x == "--": return ""
    else: return _pu.ppt(x) + "pi"
Pi = { 'html': _fmtPi_html, 'latex': _fmtPi_latex, 'py': _fmtPi_py, 'ppt': _fmtPi_ppt }

# 'bracket' formatting: add brackets around given quantity
def _fmtBrk_html(x):  return _hu.html(x, brackets=True)
def _fmtBrk_latex(x): return _lu.latex(x, brackets=True)
def _fmtBrk_py(x): return x
def _fmtBrk_ppt(x): return _pu.ppt(x, brackets=True)
Brk = { 'html': _fmtBrk_html, 'latex': _fmtBrk_latex, 'py': _fmtBrk_py, 'ppt': _fmtBrk_ppt }

# 'conversion' formatting: catch all for find/replacing specially formatted text
def _fmtCnv_html(x):
    x = x.replace("|"," ") #remove pipes=>newlines, since html wraps table text automatically
    x = x.replace("<STAR>","REPLACEWITHSTARCODE") #b/c cgi.escape would mangle <STAR> marker
    x = _cgi.escape(x).encode("ascii","xmlcharrefreplace")
    x = x.replace("REPLACEWITHSTARCODE","&#9733;") #replace new marker with HTML code
    return x
def _fmtCnv_latex(x): 
    x = x.replace('%','\\%')
    x = x.replace('#','\\#')
    x = x.replace("half-width", "$\\nicefrac{1}{2}$-width")
    x = x.replace("1/2", "$\\nicefrac{1}{2}$")
    x = x.replace("Diamond","$\\Diamond$")
    x = x.replace("Check","\\checkmark")
    if "<STAR>" in x: #assume <STAR> never has $ around it already
        x = "$" + x.replace("<STAR>","\\bigstar") + "$"
    if "|" in x:
        return '\\begin{tabular}{c}' + '\\\\'.join(x.split("|")) + '\\end{tabular}'
    else:
        return x
def _fmtCnv_py(x):
    x = x.replace("<STAR>","*")
    x = x.replace("|"," ")
    return x
def _fmtCnv_ppt(x):
    x = x.replace("<STAR>","*")
    x = x.replace("|","\n")
    return x
TxtCnv = { 'html': _fmtCnv_html, 'latex': _fmtCnv_latex, 'py': _fmtCnv_py, 'ppt': _fmtCnv_ppt }

# 'errorbars' formatting: display a scalar value +/- error bar
def _fmtEB_html(t): 
    if t[1] is not None: 
        return "%s +/- %s" % (_hu.html(t[0]), _hu.html(t[1]))
    else: return _hu.html(t[0])
def _fmtEB_latex(t): 
    if t[1] is not None: 
        return "$ \\begin{array}{c} %s \\\\ \pm %s \\end{array} $" % (_lu.latex_value(t[0]), _lu.latex_value(t[1]))
    else: return _lu.latex_value(t[0])
def _fmtEB_py(t): 
    return { 'value': t[0], 'errbar': t[1] }
def _fmtEB_ppt(t): 
    if t[1] is not None: 
        return "%s +/- %s" % (_pu.ppt(t[0]), _pu.ppt(t[1]))
    else: return _pu.ppt(t[0])
EB = { 'html': _fmtEB_html, 'latex': _fmtEB_latex, 'py': _fmtEB_py, 'ppt': _fmtEB_ppt }


# 'vector errorbars' formatting: display a vector value +/- error bar
def _fmtEBvec_html(t): 
    if t[1] is not None: 
        return "%s +/- %s" % (_hu.html(t[0]), _hu.html(t[1]))
    else: return _hu.html(t[0])
def _fmtEBvec_latex(t): 
    if t[1] is not None: 
        return "%s $\pm$ %s" % (_lu.latex(t[0]), _lu.latex(t[1]))
    else: return _lu.latex(t[0])
def _fmtEBvec_py(t): return { 'value': t[0], 'errbar': t[1] }
def _fmtEBvec_ppt(t): 
    if t[1] is not None: 
        return "%s +/- %s" % (_pu.ppt(t[0]), _pu.ppt(t[1]))
    else: return _pu.ppt(t[0])
EBvec = { 'html': _fmtEBvec_html, 'latex': _fmtEBvec_latex, 'py': _fmtEBvec_py, 'ppt': _fmtEBvec_ppt }


# 'errorbars with pi' formatting: display (scalar_value +/- error bar) * pi
def _fmtEBPi_html(t): 
    if t[1] is not None: 
        return "(%s +/- %s)&pi;" % (_hu.html(t[0]), _hu.html(t[1]))
    else: return _fmtPi_html(t[0])
def _fmtEBPi_latex(t): 
    if t[1] is not None: 
        return "$ \\begin{array}{c}(%s \\\\ \pm %s)\\pi \\end{array} $" % (_lu.latex(t[0]), _lu.latex(t[1]))
    else: return _fmtPi_latex(t[0])
def _fmtEBPi_py(t): return { 'value': t[0], 'errbar': t[1] }
def _fmtEBPi_ppt(t): 
    if t[1] is not None: 
        return "(%s +/- %s)pi" % (_pu.ppt(t[0]), _pu.ppt(t[1]))
    else: return _pu.ppt(t[0])
EBPi = { 'html': _fmtEBPi_html, 'latex': _fmtEBPi_latex, 'py': _fmtEBPi_py, 'ppt': _fmtEBPi_ppt }


# 'gatestring' formatting: display a gate string
def _fmtGStr_html(s): 
    return '.'.join(s) if s is not None else ""
def _fmtGStr_latex(s):
    if s is None: 
        return ""
    else:
        boxed = [ ("\\mbox{%s}" % gl) for gl in s ]
        return "$" + '\\cdot'.join(boxed) + "$"
def _fmtGStr_py(s): 
    return tuple(s) if s is not None else None
def _fmtGStr_ppt(s): 
    return '.'.join(s) if s is not None else ""
GStr = { 'html': _fmtGStr_html, 'latex': _fmtGStr_latex, 'py': _fmtGStr_py, 'ppt': _fmtGStr_ppt }


def _formatList(items, formatters, fmt):
    assert(len(items) == len(formatters))
    formatted_items = []
    for i,item in enumerate(items):
        if formatters[i] is not None:
            formatted_items.append( formatters[i][fmt](item) )
        else:
            formatted_items.append( item )
    return formatted_items


def create_table(formats, tables, colHeadings, formatters, tableclass, longtable, customHeader=None):
    """ Create a new table for each specified format in the tables dictionary """

    if "latex" in formats:

        table = "longtable" if longtable else "tabular"
        if customHeader is not None and "latex" in customHeader:
            latex = customHeader['latex']
        else:
            colHeadings_formatted = _formatList(colHeadings, formatters, "latex")
            latex  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
            latex += "%s \\\\ \hline\n" % (" & ".join(colHeadings_formatted))
        
        if "latex" not in tables: tables['latex'] = ""
        tables['latex'] += latex


    if "html" in formats:

        if customHeader is not None and "html" in customHeader:
            html = customHeader['html']
        else:
            colHeadings_formatted = _formatList(colHeadings, formatters, "html")
            html  = "<table class=%s><thead>" % tableclass
            html += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings_formatted))
            html += "</thead><tbody>"
        
        if "html" not in tables: tables['html'] = ""
        tables['html'] += html


    if "py" in formats:

        if customHeader is not None and "py" in customHeader:
            raise ValueError("custom headers not supported for python format")
        colHeadings_formatted = _formatList(colHeadings, formatters, "py")

        if "py" not in tables: tables['py'] = []
        tableDict = { 'column names': colHeadings_formatted }
        tables['py'].append( tableDict )

    if "ppt" in formats:

        if customHeader is not None and "ppt" in customHeader:
            raise ValueError("custom headers not supported for powerpoint format")
        colHeadings_formatted = _formatList(colHeadings, formatters, "ppt")

        if "ppt" not in tables: tables['ppt'] = []
        tableDict = { 'column names': colHeadings_formatted }
        tables['ppt'].append( tableDict )



def create_table_preformatted(formats, tables, colHeadings, tableclass, longtable):
    """ Create a new table for each specified format in the tables dictionary
        colHeadings is assumed to be a dictionary with pre-formatted column
        heading appropriate for each format
    """

    if "latex" in formats:

        table = "longtable" if longtable else "tabular"
        latex  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings['latex']) + "|")
        latex += "%s \\\\ \hline\n" % (" & ".join(colHeadings['latex']))
        
        if "latex" not in tables: tables['latex'] = ""
        tables['latex'] += latex

    if "html" in formats:
        
        html  = "<table class=%s><thead>" % tableclass
        html += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings['html']))
        html += "</thead><tbody>"
        
        if "html" not in tables: tables['html'] = ""
        tables['html'] += html

    if "py" in formats:

        if "py" not in tables: tables['py'] = []
        tableDict = { 'column names': colHeadings['py'] }
        tables['py'].append( tableDict )

    if "ppt" in formats:

        if "ppt" not in tables: tables['ppt'] = []
        tableDict = { 'column names': colHeadings['ppt'] }
        tables['ppt'].append( tableDict )


def add_table_row(formats, tables, rowData, formatters):
    """ Add a row to each table in tables dictionary """

    if "latex" in formats:
        assert("latex" in tables)
        formatted_rowData = _formatList(rowData, formatters, "latex")
        if len(formatted_rowData) > 0:
            tables['latex'] += " & ".join(formatted_rowData) + " \\\\ \hline\n"

    if "html" in formats:
        assert("html" in tables)
        formatted_rowData = _formatList(rowData, formatters, "html")
        if len(formatted_rowData) > 0:
            tables['html'] += "<tr><td>" + "</td><td>".join(formatted_rowData) + "</td></tr>\n"


    if "py" in formats:
        assert("py" in tables)
        formatted_rowData = _formatList(rowData, formatters, "py")
        if len(formatted_rowData) > 0:
            curTableDict = tables["py"][-1] #last table dict is "current" one
            if "row data" not in curTableDict: curTableDict['row data'] = []
            curTableDict['row data'].append( formatted_rowData )

    if "ppt" in formats:
        assert("ppt" in tables)
        formatted_rowData = _formatList(rowData, formatters, "ppt")
        if len(formatted_rowData) > 0:
            curTableDict = tables["ppt"][-1] #last table dict is "current" one
            if "row data" not in curTableDict: curTableDict['row data'] = []
            curTableDict['row data'].append( formatted_rowData )



def finish_table(formats, tables, longtable):
    """ Finish (end) each table in tables dictionary """

    if "latex" in formats:
        assert("latex" in tables)
        table = "longtable" if longtable else "tabular"
        tables['latex'] += "\end{%s}\n" % table        

    if "html" in formats:
        assert("html" in tables)
        tables['html'] += "</tbody></table>"

    if "py" in formats:
        assert("py" in tables)
        #pass #nothing to do to mark table dict as finished

    if "ppt" in formats:
        assert("ppt" in tables)
        #pass
#        curTableDict = tables["ppt"][-1] #last table dict is "current" one
#        tables["ppt"][-1] = _pu.PPTTable(curTableDict) # convert dict to a ppt table object for later rendering


#def add_inter_table_space(formats, tables):
#    """ Add some space (if appropriate) to each table in tables dictionary.
#        Should only be used after calling finish_table """
#
#    if "latex" in formats:
#        assert("latex" in tables)
#        tables['latex'] += "\n\n\\vspace{2em}\n\n"
#
#    if "html" in formats:
#        assert("html" in tables)
#        tables['html'] += "<br/>"
#
#    if "py" in formats:
#        assert("py" in tables)
#        pass #adding space N/A for python format
#
#    if "ppt" in formats:
#        assert("ppt" in tables)
#        pass #adding space N/A for powerpoint format

