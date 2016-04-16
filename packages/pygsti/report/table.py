import tableformat as _tf

class ReportTable(object):
    def __init__(self, formats, colHeadings, formatters, tableclass,
                 longtable, customHeader=None):
        self.formats = formats
        self.tableclass = tableclass
        self.longtable = longtable
        self._finished = False
        self._raw_tables = {} 
        
        if "latex" in formats:
    
            table = "longtable" if longtable else "tabular"
            if customHeader is not None and "latex" in customHeader:
                latex = customHeader['latex']
            else:
                if formatters is not None:
                    colHeadings_formatted = \
                        _tf.formatList(colHeadings, formatters, "latex")
                else: #formatters is None => colHeadings is dict w/formats
                    colHeadings_formatted = colHeadings['latex']
                
                latex  = "\\begin{%s}[l]{%s}\n\hline\n" % \
                    (table, "|c" * len(colHeadings_formatted) + "|")
                latex += "%s \\\\ \hline\n" % \
                    (" & ".join(colHeadings_formatted))
                
            self._raw_tables['latex'] = latex

    
        if "html" in formats:
    
            if customHeader is not None and "html" in customHeader:
                html = customHeader['html']
            else:
                if formatters is not None:
                    colHeadings_formatted = \
                        _tf.formatList(colHeadings, formatters, "html")
                else: #formatters is None => colHeadings is dict w/formats
                    colHeadings_formatted = colHeadings['html']

                html  = "<table class=%s><thead>" % tableclass
                html += "<tr><th> %s </th></tr>" % \
                    (" </th><th> ".join(colHeadings_formatted))
                html += "</thead><tbody>"
            
            self._raw_tables['html'] = html
    
    
        if "py" in formats:
    
            if customHeader is not None and "py" in customHeader:
                raise ValueError("custom headers unsupported for python format")

            if formatters is not None:
                colHeadings_formatted = \
                    _tf.formatList(colHeadings, formatters, "py")
            else: #formatters is None => colHeadings is dict w/formats
                colHeadings_formatted = colHeadings['py']
    
            self._raw_tables['py'] = \
                { 'column names': colHeadings_formatted }
    
        if "ppt" in formats:
    
            if customHeader is not None and "ppt" in customHeader:
                raise ValueError("custom headers unsupported for " +
                                 "powerpoint format")

            if formatters is not None:
                colHeadings_formatted = \
                    _tf.formatList(colHeadings, formatters, "ppt")
            else: #formatters is None => colHeadings is dict w/formats
                colHeadings_formatted = colHeadings['ppt']
    
            self._raw_tables['ppt'] = \
                { 'column names': colHeadings_formatted }

        

    def addrow(self, rowData, formatters):
        if self._finished:
            raise ValueError("Cannot add rows to a ReportTable after finish()")

        if "latex" in self.formats:
            assert("latex" in self._raw_tables)
            formatted_rowData = _tf.formatList(rowData, formatters, "latex")
            if len(formatted_rowData) > 0:
                self._raw_tables['latex'] += " & ".join(formatted_rowData) \
                    + " \\\\ \hline\n"
    
        if "html" in self.formats:
            assert("html" in self._raw_tables)
            formatted_rowData = _tf.formatList(rowData, formatters, "html")
            if len(formatted_rowData) > 0:
                self._raw_tables['html'] += "<tr><td>" + \
                    "</td><td>".join(formatted_rowData) + "</td></tr>\n"
    
    
        if "py" in self.formats:
            assert("py" in self._raw_tables)
            formatted_rowData = _tf.formatList(rowData, formatters, "py")
            if len(formatted_rowData) > 0:
                if "row data" not in self._raw_tables["py"]:
                    self._raw_tables["py"]['row data'] = []
                self._raw_tables["py"]['row data'].append( formatted_rowData )
    
        if "ppt" in self.formats:
            assert("ppt" in self._raw_tables)
            formatted_rowData = _tf.formatList(rowData, formatters, "ppt")
            if len(formatted_rowData) > 0:
                if "row data" not in self._raw_tables["ppt"]:
                    self._raw_tables["ppt"]['row data'] = []
                self._raw_tables["ppt"]['row data'].append( formatted_rowData )

    def finish(self):
        """ Finish (end) this table."""
    
        if "latex" in self.formats:
            assert("latex" in self._raw_tables)
            table = "longtable" if self.longtable else "tabular"
            self._raw_tables['latex'] += "\end{%s}\n" % table        
    
        if "html" in self.formats:
            assert("html" in self._raw_tables)
            self._raw_tables['html'] += "</tbody></table>"
    
        if "py" in self.formats:
            assert("py" in self._raw_tables)
            #nothing to do to mark table dict as finished
    
        if "ppt" in self.formats:
            assert("ppt" in self._raw_tables)
            #nothing to do to mark table dict as finished

        self._finished = True #mark table as finished

    def _checkpy(self):
        if "py" not in self.formats:
            raise KeyError("Must include 'py' format to access" + 
                           "specific data within this object.")

    def __getitem__(self, key):
        """Indexes the first column rowdata"""
        self._checkpy()
        tblDict = self._raw_tables['py']
        for row_data in tblDict['row data']:
            if len(row_data) > 0 and row_data[0] == key:
                return { key:val for key,val in \
                             zip(tblDict['column names'],row_data) }
        raise KeyError("%s not found as a first-column value" % key)

    def __len__(self):
        self._checkpy()
        return len(self._raw_tables['py']['row data'])

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        """ 
        Return a list of the first element of each row, which can be
        used for indexing.
        """
        self._checkpy()
        tblDict = self._raw_tables['py']
        return [ d[0] for d in tblDict['row data'] if len(d) > 0 ]

    def has_key(self, key):
        return key in self.keys()

    def row(self, key=None, index=None):
        self._checkpy()
        tblDict = self._raw_tables['py']

        if key is not None:
            if index is not None:
                raise ValueError("Cannot specify *both* key and index")
            for row_data in tblDict['row data']:
                if len(row_data) > 0 and row_data[0] == key:
                    return row_data
            raise KeyError("%s not found as a first-column value" % key)
        
        elif index is not None:
            if 0 <= index < len(tblDict['row data']):
                return tblDict['row data'][index]
            else:
                raise ValueError("Index %d is out of bounds" % index)

        else:
            raise ValueError("Must specify either key or index")


    def col(self, key=None, index=None):
        self._checkpy()
        tblDict = self._raw_tables['py']

        if key is not None:
            if index is not None:
                raise ValueError("Cannot specify *both* key and index")
            if key in tblDict['column names']:
                iCol = tblDict['column names'].index(key)
                return [ d[iCol] for d in tblDict['row data'] ] #if len(d)>iCol
            raise KeyError("%s is not a column name." % key)
        
        elif index is not None:
            if 0 <= index < len(tblDict['column names']):
                return [ d[index] for d in tblDict['row data'] ] #if len(d)>iCol
            else:
                raise ValueError("Index %d is out of bounds" % index)

        else:
            raise ValueError("Must specify either key or index")


    @property
    def num_rows(self):
        self._checkpy()
        return len(self._raw_tables['py']['row data'])
    
    @property
    def num_cols(self):
        self._checkpy()
        return len(self._raw_tables['py']['column names'])

    @property
    def row_names(self):
        return self.keys()

    @property
    def col_names(self):
        self._checkpy()
        return self._raw_tables['py']['column names'][:]

    def render(self, fmt):
        if fmt in self._raw_tables:
            return self._raw_tables[fmt]
        raise ValueError("Unrecognized format %s.  Valid formats are %s" \
                             % (fmt, self.formats) )

        
    





#OLD: SCRATCH -- TODO: REMOVE

#def create_table(formats, tables, colHeadings, formatters, tableclass, longtable, customHeader=None):
#    """ Create a new table for each specified format in the tables dictionary """
#
#    if "latex" in formats:
#
#        table = "longtable" if longtable else "tabular"
#        if customHeader is not None and "latex" in customHeader:
#            latex = customHeader['latex']
#        else:
#            colHeadings_formatted = _formatList(colHeadings, formatters, "latex")
#            latex  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings) + "|")
#            latex += "%s \\\\ \hline\n" % (" & ".join(colHeadings_formatted))
#        
#        if "latex" not in tables: tables['latex'] = ""
#        tables['latex'] += latex
#
#
#    if "html" in formats:
#
#        if customHeader is not None and "html" in customHeader:
#            html = customHeader['html']
#        else:
#            colHeadings_formatted = _formatList(colHeadings, formatters, "html")
#            html  = "<table class=%s><thead>" % tableclass
#            html += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings_formatted))
#            html += "</thead><tbody>"
#        
#        if "html" not in tables: tables['html'] = ""
#        tables['html'] += html
#
#
#    if "py" in formats:
#
#        if customHeader is not None and "py" in customHeader:
#            raise ValueError("custom headers not supported for python format")
#        colHeadings_formatted = _formatList(colHeadings, formatters, "py")
#
#        if "py" not in tables: tables['py'] = []
#        tableDict = { 'column names': colHeadings_formatted }
#        tables['py'].append( tableDict )
#
#    if "ppt" in formats:
#
#        if customHeader is not None and "ppt" in customHeader:
#            raise ValueError("custom headers not supported for powerpoint format")
#        colHeadings_formatted = _formatList(colHeadings, formatters, "ppt")
#
#        if "ppt" not in tables: tables['ppt'] = []
#        tableDict = { 'column names': colHeadings_formatted }
#        tables['ppt'].append( tableDict )
#
#
#
#def create_table_preformatted(formats, tables, colHeadings, tableclass, longtable):
#    """ Create a new table for each specified format in the tables dictionary
#        colHeadings is assumed to be a dictionary with pre-formatted column
#        heading appropriate for each format
#    """
#
#    if "latex" in formats:
#
#        table = "longtable" if longtable else "tabular"
#        latex  = "\\begin{%s}[l]{%s}\n\hline\n" % (table, "|c" * len(colHeadings['latex']) + "|")
#        latex += "%s \\\\ \hline\n" % (" & ".join(colHeadings['latex']))
#        
#        if "latex" not in tables: tables['latex'] = ""
#        tables['latex'] += latex
#
#    if "html" in formats:
#        
#        html  = "<table class=%s><thead>" % tableclass
#        html += "<tr><th> %s </th></tr>" % (" </th><th> ".join(colHeadings['html']))
#        html += "</thead><tbody>"
#        
#        if "html" not in tables: tables['html'] = ""
#        tables['html'] += html
#
#    if "py" in formats:
#
#        if "py" not in tables: tables['py'] = []
#        tableDict = { 'column names': colHeadings['py'] }
#        tables['py'].append( tableDict )
#
#    if "ppt" in formats:
#
#        if "ppt" not in tables: tables['ppt'] = []
#        tableDict = { 'column names': colHeadings['ppt'] }
#        tables['ppt'].append( tableDict )
#
#
#def add_table_row(formats, tables, rowData, formatters):
#    """ Add a row to each table in tables dictionary """
#
#    if "latex" in formats:
#        assert("latex" in tables)
#        formatted_rowData = _formatList(rowData, formatters, "latex")
#        if len(formatted_rowData) > 0:
#            tables['latex'] += " & ".join(formatted_rowData) + " \\\\ \hline\n"
#
#    if "html" in formats:
#        assert("html" in tables)
#        formatted_rowData = _formatList(rowData, formatters, "html")
#        if len(formatted_rowData) > 0:
#            tables['html'] += "<tr><td>" + "</td><td>".join(formatted_rowData) + "</td></tr>\n"
#
#
#    if "py" in formats:
#        assert("py" in tables)
#        formatted_rowData = _formatList(rowData, formatters, "py")
#        if len(formatted_rowData) > 0:
#            curTableDict = tables["py"][-1] #last table dict is "current" one
#            if "row data" not in curTableDict: curTableDict['row data'] = []
#            curTableDict['row data'].append( formatted_rowData )
#
#    if "ppt" in formats:
#        assert("ppt" in tables)
#        formatted_rowData = _formatList(rowData, formatters, "ppt")
#        if len(formatted_rowData) > 0:
#            curTableDict = tables["ppt"][-1] #last table dict is "current" one
#            if "row data" not in curTableDict: curTableDict['row data'] = []
#            curTableDict['row data'].append( formatted_rowData )
#
#
#
#def finish_table(formats, tables, longtable):
#    """ Finish (end) each table in tables dictionary """
#
#    if "latex" in formats:
#        assert("latex" in tables)
#        table = "longtable" if longtable else "tabular"
#        tables['latex'] += "\end{%s}\n" % table        
#
#    if "html" in formats:
#        assert("html" in tables)
#        tables['html'] += "</tbody></table>"
#
#    if "py" in formats:
#        assert("py" in tables)
#        #pass #nothing to do to mark table dict as finished
#
#    if "ppt" in formats:
#        assert("ppt" in tables)
#        #pass


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
