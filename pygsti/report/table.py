"""
Defines the ReportTable class
"""

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict as _OrderedDict
from .row import Row
from .convert import convert_dict as _convert_dict


class ReportTable(object):
    """
    Table representation, renderable in multiple formats

    Parameters
    ----------
    col_headings : list or dict
        Default column headings if list; dictionary of overrides if dict.

    formatters : list
        Names of default column heading formatters.

    custom_header : dict
        Dictionary of overriden headers.

    col_heading_labels : list
        Labels for column headings (tooltips).

    confidence_region_info : ConfidenceRegion, optional
        If not None, specifies a confidence-region used to display error
        intervals.  Specifically, tells reportableqtys if they should use non
        markovian error bars or not

    Attributes
    ----------
    num_rows : int
        The number of rows in this table

    num_cols : int
        The number of columns in this table

    row_names : list
        The names (keys) of the rows in this table

    col_names : list
        The names (keys) of the columns in this table
    """

    def __init__(self, col_headings, formatters, custom_header=None,
                 col_heading_labels=None, confidence_region_info=None):
        '''
        Create a table object

        Parameters
        ----------
        col_headings : list or dict
            default column headings if list,
            dictionary of overrides if dict
        formatters : list
            names of default column heading formatters
        custom_header : dict
            dictionary of overriden headers
        col_heading_labels : list
            labels for column headings (tooltips)
        confidence_region_info : ConfidenceRegion, optional
            If not None, specifies a confidence-region
            used to display error intervals.
            Specifically, tells reportableqtys if
            they should use non markovian error bars or not
        '''
        self.nonMarkovianEBs = bool(confidence_region_info is not None
                                    and confidence_region_info.nonMarkRadiusSq > 0)
        self._customHeadings = custom_header
        self._rows = []
        self._override = isinstance(col_headings, dict)

        if self._override:
            self._columnNames = col_headings['python']
        else:
            self._columnNames = col_headings

        if col_heading_labels is None:
            col_heading_labels = self._columnNames

        if self._override:
            # Dictionary of overridden formats
            self._headings = {k: Row(v, labels=col_heading_labels, non_markovian_ebs=self.nonMarkovianEBs)
                              for k, v in col_headings.items()}
        else:
            self._headings = Row(col_headings, formatters, col_heading_labels, self.nonMarkovianEBs)

    def add_row(self, data, formatters=None, labels=None, non_markovian_ebs=None):
        """
        Adds a row to the table.

        Parameters
        ----------
        data : list
            A list of the data for each cell of the added row.

        formatters : list[string], optional
            Formatting options for each cell of the added row.

        labels : list[string], optional
            Labeling options for each cell of the added row.

        non_markovian_ebs : bool
            Whether non-Markovian error bars should be used

        Returns
        -------
        None
        """
        if non_markovian_ebs is None:
            non_markovian_ebs = self.nonMarkovianEBs
        self._rows.append(Row(data, formatters, labels, non_markovian_ebs))

    def finish(self):
        """
        Finish table creation.  Indicates no more rows will be added.

        Returns
        -------
        None
        """
        pass  # nothing to do currently

    def _get_col_headings(self, fmt, spec):
        if self._override:
            # _headings is a dictionary of overridden formats
            return self._headings[fmt].render(fmt, spec)
        else:
            # _headings is a row object
            return self._headings.render(fmt, spec)

    def render(self, fmt, longtables=False, table_id=None, tableclass=None,
               output_dir=None, precision=6, polarprecision=3, sciprecision=0,
               resizable=False, autosize=False, fontsize=None, complex_as_polar=True,
               brackets=False, click_to_display=False, link_to=None, render_includes=True):
        """
        Render a table object

        Parameters
        ----------
        fmt : string
            name of format to be used

        longtables : bool
            latex table option

        table_id : string
            id tag for HTML tables

        tableclass : string
            class tag for HTML tables

        output_dir : string
            directory for latex figures to be rendered in

        precision : int
            number of digits to render

        polarprecision : int
            number of digits to render for polars

        sciprecision : int
            number of digits to render for scientific notation

        resizable : bool
            allow a table to be resized

        autosize : bool
            allow a table to be automatically sized

        fontsize : int
            override fontsize of a tabel

        complex_as_polar : bool
            render complex numbers as polars

        brackets : bool
            render matrix like types w/ brackets

        click_to_display : bool
            table plots must be clicked to prompt creation

        link_to : list or {'tex', 'pdf', 'pkl'}
            whether to create links to TEX, PDF, and/or PKL files

        render_includes : bool
            whether files included in rendered latex should also be rendered
            (usually as PDFs for use with the 'includegraphics' latex statement)

        Returns
        -------
        string
        """
        spec = {
            'output_dir': output_dir,
            'precision': precision,
            'polarprecision': polarprecision,
            'sciprecision': sciprecision,
            'resizable': resizable,
            'autosize': autosize,
            'click_to_display': click_to_display,
            'fontsize': fontsize,
            'complex_as_polar': complex_as_polar,
            'brackets': brackets,
            'longtables': longtables,
            'table_id': table_id,
            'tableclass': tableclass,
            'link_to': link_to,
            'render_includes': render_includes}

        if fmt not in _convert_dict:
            raise NotImplementedError('%s format option is not currently supported' % fmt)

        table = _convert_dict[fmt]['table']  # Function for rendering a table in the format "fmt"
        rows = [row.render(fmt, spec) for row in self._rows]

        colHeadingsFormatted = self._get_col_headings(fmt, spec)
        return table(self._customHeadings, colHeadingsFormatted, rows, spec)

    def __str__(self):

        def _strlen(x):
            return max([len(p) for p in str(x).split('\n')])

        def _nlines(x):
            return len(str(x).split('\n'))

        def _getline(x, i):
            lines = str(x).split('\n')
            return lines[i] if i < len(lines) else ""

        #self.render('text')
        col_widths = [0] * len(self._columnNames)
        row_lines = [0] * len(self._rows)
        header_lines = 0

        for i, nm in enumerate(self._columnNames):
            col_widths[i] = max(_strlen(nm), col_widths[i])
            header_lines = max(header_lines, _nlines(nm))
        for k, row in enumerate(self._rows):
            for i, el in enumerate(row.cells):
                el = el.data.value()
                col_widths[i] = max(_strlen(el), col_widths[i])
                row_lines[k] = max(row_lines[k], _nlines(el))

        row_separator = "|" + '-' * (sum([w + 5 for w in col_widths]) - 1) + "|\n"
        # +5 for pipe & spaces, -1 b/c don't count first pipe

        s = "*** ReportTable object ***\n"
        s += row_separator

        for k in range(header_lines):
            for i, nm in enumerate(self._columnNames):
                s += "|  %*s  " % (col_widths[i], _getline(nm, k))
            s += "|\n"
        s += row_separator

        for rowIndex, row in enumerate(self._rows):
            for k in range(row_lines[rowIndex]):
                for i, el in enumerate(row.cells):
                    el = el.data.value()
                    s += "|  %*s  " % (col_widths[i], _getline(el, k))
                s += "|\n"
            s += row_separator

        s += "\n"
        s += "Access row and column data by indexing into this object\n"
        s += " as a dictionary using the column header followed by the\n"
        s += " value of the first element of each row, i.e.,\n"
        s += " tableObj[<column header>][<first row element>].\n"

        return s

    def __getitem__(self, key):
        """Indexes the first column rowdata"""
        for row in self._rows:
            row_data = row.cells
            if len(row_data) > 0 and row_data[0].data.value() == key:
                return _OrderedDict(zip(self._columnNames, row_data))
        raise KeyError("%s not found as a first-column value" % key)

    def __len__(self):
        return len(self._rows)

    def __contains__(self, key):
        return key in list(self.keys())

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__.update(d)

    def keys(self):
        """
        A list of the first element of each row, which can be used for indexing.

        Returns
        -------
        list
        """
        return [row.cells[0].data.value() for row in self._rows if len(row.cells) > 0]

    def row(self, key=None, index=None):
        """
        Retrieve a row's cell data.

        A row is identified by either its `key` (its first cell's value) OR its `index` (0-based).
        You cannot specify both `key` and `index`.

        Parameters
        ----------
        key : object, optional
            Value of a row's first cell.

        index : int, optional
            Row index.

        Returns
        -------
        list
        """
        if key is not None:
            if index is not None:
                raise ValueError("Cannot specify *both* key and index")
            for row in self._rows:
                row_data = row.cells
                if len(row_data) > 0 and row_data[0].data.value() == key:
                    return row_data
            raise KeyError("%s not found as a first-column value" % key)

        elif index is not None:
            if 0 <= index < len(self):
                return self._rows[index].cells
            else:
                raise ValueError("Index %d is out of bounds" % index)

        else:
            raise ValueError("Must specify either key or index")

    def col(self, key=None, index=None):
        """
        Retrieve a column's cell data.

        A column is identified by either its `key` (its column header's value) OR its `index` (0-based).
        You cannot specify both `key` and `index`.

        Parameters
        ----------
        key : object, optional
            Value of a column's header value.

        index : int, optional
            Column index.

        Returns
        -------
        list
        """
        if key is not None:
            if index is not None:
                raise ValueError("Cannot specify *both* key and index")
            if key in self._columnNames:
                iCol = self._columnNames.index(key)
                return [row.cells[iCol] for row in self._rows]  # if len(d)>iCol
            raise KeyError("%s is not a column name." % key)

        elif index is not None:
            if 0 <= index < len(self._columnNames):
                return [row.cells[index] for row in self._rows]  # if len(d)>iCol
            else:
                raise ValueError("Index %d is out of bounds" % index)

        else:
            raise ValueError("Must specify either key or index")

    @property
    def num_rows(self):
        """
        The number of rows in this table

        Returns
        -------
        int
        """
        return len(self._rows)

    @property
    def num_cols(self):
        """
        The number of columns in this table

        Returns
        -------
        int
        """
        return len(self._columnNames)

    @property
    def row_names(self):
        """
        The names (keys) of the rows in this table

        Returns
        -------
        list
        """
        return list(self.keys())

    @property
    def col_names(self):
        """
        The names (keys) of the columns in this table

        Returns
        -------
        list
        """
        return self._columnNames
