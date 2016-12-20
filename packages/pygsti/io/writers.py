from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for writing GST objects to text files."""

import json as _json
# from . import stdinput as _stdinput
from .. import tools as _tools
from .. import objects as _objs

def write_parameter_file(filename, params):
    """
    Write a json-formatted parameter file.

    Parameters
    ----------
    filename : string
        The name of the file to write.

    params: dict
        The parameters to save.
    """
    with open(filename, 'w') as output:
        return _json.dump(params, output, indent=4)
    #return _json.dump( params, open(filename, "wb"), indent=4) # object_pairs_hook=_collections.OrderedDict


def write_empty_dataset(filename, gatestring_list,
                        headerString='## Columns = plus frequency, count total', numZeroCols=None,
                        appendWeightsColumn=False):
    """
    Write an empty dataset file to be used as a template.

    Parameters
    ----------
    filename : string
        The filename to write.

    gatestring_list : list of GateStrings
        List of gate strings to write, each to be followed by numZeroCols zeros.

    headerString : string, optional
        Header string for the file; should start with a pound (#) or double-pound (##)
        so it is treated as a commend or directive, respectively.

    numZeroCols : int, optional
        The number of zero columns to place after each gate string.  If None,
        then headerString must begin with "## Columns = " and number of zero
        columns will be inferred.

    appendWeightsColumn : bool, optional
        Add an additional 'weights' column.

    """

    if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
        raise ValueError("Argument gatestring_list must be a list of GateString objects!")

    if numZeroCols is None: #TODO: cleaner way to extract number of columns from headerString?
        if headerString.startswith('## Columns = '):
            numZeroCols = len(headerString.split(','))
        else:
            raise ValueError("Must specify numZeroCols since I can't figure it out from the header string")

    with open(filename, 'w') as output:
        zeroCols = "  ".join( ['0']*numZeroCols )
        output.write(headerString + '\n')
        for gateString in gatestring_list: #gateString should be a GateString object here
            output.write(gateString.str + "  " + zeroCols + (("  %f" % gateString.weight) if appendWeightsColumn else "") + '\n')


def write_dataset(filename, dataset, gatestring_list=None, spamLabelOrder=None):
    """
    Write a text-formatted dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    dataset : DataSet
        The data set from which counts are obtained.

    gatestring_list : list of GateStrings, optional
        The list of gate strings to include in the written dataset.
        If None, all gate strings are output.

    spamLabelOrder : list, optional
        A list of the SPAM labels in dataset which specifies
        the column order in the output file.
    """
    if gatestring_list is not None:
        if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
            raise ValueError("Argument gatestring_list must be a list of GateString objects!")
    else:
        gatestring_list = list(dataset.keys())

    spamLabels = dataset.get_spam_labels()
    if spamLabelOrder is not None:
        assert(len(spamLabelOrder) == len(spamLabels))
        assert(all( [sl in spamLabels for sl in spamLabelOrder] ))
        assert(all( [sl in spamLabelOrder for sl in spamLabels] ))
        spamLabels = spamLabelOrder

    headerString = '## Columns = ' + ", ".join( [ "%s count" % sl for sl in spamLabels ])
    # parser = _stdinput.StdInputParser()

    with open(filename, 'w') as output:
        output.write(headerString + '\n')
        for gateString in gatestring_list: #gateString should be a GateString object here
            dataRow = dataset[gateString.tup]
            output.write(gateString.str + "  " + "  ".join( [("%g" % dataRow[sl]) for sl in spamLabels] ) + '\n')

def write_multidataset(filename, multidataset, gatestring_list=None, spamLabelOrder=None):
    """
    Write a text-formatted multi-dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    multidataset : MultiDataSet
        The multi data set from which counts are obtained.

    gatestring_list : list of GateStrings
        The list of gate strings to include in the written dataset.
        If None, all gate strings are output.

    spamLabelOrder : list, optional
        A list of the SPAM labels in multidataset which specifies
        the column order in the output file.
    """

    if gatestring_list is not None:
        if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
            raise ValueError("Argument gatestring_list must be a list of GateString objects!")
    else:
        gatestring_list = list(multidataset.gsIndex.keys()) #TODO: make access function for gatestrings?

    spamLabels = multidataset.get_spam_labels()
    if spamLabelOrder is not None:
        assert(len(spamLabelOrder) == len(spamLabels))
        assert(all( [sl in spamLabels for sl in spamLabelOrder] ))
        assert(all( [sl in spamLabelOrder for sl in spamLabels] ))
        spamLabels = spamLabelOrder

    dsLabels = list(multidataset.keys())

    headerString = '## Columns = ' + ", ".join( [ "%s %s count" % (dsl,sl)
                                                  for dsl in dsLabels
                                                  for sl in spamLabels ])
    # parser = _stdinput.StdInputParser()

    with open(filename, 'w') as output:
        output.write(headerString + '\n')
        for gateString in gatestring_list: #gateString should be a GateString object here
            gs = gateString.tup #gatestring tuple
            output.write(gateString.str + "  " + "  ".join( [("%g" % multidataset[dsl][gs][sl])
                                                            for dsl in dsLabels for sl in spamLabels] ) + '\n')

def write_gatestring_list(filename, gatestring_list, header=None):
    """
    Write a text-formatted gate string list file.

    Parameters
    ----------
    filename : string
        The filename to write.

    gatestring_list : list of GateStrings
        The list of gate strings to include in the written dataset.

    header : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    """
    if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
        raise ValueError("Argument gatestring_list must be a list of GateString objects!")

    with open(filename, 'w') as output:
        if header is not None:
            output.write("# %s" % header + '\n')

        for gateString in gatestring_list:
            output.write(gateString.str + '\n')


def write_gateset(gs,filename,title=None):
    """
    Write a text-formatted gate set file.

    Parameters
    ----------
    gs : GateSet
        The gate set to write to file.

    filename : string
        The filename to write.

    title : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    """

    with open(filename, 'w') as output:

        if title is not None:
            output.write("# %s" % title + '\n')
        output.write('\n')

        for (prepLabel,rhoVec) in gs.preps.items():
            output.write("%s\n" % prepLabel)
            output.write("PauliVec\n")
            output.write(" ".join( "%.8g" % el for el in rhoVec ) + '\n')
            output.write("\n")

        for (ELabel,EVec) in gs.effects.items():
            output.write("%s\n" % ELabel)
            output.write("PauliVec\n")
            output.write(" ".join( "%.8g" % el for el in EVec ) + '\n')
            output.write("\n")

        for (label,gate) in gs.gates.items():
            output.write(label + '\n')
            output.write("PauliMx\n")
            output.write(_tools.mx_to_string(gate, width=16, prec=8) + '\n')
            output.write("\n")

        if gs.povm_identity is not None:
            output.write("IDENTITYVEC " + " ".join( "%.8g" % el for el in gs.povm_identity ) + '\n')
        else:
            output.write("IDENTITYVEC None\n")

        for sl,(prepLabel,ELabel) in gs.spamdefs.items():
            output.write("SPAMLABEL %s = %s %s\n" % (sl, prepLabel, ELabel))

        dims = gs.get_basis_dimension()
        if dims is None:
            output.write("BASIS %s\n" % gs.get_basis_name())
        else:
            if type(dims) != int:
                dimStr = ",".join(map(str,dims))
            else: dimStr = str(dims)
            output.write("BASIS %s %s\n" % (gs.get_basis_name(), dimStr))
