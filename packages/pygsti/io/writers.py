""" Functions for writing GST objects to text files."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import warnings as _warnings

# from . import stdinput as _stdinput
from .. import tools as _tools
from .. import objects as _objs

def write_empty_dataset(filename, gatestring_list,
                        headerString='## Columns = 1 frequency, count total', numZeroCols=None,
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

            
def _outcome_to_str(x):
    if _tools.isstr(x): return x
    else: return ":".join([str(i) for i in x])

    
def write_dataset(filename, dataset, gatestring_list=None,
                  outcomeLabelOrder=None, fixedColumnMode=True):
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

    outcomeLabelOrder : list, optional
        A list of the outcome labels in dataset which specifies
        the column order in the output file.

    fixedColumnMode : bool, optional
        When `True`, a file is written with column headers indicating which
        outcome each column of counts corresponds to.  If a row doesn't have
        any counts for an outcome, `'--'` is used in its place.  When `False`,
        each row's counts are written in an expanded form that includes the
        outcome labels (each "count" has the format <outcomeLabel>:<count>).
    """
    if gatestring_list is not None:
        if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
            raise ValueError("Argument gatestring_list must be a list of GateString objects!")
    else:
        gatestring_list = list(dataset.keys())

    if outcomeLabelOrder is not None: #convert to tuples if needed
        outcomeLabelOrder = [ (ol,) if _tools.isstr(ol) else ol
                              for ol in outcomeLabelOrder ]
        
    outcomeLabels = dataset.get_outcome_labels()
    if outcomeLabelOrder is not None:
        assert(len(outcomeLabelOrder) == len(outcomeLabels))
        assert(all( [ol in outcomeLabels for ol in outcomeLabelOrder] ))
        assert(all( [ol in outcomeLabelOrder for ol in outcomeLabels] ))
        outcomeLabels = outcomeLabelOrder

    headerString = ""
    if hasattr(dataset,'comment') and dataset.comment is not None:
        for commentLine in dataset.comment.split('\n'):
            if commentLine.startswith('#'):
                headerString += commentLine + '\n'
            else:
                headerString += "# " + commentLine + '\n'

    if fixedColumnMode:
        headerString += '## Columns = ' + ", ".join( [ "%s count" % _outcome_to_str(ol)
                                                       for ol in outcomeLabels ]) + '\n'
    with open(filename, 'w') as output:
        output.write(headerString)
        for gateString in gatestring_list: #gateString should be a GateString object here
            dataRow = dataset[gateString.tup]
            counts = dataRow.counts

            if fixedColumnMode:
                #output '--' for outcome labels that aren't present in this row
                output.write(gateString.str + "  " +
                             "  ".join( [(("%g" % counts[ol]) if (ol in counts) else '--')
                                         for ol in outcomeLabels] ) + '\n')
            else: # use expanded label:count format
                output.write(
                    gateString.str + "  " +
                    "  ".join( [("%s:%g" % (_outcome_to_str(ol),counts[ol]))
                                for ol in outcomeLabels if ol in counts] )+'\n')
                

def write_multidataset(filename, multidataset, gatestring_list=None, outcomeLabelOrder=None):
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

    outcomeLabelOrder : list, optional
        A list of the SPAM labels in multidataset which specifies
        the column order in the output file.
    """

    if gatestring_list is not None:
        if len(gatestring_list) > 0 and not isinstance(gatestring_list[0], _objs.GateString):
            raise ValueError("Argument gatestring_list must be a list of GateString objects!")
    else:
        gatestring_list = list(multidataset.gsIndex.keys()) #TODO: make access function for gatestrings?

    if outcomeLabelOrder is not None: #convert to tuples if needed
        outcomeLabelOrder = [ (ol,) if _tools.isstr(ol) else ol
                              for ol in outcomeLabelOrder ]

    outcomeLabels = multidataset.get_outcome_labels()
    if outcomeLabelOrder is not None:
        assert(len(outcomeLabelOrder) == len(outcomeLabels))
        assert(all( [ol in outcomeLabels for ol in outcomeLabelOrder] ))
        assert(all( [ol in outcomeLabelOrder for ol in outcomeLabels] ))
        outcomeLabels = outcomeLabelOrder

    dsLabels = list(multidataset.keys())

    headerString = ""
    if hasattr(multidataset,'comment') and multidataset.comment is not None:
        for commentLine in multidataset.comment.split('\n'):
            if commentLine.startswith('#'):
                headerString += commentLine + '\n'
            else:
                headerString += "# " + commentLine + '\n'
    headerString += '## Columns = ' + ", ".join( [ "%s %s count" % (dsl,_outcome_to_str(ol))
                                                   for dsl in dsLabels
                                                   for ol in outcomeLabels ])
    # parser = _stdinput.StdInputParser()

    with open(filename, 'w') as output:
        output.write(headerString + '\n')
        for gateString in gatestring_list: #gateString should be a GateString object here
            gs = gateString.tup #gatestring tuple
            output.write(gateString.str + "  " + "  ".join( [("%g" % multidataset[dsl][gs].counts.get(ol,'--'))
                                                            for dsl in dsLabels for ol in outcomeLabels] ) + '\n')

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

        for prepLabel,rhoVec in gs.preps.items():
            if isinstance(rhoVec, _objs.FullyParameterizedSPAMVec): typ = "PREP"
            elif isinstance(rhoVec, _objs.TPParameterizedSPAMVec): typ = "TP-PREP"
            elif isinstance(rhoVec, _objs.StaticSPAMVec): typ = "STATIC-PREP"
            else:
                _warnings.warn(
                    ("Non-standard prep of type {typ} cannot be described by"
                     "text format gate set files.  It will be read in as a"
                     "fully parameterized spam vector").format(typ=str(type(rhoVec))))
                typ = "PREP"
            output.write("%s: %s\n" % (typ,prepLabel))
            output.write("LiouvilleVec\n")
            output.write(" ".join( "%.8g" % el for el in rhoVec ) + '\n')
            output.write("\n")

        for povmLabel,povm in gs.povms.items():
            if isinstance(povm, _objs.UnconstrainedPOVM): povmType = "POVM"
            elif isinstance(povm, _objs.TPPOVM): povmType = "TP-POVM"
            else:
                _warnings.warn(
                    ("Non-standard POVM of type {typ} cannot be described by"
                     "text format gate set files.  It will be read in as a"
                     "standard POVM").format(typ=str(type(povm))))
                typ = "POVM"
                
            output.write("%s: %s\n\n" % (povmType,povmLabel))
                
            for ELabel,EVec in povm.items():
                if isinstance(EVec, _objs.FullyParameterizedSPAMVec): typ = "EFFECT"
                elif isinstance(EVec, _objs.ComplementSPAMVec): typ = "EFFECT" # ok
                elif isinstance(EVec, _objs.TPParameterizedSPAMVec): typ = "TP-EFFECT"
                elif isinstance(EVec, _objs.StaticSPAMVec): typ = "STATIC-EFFECT"
                else:
                    _warnings.warn(
                        ("Non-standard effect of type {typ} cannot be described by"
                         "text format gate set files.  It will be read in as a"
                         "fully parameterized spam vector").format(typ=str(type(EVec))))
                    typ = "EFFECT"
                output.write("%s: %s\n" % (typ,ELabel))
                output.write("LiouvilleVec\n")
                output.write(" ".join( "%.8g" % el for el in EVec ) + '\n')
                output.write("\n")

            output.write("END POVM\n\n")

        for label,gate in gs.gates.items():
            if isinstance(gate, _objs.FullyParameterizedGate): typ = "GATE"
            elif isinstance(gate, _objs.TPParameterizedGate): typ = "TP-GATE"
            elif isinstance(gate, _objs.LindbladParameterizedGate): typ = "CPTP-GATE"
            elif isinstance(gate, _objs.StaticGate): typ = "STATIC-GATE"
            else:
                _warnings.warn(
                    ("Non-standard gate of type {typ} cannot be described by"
                     "text format gate set files.  It will be read in as a"
                     "fully parameterized gate").format(typ=str(type(gate))))
                typ = "GATE"
            output.write(typ + ": " + label + '\n')
            output.write("LiouvilleMx\n")
            output.write(_tools.mx_to_string(gate, width=16, prec=8) + '\n')
            output.write("\n")

        for instLabel,inst in gs.instruments.items():
            if isinstance(inst, _objs.Instrument): typ = "Instrument" 
            elif isinstance(inst, _objs.TPInstrument): typ = "TP-Instrument"
            else:
                _warnings.warn(
                    ("Non-standard Instrument of type {typ} cannot be described by"
                     "text format gate set files.  It will be read in as a"
                     "standard Instrument").format(typ=str(type(inst))))
                typ = "Instrument"
            output.write(typ + ": " + instLabel + '\n\n')

            for label,gate in inst.items():
                if isinstance(gate, _objs.FullyParameterizedGate): typ = "IGATE"
                elif isinstance(gate, _objs.TPInstrumentGate): typ = "IGATE" # ok b/c instrument itself is marked as TP
                elif isinstance(gate, _objs.StaticGate): typ = "STATIC-IGATE"
                else:
                    _warnings.warn(
                        ("Non-standard gate of type {typ} cannot be described by"
                         "text format gate set files.  It will be read in as a"
                         "fully parameterized gate").format(typ=str(type(gate))))
                    typ = "IGATE"
                output.write(typ + ": " + label + '\n')
                output.write("LiouvilleMx\n")
                output.write(_tools.mx_to_string(gate, width=16, prec=8) + '\n')
                output.write("\n")
            output.write("END Instrument\n\n")

        dims = gs.basis.dim.blockDims
        if dims is None:
            output.write("BASIS: %s\n" % gs.basis.name)
        else:
            if type(dims) != int:
                dimStr = ",".join(map(str,dims))
            else: dimStr = str(dims)
            output.write("BASIS: %s %s\n" % (gs.basis.name, dimStr))

        if isinstance(gs.default_gauge_group, _objs.FullGaugeGroup):
            output.write("GAUGEGROUP: Full\n")
        elif isinstance(gs.default_gauge_group, _objs.TPGaugeGroup):
            output.write("GAUGEGROUP: TP\n")
        elif isinstance(gs.default_gauge_group, _objs.UnitaryGaugeGroup):
            output.write("GAUGEGROUP: Unitary\n")
