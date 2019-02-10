""" Functions for writing GST objects to text files."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import warnings as _warnings
import numpy as _np

# from . import stdinput as _stdinput
from .. import tools as _tools
from .. import objects as _objs

def write_empty_dataset(filename, circuit_list,
                        headerString='## Columns = 1 frequency, count total', numZeroCols=None,
                        appendWeightsColumn=False):
    """
    Write an empty dataset file to be used as a template.

    Parameters
    ----------
    filename : string
        The filename to write.

    circuit_list : list of Circuits
        List of operation sequences to write, each to be followed by numZeroCols zeros.

    headerString : string, optional
        Header string for the file; should start with a pound (#) or double-pound (##)
        so it is treated as a commend or directive, respectively.

    numZeroCols : int, optional
        The number of zero columns to place after each operation sequence.  If None,
        then headerString must begin with "## Columns = " and number of zero
        columns will be inferred.

    appendWeightsColumn : bool, optional
        Add an additional 'weights' column.

    """

    if len(circuit_list) > 0 and not isinstance(circuit_list[0], _objs.Circuit):
        raise ValueError("Argument circuit_list must be a list of Circuit objects!")

    if numZeroCols is None: #TODO: cleaner way to extract number of columns from headerString?
        if headerString.startswith('## Columns = '):
            numZeroCols = len(headerString.split(','))
        else:
            raise ValueError("Must specify numZeroCols since I can't figure it out from the header string")

    with open(filename, 'w') as output:
        zeroCols = "  ".join( ['0']*numZeroCols )
        output.write(headerString + '\n')
        for circuit in circuit_list: #circuit should be a Circuit object here
            output.write(circuit.str + "  " + zeroCols + (("  %f" % circuit.weight) if appendWeightsColumn else "") + '\n')

            
def _outcome_to_str(x):
    if _tools.isstr(x): return x
    else: return ":".join([str(i) for i in x])

    
def write_dataset(filename, dataset, circuit_list=None,
                  outcomeLabelOrder=None, fixedColumnMode=True):
    """
    Write a text-formatted dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    dataset : DataSet
        The data set from which counts are obtained.

    circuit_list : list of Circuits, optional
        The list of operation sequences to include in the written dataset.
        If None, all operation sequences are output.

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
    if circuit_list is not None:
        if len(circuit_list) > 0 and not isinstance(circuit_list[0], _objs.Circuit):
            raise ValueError("Argument circuit_list must be a list of Circuit objects!")
    else:
        circuit_list = list(dataset.keys())

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
        for circuit in circuit_list: #circuit should be a Circuit object here
            dataRow = dataset[circuit.tup]
            counts = dataRow.counts

            if fixedColumnMode:
                #output '--' for outcome labels that aren't present in this row
                output.write(circuit.str + "  " +
                             "  ".join( [(("%g" % counts[ol]) if (ol in counts) else '--')
                                         for ol in outcomeLabels] ))
            else: # use expanded label:count format
                output.write(
                    circuit.str + "  " +
                    "  ".join( [("%s:%g" % (_outcome_to_str(ol),counts[ol]))
                                for ol in outcomeLabels if ol in counts] ))

            #write aux info
            if dataRow.aux:
                output.write(" # %s" % str(repr(dataRow.aux)))
            output.write('\n') # finish the line
                

def write_multidataset(filename, multidataset, circuit_list=None, outcomeLabelOrder=None):
    """
    Write a text-formatted multi-dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    multidataset : MultiDataSet
        The multi data set from which counts are obtained.

    circuit_list : list of Circuits
        The list of operation sequences to include in the written dataset.
        If None, all operation sequences are output.

    outcomeLabelOrder : list, optional
        A list of the SPAM labels in multidataset which specifies
        the column order in the output file.
    """

    if circuit_list is not None:
        if len(circuit_list) > 0 and not isinstance(circuit_list[0], _objs.Circuit):
            raise ValueError("Argument circuit_list must be a list of Circuit objects!")
    else:
        circuit_list = list(multidataset.cirIndex.keys()) #TODO: make access function for circuits?

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
        for circuit in circuit_list: #circuit should be a Circuit object here
            opstr = circuit.tup #circuit tuple
            cnts = [multidataset[dsl][opstr].counts.get(ol,'--') for dsl in dsLabels for ol in outcomeLabels]
            output.write(circuit.str + "  " + "  ".join( [ (("%g" % cnt) if (cnt != '--') else cnt)
                                                              for cnt in cnts] ) + '\n')

def write_circuit_list(filename, circuit_list, header=None):
    """
    Write a text-formatted operation sequence list file.

    Parameters
    ----------
    filename : string
        The filename to write.

    circuit_list : list of Circuits
        The list of operation sequences to include in the written dataset.

    header : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    """
    if len(circuit_list) > 0 and not isinstance(circuit_list[0], _objs.Circuit):
        raise ValueError("Argument circuit_list must be a list of Circuit objects!")

    with open(filename, 'w') as output:
        if header is not None:
            output.write("# %s" % header + '\n')

        for circuit in circuit_list:
            output.write(circuit.str + '\n')


def write_model(mdl,filename,title=None):
    """
    Write a text-formatted model file.

    Parameters
    ----------
    mdl : Model
        The model to write to file.

    filename : string
        The filename to write.

    title : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    """

    def writeprop(f, lbl, val):
        """ Write (label,val) property to output file """
        if isinstance(val, _np.ndarray): # then write as rows
            f.write("%s\n" % lbl)
            if val.ndim == 1:
                f.write(" ".join( "%.8g" % el for el in val ) + '\n')
            elif val.ndim == 2:
                f.write(_tools.mx_to_string(val, width=16, prec=8))
            else:
                raise ValueError("Cannot write an ndarray with %d dimensions!" % val.ndim)
            f.write("\n")
        else:
            f.write("%s = %s\n" % (lbl, repr(val)))
        

    with open(filename, 'w') as output:

        if title is not None:
            output.write("# %s" % title + '\n')
        output.write('\n')

        for prepLabel,rhoVec in mdl.preps.items():
            props = None
            if isinstance(rhoVec, _objs.FullSPAMVec): typ = "PREP"
            elif isinstance(rhoVec, _objs.TPSPAMVec): typ = "TP-PREP"
            elif isinstance(rhoVec, _objs.StaticSPAMVec): typ = "STATIC-PREP"
            elif isinstance(rhoVec, _objs.LindbladSPAMVec):
                typ = "CPTP-PREP"
                props = [ ("PureVec", rhoVec.state_vec.todense()),
                          ("ErrgenMx", rhoVec.error_map.todense()) ]
            else:
                _warnings.warn(
                    ("Non-standard prep of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "fully parameterized spam vector").format(typ=str(type(rhoVec))))
                typ = "PREP"

            if props is None: props = [("LiouvilleVec", rhoVec.todense())]
            output.write("%s: %s\n" % (typ,prepLabel))
            for lbl,val in props:
                writeprop(output, lbl, val)

        for povmLabel,povm in mdl.povms.items():
            props = None; povm_to_write = povm
            if isinstance(povm, _objs.UnconstrainedPOVM): povmType = "POVM"
            elif isinstance(povm, _objs.TPPOVM): povmType = "TP-POVM"
            elif isinstance(povm, _objs.LindbladPOVM):
                povmType = "CPTP-POVM"
                props = [ ("ErrgenMx", povm.error_map.todense()) ]
                povm_to_write = povm.base_povm
            else:
                _warnings.warn(
                    ("Non-standard POVM of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "standard POVM").format(typ=str(type(povm))))
                povmType = "POVM"
                
            output.write("%s: %s\n\n" % (povmType,povmLabel))
            if props is not None:
                for lbl,val in props:
                    writeprop(output, lbl, val)

            for ELabel,EVec in povm_to_write.items():
                if isinstance(EVec, _objs.FullSPAMVec): typ = "EFFECT"
                elif isinstance(EVec, _objs.ComplementSPAMVec): typ = "EFFECT" # ok
                elif isinstance(EVec, _objs.TPSPAMVec): typ = "TP-EFFECT"
                elif isinstance(EVec, _objs.StaticSPAMVec): typ = "STATIC-EFFECT"
                else:
                    _warnings.warn(
                        ("Non-standard effect of type {typ} cannot be described by"
                         "text format model files.  It will be read in as a"
                         "fully parameterized spam vector").format(typ=str(type(EVec))))
                    typ = "EFFECT"
                output.write("%s: %s\n" % (typ,ELabel))
                writeprop(output, "LiouvilleVec", EVec.todense())
                
            output.write("END POVM\n\n")

        for label,gate in mdl.operations.items():
            props = None
            if isinstance(gate, _objs.FullDenseOp): typ = "GATE"
            elif isinstance(gate, _objs.TPDenseOp): typ = "TP-GATE"
            elif isinstance(gate, _objs.StaticDenseOp): typ = "STATIC-GATE"
            elif isinstance(gate, _objs.LindbladDenseOp):
                typ = "CPTP-GATE"
                props = [ ("LiouvilleMx", gate.todense()) ]
                if gate.unitary_postfactor is not None:
                    upost = gate.unitary_postfactor.todense() \
                            if isinstance(gate.unitary_postfactor,_objs.LinearOperator) \
                            else gate.unitary_postfactor
                    props.append( ("RefLiouvilleMx", upost) )
            else:
                _warnings.warn(
                    ("Non-standard gate of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "fully parameterized gate").format(typ=str(type(gate))))
                typ = "GATE"

            if props is None: props = [("LiouvilleMx", gate.todense())]
            output.write(typ + ": " + str(label) + '\n')
            for lbl,val in props:
                writeprop(output, lbl, val)


        for instLabel,inst in mdl.instruments.items():
            if isinstance(inst, _objs.Instrument): typ = "Instrument" 
            elif isinstance(inst, _objs.TPInstrument): typ = "TP-Instrument"
            else:
                _warnings.warn(
                    ("Non-standard Instrument of type {typ} cannot be described by"
                     "text format model files.  It will be read in as a"
                     "standard Instrument").format(typ=str(type(inst))))
                typ = "Instrument"
            output.write(typ + ": " + str(instLabel) + '\n\n')

            for label,gate in inst.items():
                if isinstance(gate, _objs.FullDenseOp): typ = "IGATE"
                elif isinstance(gate, _objs.TPInstrumentOp): typ = "IGATE" # ok b/c instrument itself is marked as TP
                elif isinstance(gate, _objs.StaticDenseOp): typ = "STATIC-IGATE"
                else:
                    _warnings.warn(
                        ("Non-standard gate of type {typ} cannot be described by"
                         "text format model files.  It will be read in as a"
                         "fully parameterized gate").format(typ=str(type(gate))))
                    typ = "IGATE"
                output.write(typ + ": " + str(label) + '\n')
                writeprop(output, "LiouvilleMx", gate.todense())
            output.write("END Instrument\n\n")

        if mdl.state_space_labels is not None:
            output.write("STATESPACE: " + str(mdl.state_space_labels) + "\n")
              # StateSpaceLabels.__str__ formats the output properly

        basisdim = mdl.basis.dim
        
        if basisdim is None:
            output.write("BASIS: %s\n" % mdl.basis.name)
        else:
            if mdl.basis.name not in ('std','pp','gm','qt'): # a "fancy" basis
                assert(mdl.state_space_labels is not None), \
                    "Must set a Model's state space labels when using fancy a basis!"
                output.write("BASIS: %s\n" % mdl.basis.name) # don't write the dim - the state space labels will cover this.
            else:
                output.write("BASIS: %s %d\n" % (mdl.basis.name, basisdim))

        if isinstance(mdl.default_gauge_group, _objs.FullGaugeGroup):
            output.write("GAUGEGROUP: Full\n")
        elif isinstance(mdl.default_gauge_group, _objs.TPGaugeGroup):
            output.write("GAUGEGROUP: TP\n")
        elif isinstance(mdl.default_gauge_group, _objs.UnitaryGaugeGroup):
            output.write("GAUGEGROUP: Unitary\n")
