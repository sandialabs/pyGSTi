""" Functions for writing GST objects to text files."""
import json as _json
import stdinput as _stdinput
from .. import tools as _tools
from .. import objects as _objs

def writeParameterFile(filename, params):
    """ 
    Write a json-formatted parameter file.

    Parameters
    ----------
    filename : string
        The name of the file to write.

    params: dict
        The parameters to save.
    """

    return _json.dump( params, open(filename, "wb"), indent=4) # object_pairs_hook=_collections.OrderedDict


def writeEmptyDatasetFile(filename, gateStringList, 
                          headerString='## Columns = plus frequency, count total', numZeroCols=None,
                          appendWeightsColumn=False):
    """
    Write an empty dataset file to be used as a template.

    Parameters
    ----------
    filename : string
        The filename to write.

    gateStringList : list of GateStrings
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

    if len(gateStringList) > 0 and not isinstance(gateStringList[0], _objs.GateString):
        raise ValueError("Argument gateStringList must be a list of GateString objects!")

    if numZeroCols is None: #TODO: cleaner way to extract number of columns from headerString?
        if headerString.startswith('## Columns = '):
            numZeroCols = len(headerString.split(',')) 
        else:
            raise ValueError("Must specify numZeroCols since I can't figure it out from the header string")

    f = open(filename, 'w')
    zeroCols = "  ".join( ['0']*numZeroCols )
    print >> f, headerString
    for gateString in gateStringList: #gateString should be a GateString object here
        print >> f, gateString.str + "  " + zeroCols + (("  %f" % gateString.weight) if appendWeightsColumn else "")
    f.close()


def writeDatasetFile(filename, gateStringList, dataset, spamLabelOrder=None):
    """
    Write a text-formatted dataset file.

    Parameters
    ----------
    filename : string
        The filename to write.

    gateStringList : list of GateStrings
        The list of gate strings to include in the written dataset.

    dataset : DataSet
        The data set from which counts are obtained.

    spamLabelOrder : list, optional
        A list of the SPAM labels in dataset which specifies
        the column order in the output file.
    """
    if len(gateStringList) > 0 and not isinstance(gateStringList[0], _objs.GateString):
        raise ValueError("Argument gateStringList must be a list of GateString objects!")

    spamLabels = dataset.getSpamLabels()
    if spamLabelOrder is not None:
        assert(len(spamLabelOrder) == len(spamLabels))
        assert(all( [sl in spamLabels for sl in spamLabelOrder] ))
        assert(all( [sl in spamLabelOrder for sl in spamLabels] ))
        spamLabels = spamLabelOrder

    headerString = '## Columns = ' + ", ".join( [ "%s count" % sl for sl in spamLabels ]) 
    parser = _stdinput.StdInputParser()

    f = open(filename, 'w')
    print >> f, headerString
    for gateString in gateStringList: #gateString should be a GateString object here
        dataRow = dataset[gateString.tup]
        print >> f, gateString.str + "  " + "  ".join( [("%g" % dataRow[sl]) for sl in spamLabels] )
    f.close()


def writeGatestringList(filename, gateStringList, header=None):
    """
    Write a text-formatted gate string list file.

    Parameters
    ----------
    filename : string
        The filename to write.

    gateStringList : list of GateStrings
        The list of gate strings to include in the written dataset.

    header : string, optional
        Header line (first line of file).  Prepended with a pound sign (#), so no
        need to include one.

    """
    if len(gateStringList) > 0 and not isinstance(gateStringList[0], _obs.GateString):
        raise ValueError("Argument gateStringList must be a list of GateString objects!")
    
    f = open(filename, "w")

    if header is not None:
        print >> f, "# %s" % header

    for gateString in gateStringList:
        print >> f, gateString.str

    f.close()


def writeGateset(gs,filename,title=None):
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
    f = open(filename, "w")

    if title is not None:
        print >> f, "# %s" % title
    print >> f, ""

    for (i,rhoVec) in enumerate(gs.rhoVecs):
        print >> f, "rho%s" % (str(i) if len(gs.rhoVecs) > 1 else "")
        print >> f, "PauliVec"
        print >> f, " ".join( "%.8g" % el for el in rhoVec )
        print >> f, ""

    for (i,EVec) in enumerate(gs.EVecs):
        print >> f, "E%s" % (str(i) if len(gs.EVecs) > 1 else "")
        print >> f, "PauliVec"
        print >> f, " ".join( "%.8g" % el for el in EVec )
        print >> f, ""

    for (label,gate) in gs.iteritems():
        print >> f, label
        print >> f, "PauliMx"
        print >> f, _tools.mxToString(gate, width=16, prec=8)
        print >> f, ""

    if gs.identityVec is not None:
        print >> f, "IDENTITYVEC " + " ".join( "%.8g" % el for el in gs.identityVec )
    else:
        print >> f, "IDENTITYVEC None"

    for sl in gs.SPAM_labels:
        (iR,iE) = gs.SPAM_labels[sl]
        if iE == -1:
            if iR == -1:
                print >> f, "SPAMLABEL %s = remainder" % sl
            else:
                print >> f, "SPAMLABEL %s = %s remainder" % (sl, "rho%s" % (str(iR) if len(gs.rhoVecs) > 1 else ""))
        else:
            assert(iR >= 0)  #iR can only be -1 when iE == -1
            print >> f, "SPAMLABEL %s = %s %s" % (sl, 
                                                  "rho%s" % (str(iR) if len(gs.rhoVecs) > 1 else ""),
                                                  "E%s" % (str(iE) if len(gs.EVecs) > 1 else "") )
    f.close()
