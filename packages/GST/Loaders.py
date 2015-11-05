""" Functions for loading GST objects from text files."""
import os as _os
import json as _json

import outputdata as _outputdata
import StdInputParser as _StdInputParser
import dataset as _dataset
import multidataset as _multidataset

def loadParameterFile(filename):
    """ 
    Load a json-formatted parameter file.

    Parameters
    ----------
    filename : string
        The name of the file to load.

    Returns
    -------
    dict
        The json file converted to a python dictionary.
    """
    return _json.load( open(filename, "rb") )

def loadDataset(filename, dskey=None, cache=False):
    """
    Load a DataSet from a file.  If dskey is not None, attempts to load file as 
    a pickled OutputData object.  Otherwise, first tries to load file as a 
    saved DataSet object, then as a standard text-formatted DataSet.
  
    Parameters
    ----------
    filename : string
        The name of the file
  
    dskey : string, optional
        The name of the key for the dataset to load, if the
        file names a pickled OutputData file.

    cache : bool, optional
        When set to True, a pickle file with the name filename + ".cache"
        is searched for and loaded instead of filename if it exists
        and is newer than filename.  If no cache file exists or one
        exists but it is older than filename, a cache file will be
        written after loading from filename.
  
    Returns
    -------
    DataSet
    """

    if dskey is not None:
        gst_data = _outputdata.OutputData(filename)
        if not gst_data.has_dataset(dskey):
            raise ValueError("Dataset '%s' was not found in target file: %s" % (dskey,filename))
        return gst_data.get_dataset(dskey)
    else:
        try: 
            # a saved Dataset object is ok
            ds = _dataset.DataSet(fileToLoadFrom=filename)
        except:

            if cache:
                bReadCache = False
                cache_filename = filename + ".cache"
                if _os.path.exists( cache_filename ) and \
                   _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                    try:
                        print "Loading from cache file: ",cache_filename
                        ds = _dataset.DataSet(fileToLoadFrom=cache_filename)
                        return ds
                    except:
                        print "Failed to load from cache file"
                else:
                    print "Cache file not found or is tool old -- one will be created after loading is completed"

                # otherwise must use standard dataset file format
                parser = _StdInputParser.StdInputParser()
                ds = parser.parse_datafile(filename)

                print "Writing cache file (to speed future loads): ",cache_filename
                ds.save(cache_filename)
            else:
                # otherwise must use standard dataset file format
                parser = _StdInputParser.StdInputParser()
                ds = parser.parse_datafile(filename)
        return ds


def loadMultiDataset(filename, cache=False):
    """
    Load a MultiDataSet from a file.  First tries to load file as a 
    saved MultiDataSet object, then as a standard text-formatted MultiDataSet.
  
    Parameters
    ----------
    filename : string
        The name of the file
  
    cache : bool, optional
        When set to True, a pickle file with the name filename + ".cache"
        is searched for and loaded instead of filename if it exists
        and is newer than filename.  If no cache file exists or one
        exists but it is older than filename, a cache file will be
        written after loading from filename.
  
    Returns
    -------
    MultiDataSet
    """

    try: 
        # a saved MultiDataset object is ok
        mds = _multidataset.MultiDataSet(fileToLoadFrom=filename)
    except:
        if cache:
            bReadCache = False
            cache_filename = filename + ".cache"
            if _os.path.exists( cache_filename ) and \
               _os.path.getmtime(filename) < _os.path.getmtime(cache_filename):
                try: 
                    print "Loading from cache file: ",cache_filename
                    mds = _multidataset.MultiDataSet(fileToLoadFrom=cache_filename)
                    return mds
                except:
                    print "Failed to load from cache file"
            else:
                print "Cache file not found or is tool old -- one will be created after loading is completed"

            # otherwise must use standard dataset file format
            parser = _StdInputParser.StdInputParser()
            mds = parser.parse_multidatafile(filename)

            print "Writing cache file (to speed future loads): ",cache_filename
            mds.save(cache_filename)

        else:
            # otherwise must use standard dataset file format
            parser = _StdInputParser.StdInputParser()
            mds = parser.parse_multidatafile(filename)
    return mds



def loadGateset(filename):
    """
    Load a GateSet from a file, formatted using the 
    standard text-format for gate sets.

    Parameters
    ----------
    filename : string
        The name of the file

    Returns
    -------
    GateSet
    """
    return _StdInputParser.readGateset(filename)

def loadGatestringDict(filename):
    """
    Load a gate string dictionary from a file, formatted 
    using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file.

    Returns
    -------
    Dictionary with keys = gate string labels and
      values = GateString objects.
    """
    std = _StdInputParser.StdInputParser()
    return std.parse_dictfile(filename)

def loadGatestringList(filename, readRawStrings=False):
    """
    Load a gate string list from a file, formatted 
    using the standard text-format.

    Parameters
    ----------
    filename : string
        The name of the file

    readRawStrings : boolean
        If True, gate strings are not converted
        to tuples of gate labels.

    Returns
    -------
    list of GateString objects
    """
    if readRawStrings:
        rawList = []
        for line in open(filename,'r'):
            if len(line.strip()) == 0: continue
            if len(line) == 0 or line[0] == '#': continue
            rawList.append( line.strip() )
        return rawList
    else:
        std = _StdInputParser.StdInputParser()
        return std.parse_stringfile(filename)

