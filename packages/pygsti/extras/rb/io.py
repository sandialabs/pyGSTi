from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
from . import results as _results

def import_rb_summary_data(filenameslist,verbosity=1):
    """
    #
    qubit number
    # RB length // Success counts // Number of repeats // Circuit depth // Circuit two-qubit gate count
    data
    """
    raw_lengths = []
    raw_scounts = []
    raw_repeats = []
    raw_cdepths = []
    raw_c2Qgc = []

    for filename in filenameslist:
        if verbosity > 0:
            print("Importing "+filename+"...",end='')
        try:
            with open(filename,'r') as f:
                counter = 0
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')
                    if line[0][0] != '#':
                        if len(line) > 1:
                            raw_lengths.append(int(line[0])) 
                            raw_scounts.append(int(line[1]))
                            raw_repeats.append(int(line[2]))
                            try:
                                raw_cdepths.append(int(line[3]))
                            except:
                                pass
                            try:
                                raw_c2Qgc.append(int(line[4]))
                            except:
                                pass
                        else:
                            n = int(line[0])
            if verbosity > 0:
                print("Complete.")
        except:
            raise ValueError("Date import failed! File does not exist or the format is incorrect.")

    # Find and order arrays of sequence lengths at each n
    lengths = []
    for l in raw_lengths:
        if l not in lengths:
              lengths.append(l)
    lengths.sort()

    # Take all the raw data and put it into lists for each sequence length
    scounts = []
    repeats = []
    cdepths = []
    c2Qgc = []
    for i in range(0,len(lengths)):
        scounts.append([])
        repeats.append([])
        cdepths.append([])
        c2Qgc.append([])

    for i in range(0,len(raw_lengths)):
        index = lengths.index(raw_lengths[i])
        scounts[index].append(raw_scounts[i])
        repeats[index].append(raw_repeats[i])
        cdepths[index].append(raw_cdepths[i])
        c2Qgc[index].append(raw_c2Qgc[i])
   
    RBSdataset = _results.RBSummaryDataset(n, lengths, scounts, repeats, cdepths, c2Qgc)

    return RBSdataset

#
# Todo : delete this legacy importer after my data is converted to the new format.
#
def legacy_import_txt_format_rb_data(filenameslist,verbosity=1):
    
    raw_lengths = []
    raw_scounts = []
    raw_cdepths = []
    raw_c2Qgc = []

    for filename in filenameslist:
        if verbosity > 0:
            print("Importing "+filename+"...",end='')
        with open(filename,'r') as f:
            counter = 0
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                try:
                    raw_lengths.append(int(line[0])) 
                    raw_scounts.append(float(line[1])) 
                    raw_cdepths.append(int(line[2])) 
                    raw_c2Qgc.append(int(line[3]))
                except:
                    counter += 1
                    if counter > 1:
                        if verbosity > 0:
                            print("File format incorrect!")
            if verbosity > 0:
                print("Complete.")
    #except:
    #    print("Failed! File does not exist!")

    # Find and order arrays of sequence lengths at each n
    lengths = []
    for l in raw_lengths:
        if l not in lengths:
              lengths.append(l)
    lengths.sort()

    # Take all the raw data and put it into lists for each sequence length
    scounts = []
    cdepths = []
    c2Qgc = []
    for i in range(0,len(lengths)):
        scounts.append([])
        cdepths.append([])
        c2Qgc.append([])

    for i in range(0,len(raw_lengths)):
        index = lengths.index(raw_lengths[i])
        scounts[index].append(raw_scounts[i])
        cdepths[index].append(raw_cdepths[i])
        c2Qgc[index].append(raw_c2Qgc[i])

    ascounts = []

    for i in range(0,len(lengths)):
        ascounts.append(_np.mean(_np.array(scounts[i])))

    return lengths, ascounts, scounts, cdepths, c2Qgc