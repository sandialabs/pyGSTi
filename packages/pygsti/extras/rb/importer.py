from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np

def import_txt_format_rb_data(filenames_list,verbosity=1):
    
    raw_lengths = []
    raw_sps = []
    raw_c_depths = []
    raw_c_2QGC = []

    for filename in filenames_list:
        if verbosity > 0:
            print("Importing "+filename+"...",end='')
        with open(filename,'r') as f:
            counter = 0
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                try:
                    raw_lengths.append(int(line[0])) 
                    raw_sps.append(float(line[1])) 
                    raw_c_depths.append(int(line[2])) 
                    raw_c_2QGC.append(int(line[3]))
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
    sps = []
    c_depths = []
    c_2QGC = []
    for i in range(0,len(lengths)):
        sps.append([])
        c_depths.append([])
        c_2QGC.append([])

    for i in range(0,len(raw_lengths)):
        index = lengths.index(raw_lengths[i])
        sps[index].append(raw_sps[i])
        c_depths[index].append(raw_c_depths[i])
        c_2QGC[index].append(raw_c_2QGC[i])

    asps = []

    for i in range(0,len(lengths)):
        asps.append(_np.mean(_np.array(sps[i])))

    return lengths, asps, sps, c_depths, c_2QGC