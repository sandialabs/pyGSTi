#!/usr/bin/env python3
from read_profile_info import read_profile_info

from collections import OrderedDict
from pprint import pprint

def main():
    dirname = 'output/'
    # Ordered by optimization importance
    files   = ['basic.out', 'cp.out', 'tp.out']
    paths   = [dirname + filename for filename in files]

    filefunctionlists = [read_profile_info(fname) for fname in paths]

    functionDict = OrderedDict()
    for filefunctions in filefunctionlists:
        for calls, time, fname in filefunctions:
            if fname in functionDict:
                functionDict[fname].append((calls, time))
            else:
                functionDict[fname] = [(calls, time)]
    valueStr      = '{:>4}%,{:>4}%'
    tableStr      = '{} | {}'
    formattedCols = ['{:>11}'.format(f) for f in files]
    colStr        = '|'.join(formattedCols)
    print(tableStr.format(colStr, ''))
    colVals       = [('calls', 'time')] * len(files)
    formattedCols = ['{:>5},{:>5}'.format(a, b) for a, b in colVals]
    colStr        = '|'.join(formattedCols)
    print(tableStr.format(colStr, 'functions'))
    print('-' * 80)
    for k, v in functionDict.items():
        if len(v) == 3:
            formattedVals = [valueStr.format(a, b) for a, b in v]
            vStr = '|'.join(formattedVals)
            print(tableStr.format(vStr, k))

if __name__ == '__main__':
    main()
