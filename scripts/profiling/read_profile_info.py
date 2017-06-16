#!/usr/bin/env python3
import subprocess, sys
from collections import namedtuple
from pprint import pprint

FuncInfo = namedtuple('FuncInfo', ['calls', 'time', 'fname'])

def read_top_n_lines(filename, n, bodystart):
    assert n > 0
    inBody = False
    content = []
    with open(filename) as infile:
        for line in infile:
            if not inBody:
                if bodystart in line:
                    inBody = True
            elif n == 0:
                break
            else:
                content.append(line)
                n -= 1
    return content

def parse_lines(content):
    functions = []
    for line in content:
        values = line.split()
        calls   = int(values[0].split('/')[0])
        time = float(values[3])
        fname   = ' '.join(values[5:])
        functions.append(FuncInfo(calls, time, fname))
    totaltime = functions[0].time
    functions = [FuncInfo(calls, int(round(time/totaltime, 2) * 100), fname) for calls, time, fname in functions]
    return functions

def main(args):
    assert len(args) == 1
    filename  = args[0]
    bodystart = 'filename:lineno(function)'

    content   = read_top_n_lines(filename, 100, bodystart)
    functions = parse_lines(content)

    tableStr = '{:<10}| {:>4}% | {:<15}'
    print(tableStr.format('calls', 'time', 'name'))
    print('-' * 80)
    for funcInfo in functions:
        print(tableStr.format(
            *funcInfo
            ))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
