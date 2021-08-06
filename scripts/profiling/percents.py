#!/usr/bin/env python3
import sys
from collections import namedtuple

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

def total_calls(parsed_lines):
    tuples = sorted(parsed_lines, key=lambda t : t.calls, reverse=True)
    return tuples[0].calls

def parse_lines(content):
    functions = []
    for line in content:
        values = line.split()
        calls   = int(values[0].split('/')[0])
        time = float(values[3])
        fname   = ' '.join(values[5:])
        functions.append(FuncInfo(calls, time, fname))
    totaltime = functions[0].time
    totalcalls = total_calls(functions)
    functions = [FuncInfo(
        int(round(calls/totalcalls, 2)* 100),
        int(round(time/totaltime, 2) * 100), 
        fname) for calls, time, fname in functions]
    return functions

def read_profile_info(filename):
    bodystart = 'filename:lineno(function)'

    content   = read_top_n_lines(filename, 100, bodystart)
    return parse_lines(content)

def main(args):
    assert len(args) == 1
    functions = read_profile_info(args[0])
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
