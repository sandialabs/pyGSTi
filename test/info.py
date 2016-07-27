#!/usr/bin/env python3
from __future__                import print_function
from helpers.automation_tools  import read_yaml, write_yaml
from helpers.info.genInfo      import gen_package_info
from helpers.info.process      import find_uncovered_lines, annotate_uncovered
from pprint import pprint
import argparse
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate info on pygsti tests')
    parser.add_argument('packages', nargs='*',
                        help='packages to update metadata on')
    parser.add_argument('--update', type=bool,
                        help='update the packages given')
    parser.add_argument('--process', type=bool,
                        help='process the packages given')

    parsed = parser.parse_args(sys.argv[1:])

    infoDict = read_yaml('output/test_info.yml')

    if parsed.update:
        # Update test info on given modules
        for packageName in parsed.packages:
            infoDict[packageName] = gen_package_info(packageName)
        write_yaml(infoDict, 'output/test_info.yml')

    if parsed.process:
        for packageName in parsed.packages:
            # interpret it, if asked
            annotate_uncovered(packageName, infoDict)
