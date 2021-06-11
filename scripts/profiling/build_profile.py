#!/usr/bin/env python3
import subprocess
import sys


def main(args):
    assert len(args) == 2
    try:
        subprocess.call('python3 -m cProfile -o output/profile.txt -s cumtime {} 2>&1 |  tee {}'.format(
            args[0], args[1]), shell=True)
    finally:
        subprocess.call('./move_output > {}'.format(args[1]), shell=True)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
