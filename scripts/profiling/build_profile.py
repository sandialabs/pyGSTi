#!/usr/bin/env python3
import subprocess, sys

def main(args):
    assert len(args) == 2
    subprocess.call('python3 -m cProfile -s cumtime {} 2>&1 |  tee {}'.format(
        args[0], args[1]), shell=True)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
