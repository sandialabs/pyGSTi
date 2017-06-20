#!/usr/bin/env python3
import subprocess, sys

def main(args):
    assert len(args) == 3
    subprocess.call('python{} -m cProfile -s cumtime {} 2>&1 |  tee {}'.format(
        args[0], args[1], args[2]), shell=True)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
