from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timed_block(label):
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print('{} block took {} seconds'.format(label, str(end-start)))
