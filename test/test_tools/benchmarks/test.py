from __future__ import print_function
from benchmark  import benchmark

if __name__ == "__main__":
    @benchmark('benchmarks.txt')
    def func(x):
        print(x)
        return x * x
    print(func(2))
