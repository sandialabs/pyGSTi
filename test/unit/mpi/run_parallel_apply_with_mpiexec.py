# Run via: mpiexec -np 4 python -W ignore run_parallel_apply_with_mpiexec.py
import pygsti
from mpi4py import MPI

comm = MPI.COMM_WORLD


def square(x):
    return x * x


def test_correctness_and_order():
    # More items than procs: exercises the normal distribution path.
    # Checking equality against the serially-computed list verifies both
    # correctness and that output order matches input order.
    items = list(range(20))
    result = pygsti.parallel_apply(square, items, comm)
    expected = [x * x for x in items]
    assert result == expected, f"rank {comm.Get_rank()}: expected {expected}, got {result}"


def test_fewer_items_than_procs():
    # Fewer items than procs: exercises the comm-splitting path in
    # distribute_indices where multiple ranks are assigned the same item.
    items = [3, 7]
    result = pygsti.parallel_apply(square, items, comm)
    assert result == [9, 49], f"rank {comm.Get_rank()}: expected [9, 49], got {result}"


if __name__ == '__main__':
    test_correctness_and_order()
    test_fewer_items_than_procs()
