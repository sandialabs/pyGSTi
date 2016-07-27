import nose
from nose.plugins import Plugin
from mpi4py import MPI

is_root = MPI.COMM_WORLD.Get_rank() == 0


class NoopStream(object):
    def write(self, *args):
        pass

    def writeln(self, *args):
        pass

    def flush(self):
        pass

class MpiOutput(Plugin):
    """
    Have only rank 0 report test results. Test results are aggregated
    across processes, i.e., if an exception happens in a single
    process then that is reported, otherwise if an assertion failed in any
    process then that is reported, otherwise it's a success.
    """
    # Required attributes:

    name = 'mpi'
    enabled = True

    def setOutputStream(self, stream):
        if not is_root:
            return NoopStream()

if __name__ == '__main__':
    import sys
    import os

    # This didn't work, mpich2 would tend to crash *shrug*
    #if MPI.COMM_WORLD.Get_size() == 1:
    #    # Launch using mpiexec
    #    args = [sys.argv[0], '1']
    #    print args
    #    sys.stderr.write('Launched without mpiexec; '
    #                     'calling with "mpiexec -np %d ..."\n' % WANTED_COMM_SIZE)
    #    os.execlp('mpiexec', 'mpiexec', '-np', str(WANTED_COMM_SIZE),
    #              sys.executable, *args)
    #    # Does not return!

    nose.main(addplugins=[MpiOutput()], argv=sys.argv)
