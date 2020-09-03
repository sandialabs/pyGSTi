


from pygsti.extras.interpygate import PhysicalProcess
from pygsti.tools import change_basis
from pygsti.extras.interpygate.process_tomography import do_process_tomography, vec, unvec

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    _comm = None

from scipy.linalg import logm as _logm, expm as _expm
import numpy as _np

class ProcessFunction(object):
    def __init__(self):
        import numpy as _np
        import scipy as _sp

        self.Hx = _np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, -1],
                            [0, 0, 1, 0]], dtype='float')
        self.Hy = _np.array([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, -1, 0, 0]], dtype='float')
        self.Hz = _np.array([[0, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]], dtype='float')

        self.dephasing_generator = _np.diag([0, -1, -1, 0])
        self.decoherence_generator = _np.diag([0, -1, -1, -1])

    def advance(self, state, v = None, times = None):
        state = _np.array(state, dtype='complex')
        if times is None:
            t, omega, phase, decoherence = v
            times = [t]
        else:
            omega, phase, decoherence = v

        H = (omega * _np.cos(phase) * self.Hx + omega * _np.sin(phase) * self.Hy)
        L = decoherence * self.decoherence_generator

        processes = [change_basis(_expm((H + L) * t),
                                                          'pp', 'col') for t in times]
        states = [unvec(_np.dot(process, vec(_np.outer(state, state.conj())))) for process in processes]

        return states

    def __call__(self, v, times=None, comm=_comm, return_auxdata=True):
        print(f'Calling process tomography as {comm.Get_rank()+1} of {comm.Get_size()} on {comm.Get_name()}.')
        processes = do_process_tomography(self.advance, opt_args={'v':v, 'times':times},
                                          n_qubits = 1, time_dependent=True, comm=comm)

        if return_auxdata:
            if times is not None:
                auxdata = _np.array([times] + [list(_np.arange(len(times)))] + list(_np.array([v]*len(times)).T))
            else:
                auxdata = _np.array([time] + [0] + [list(v)])

        if return_auxdata:
            return processes, auxdata
        else:
            return processes

gy = PhysicalProcess(mpi_workers_per_process=1, basis='col')
gy.set_process_function(ProcessFunction(), mpi_enabled=True, has_auxdata=True)
target = change_basis(_np.array([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,0,-1],
                                 [0,0,1,0]], dtype='complex'), 'pp', 'qsim')
gy.set_target(target)

# # # Evaluate one time point per run
# # gy.set_parameter_range([[_np.pi/2-.5, _np.pi/2+.5],[0.9,1.1],[-.1,.1],[-.1,.1],[0,0.1],[0,0.1]])
# # gy.set_interpolation_order([3,3,3,3,3,3])
#
# Evaluate many time points per run
# Notice that the number of parameters listed below is 1 fewer than in the previous example
gy.set_parameter_range([[0.9, 1.1], [-.1, .1], [0, 0.1]])
gy.set_interpolation_order([3,3,3])
gy.set_times(_np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10))


gy.interpolate()
gy.save('intertest.dat')

# #
# # def diamond_norm(a, b):
# #     A = pygsti.tools.jamiolkowski_iso(a)
# #     B = pygsti.tools.jamiolkowski_iso(b)
# #     return pygsti.tools.diamonddist(a, b)
# #
# #

if _rank == 0:
    from matplotlib import pyplot as plt
    cb = lambda x: change_basis(x, 'qsim', 'pp')
    gx = PhysicalProcess('intertest.dat')
    print(_np.round_(gx.from_vector([_np.pi / 2 + .3, 1, 0, 0]),3))
    # print(_np.round_(gx.from_vector_physical([_np.pi / 2 + .3, 1, 0, 0, 0, 0]),3))
    nom = _np.array([_np.pi / 2, 1, 0, 0])
    labels = ["Timing Error", "Amplitude Error", "Phase Error", "Frequency Error", "Additional Dephasing", "Additional Decoherence"]
    for ind in range(len(nom)):
        tnom = nom.copy()
        tnom[ind] += .1
        ergen = _np.round_(cb(gx._error_generator_from_gate(gx(tnom))), 3) / .1
        plt.matshow(ergen.real)
        plt.title(labels[ind])
        print(ind)
        print(ergen)
        plt.savefig(f'./figures/param_{ind}.pdf')
