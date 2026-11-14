#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Evaluating lane objective functions, serially or in parallel worker processes.

Every lane term of a :class:`~pygsti.extras.crosstalkfreegst.fit.FactorizedObjective` is
independent given the global parameter vector, so the terms are split among persistent worker
processes.  Each worker builds its own terms' pyGSTi objective functions once per fit stage
(:meth:`LanePool.load`); after that an evaluation sends each worker only the global parameter
vector, and gets back each term's ``r.r`` and, for derivatives, ``J^T r`` and ``J^T J``.  These are
parameter-sized, so the (much larger) lane Jacobians never cross a process boundary.
"""
import os as _os
import subprocess as _subprocess
import sys as _sys
import threading as _threading
import time as _time
import traceback as _traceback
from multiprocessing.connection import Listener as _Listener

import numpy as _np

from pygsti.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitsStore
from pygsti.objectivefns import ObjectiveFunctionBuilder as _ObjectiveFunctionBuilder


def build_lane_objective(lane_model, dataset, circuits, objective, max_total=None):
    """
    A pyGSTi objective function for one lane, able to compute ``lsvec`` and ``dlsvec``.

    Pooling makes a lane circuit's total count large (the sum over every circuit containing
    it), so one observed shot can have a frequency below pyGSTi's default ``min_prob_clip``
    (1e-4) for the log-likelihood, whose regularization then fails ("it should be smaller
    than the smallest frequency").  It is therefore lowered, per lane, below the smallest
    possible nonzero frequency, ``1 / max_total``.

    `max_total` is the largest total count of any of the lane's circuits (default: of any of
    `circuits`).  Pass it when `circuits` is only a chunk of the lane's circuits, so that every
    chunk is regularized alike and the objective does not depend on how the lane was chunked.
    """
    builder = _ObjectiveFunctionBuilder.cast(objective)
    regularization = builder.regularization
    if regularization and 'min_prob_clip' in regularization:
        if max_total is None:
            max_total = max((dataset[c].total for c in circuits), default=1.0)
        min_p = min(regularization['min_prob_clip'], 0.5 / max(max_total, 1.0))
        if min_p < regularization['min_prob_clip']:
            builder = _ObjectiveFunctionBuilder(builder.cls_to_build, builder.name, builder.description,
                                                dict(regularization, min_prob_clip=min_p), builder.penalties,
                                                **builder.additional_args)
    array_types = builder.compute_array_types(('lsvec', 'dlsvec'), lane_model.sim)
    store = _ModelDatasetCircuitsStore(lane_model, dataset, circuits, array_types=array_types)
    return builder.build_from_store(store)


def num_outcomes(lane_model, circuits) -> int:
    """
    The number of possible outcomes of `circuits`, summed over them.

    An objective function's layout can hold fewer elements: where a circuit's counts are stored
    sparsely (lane circuits with instruments), it has only the outcomes that were observed.
    """
    return sum(len(lane_model.circuit_outcomes(c)) for c in circuits)


def lane_pieces(objfn, x_lane, derivatives: bool):
    """``r.r`` for one lane, or ``(r.r, J^T r, J^T J)`` if `derivatives`."""
    r = objfn.lsvec(x_lane)
    rr = float(_np.dot(r, r))
    if not derivatives:
        return rr
    J = objfn.dlsvec(x_lane)
    return rr, J.T @ r, J.T @ J


def _worker(conn):
    """A worker process's loop: build terms on 'load', evaluate them on 'fn'/'grad', exit on 'stop'."""
    terms = []  # (term index, global indices, objective function)
    while True:
        command, payload = conn.recv()
        try:
            if command == 'stop':
                break
            if command == 'load':
                specs, objective = payload
                terms = []  # so that a failed load leaves no terms, rather than the previous ones
                built, counts = [], []
                for i, model, gidx, dataset, circuits, max_total in specs:
                    built.append((i, gidx, build_lane_objective(model, dataset, circuits, objective, max_total)))
                    counts.append((i, num_outcomes(model, circuits)))
                terms = built
                conn.send(('ok', counts))
            else:
                x = payload
                conn.send(('ok', [(i, lane_pieces(objfn, x[gidx], command == 'grad')) for i, gidx, objfn in terms]))
        except Exception:
            conn.send(('error', _traceback.format_exc()))
    conn.close()


class LanePool(object):
    """
    Persistent worker processes that evaluate lane objective terms in parallel.

    Use it as a context manager, or call :meth:`close`.  Each worker is a fresh Python
    interpreter (running :mod:`._worker_main`), connected back over an authenticated local
    socket.  That avoids the pitfalls of :mod:`multiprocessing`'s start methods: forking a
    multithreaded process (BLAS starts threads) can deadlock the child, and
    ``'spawn'``/``'forkserver'`` re-import the calling script, so an unguarded script would
    re-run itself in every worker.
    BLAS/OpenMP are limited to one thread per worker so that `processes` workers do not
    oversubscribe the machine.

    Parameters
    ----------
    processes : int
        Number of worker processes.

    timeout : float, optional
        Seconds to wait for each worker to start and connect.
    """

    _THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

    def __init__(self, processes: int, timeout: float = 300.0):
        if processes < 1:
            raise ValueError("processes must be at least 1")
        authkey = _os.urandom(32)
        listener = _Listener(authkey=authkey)
        env = dict(_os.environ, **{v: '1' for v in self._THREAD_VARS})
        self._conns, self._procs = [], []
        try:
            for _ in range(processes):
                command = [_sys.executable, '-m', 'pygsti.extras.crosstalkfreegst._worker_main',
                           str(listener.address), authkey.hex()]
                self._procs.append(_subprocess.Popen(command, env=env))
            for _ in range(processes):
                self._conns.append(self._accept(listener, timeout))
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()
        self.num_terms = None  # the number of terms loaded; None until a load succeeds
        self.num_loads = 0  # lets a FactorizedObjective tell whether its terms are still the ones loaded

    def _accept(self, listener, timeout):
        """Accept one worker's connection, failing (not hanging) if the workers die or time out."""
        result = {}

        def accept():
            try:
                result['conn'] = listener.accept()
            except Exception as e:  # e.g. the listener was closed
                result['error'] = e
        thread = _threading.Thread(target=accept, daemon=True)
        thread.start()
        deadline = _time.time() + timeout
        while thread.is_alive():
            thread.join(0.2)
            if all(p.poll() is not None for p in self._procs) or _time.time() > deadline:
                listener.close()  # unblocks accept()
                thread.join(5)
                raise RuntimeError("Crosstalk-free GST worker processes failed to start (exit codes "
                                   f"{[p.poll() for p in self._procs]}).")
        if 'conn' not in result:
            raise RuntimeError(f"Failed to connect to a crosstalk-free GST worker: {result.get('error')!r}")
        return result['conn']

    @property
    def processes(self) -> int:
        return len(self._procs)

    def _exchange(self, messages):
        """
        Send each worker its message and return the workers' replies.

        An error in a worker is raised only once every worker has replied.  Raising at the first
        error would leave the other replies unread, to be taken for the answers to the next call.
        If a worker has died the pool is closed, as it cannot be used again.
        """
        if not self._conns:
            raise RuntimeError("This LanePool is closed.")
        results, errors, died = [], [], False
        for conn, message in zip(self._conns, messages):
            try:
                conn.send(message)
            except (OSError, EOFError):
                died = True
        for conn in self._conns:
            try:
                status, result = conn.recv()
            except (OSError, EOFError):
                died = True
                continue
            if status == 'error':
                errors.append(result)
            else:
                results.append(result)
        if died:
            self.close()
            raise RuntimeError("A crosstalk-free GST worker process died unexpectedly; the pool is now closed."
                               + ''.join(f"\nError in another worker:\n{e}" for e in errors))
        if errors:
            raise RuntimeError("Error in a crosstalk-free GST worker process:\n" + "\n".join(errors))
        return results

    def load(self, specs, objective):
        """
        Distribute lane terms among the workers, which build their objective functions.

        `specs` is a list of ``(lane_model, global_indices, dataset, circuits, max_total)``, one
        per term, the last three as for :func:`build_lane_objective`.  Terms are assigned greedily
        by estimated cost (circuits x parameters), largest first, each to the least-loaded worker.
        They replace the terms loaded before; if a worker fails to build its terms, none are left
        loaded.  Returns each term's number of possible outcomes (:func:`num_outcomes`), in order.
        """
        if not self._conns:
            raise RuntimeError("This LanePool is closed.")
        self.num_terms = None
        loads = [0.0] * self.processes
        assignments = [[] for _ in range(self.processes)]
        order = sorted(range(len(specs)), key=lambda i: -len(specs[i][3]) * len(specs[i][1]))
        for i in order:
            w = loads.index(min(loads))
            loads[w] += len(specs[i][3]) * max(len(specs[i][1]), 1)
            assignments[w].append((i,) + tuple(specs[i]))
        num_elements = [0] * len(specs)
        for reply in self._exchange([('load', (assigned, objective)) for assigned in assignments]):
            for i, n in reply:
                num_elements[i] = n
        self.num_terms = len(specs)
        self.num_loads += 1
        return num_elements

    def evaluate(self, x, derivatives: bool):
        """Every loaded term's :func:`lane_pieces` at global parameter vector `x`, in term order."""
        if self._conns and self.num_terms is None:
            raise RuntimeError("No lane terms are loaded in this LanePool.")
        out = [None] * (self.num_terms or 0)
        for reply in self._exchange([('grad' if derivatives else 'fn', x)] * len(self._conns)):
            for i, pieces in reply:
                out[i] = pieces
        return out

    def close(self):
        for conn in self._conns:
            try:
                conn.send(('stop', None))
                conn.close()
            except (OSError, EOFError):
                pass
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except _subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._conns, self._procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
