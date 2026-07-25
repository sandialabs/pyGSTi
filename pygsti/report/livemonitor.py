"""
A "live" monitoring dashboard for a running (or already-completed)
:class:`~pygsti.protocols.gst.GateSetTomography` (or
:class:`~pygsti.protocols.gst.StandardGST`) fit.

Overview
--------
GST fits proceed as a sequence of iterations, each fitting a model to a
circuit list of increasing length ("depth"), followed - at the very end,
and only at the very end - by gauge optimization. Most familiar reportable
quantities (fidelity to target, SPAM tables, diamond-norm-to-target, ...)
depend on a choice of gauge, and so are meaningless (or actively misleading)
for any model prior to that final gauge-optimization step. See
:mod:`pygsti.report.livemetrics` for the two families of quantities that
*are* safe to report before gauge optimization: goodness-of-fit statistics,
and gate eigenvalue spectra.

This module watches the checkpoint files that
:meth:`~pygsti.protocols.gst.GateSetTomography.run` already writes to disk
(once per completed circuit-depth iteration, by default) and renders those
two families of gauge-invariant quantities as a live-updating
:class:`plotly.graph_objects.FigureWidget`, suitable for display in a
Jupyter notebook.

Execution model
----------------
Because a blocking call to ``protocol.run(...)`` occupies the entire
notebook kernel, updating a widget from a background thread while that
blocking call is in progress is unreliable (there is no guarantee that
front-end widget state syncs get flushed while the main thread is busy in a
long-running, non-cooperative computation). :class:`LiveGSTMonitor`
therefore inverts the usual relationship: *it* is the foreground loop, and
the actual GST fit runs somewhere else:

- :meth:`LiveGSTMonitor.watch` attaches to an already-running (or already
  finished) job's checkpoint directory - e.g. one launched separately via
  :meth:`~pygsti.protocols.protocol.Protocol.run_mpi`,
  :meth:`~pygsti.protocols.protocol.Protocol.stage_slurm` plus ``sbatch``,
  or simply a plain script running elsewhere. This works identically
  whether the fit is local or on a shared cluster filesystem, since it only
  ever needs read access to the checkpoint directory.

- :meth:`LiveGSTMonitor.run` is a convenience wrapper that additionally
  launches the fit for you, as a single (non-MPI) child process, and then
  watches it the same way. This intentionally does *not* attempt to also
  reuse pyGSTi's MPI-launcher machinery (which pulls in a hard dependency
  on ``mpi4py``); for a multi-rank run, launch it yourself (e.g. via
  ``run_mpi``/``stage_slurm``) and use :meth:`watch` to monitor it.

In both cases, the polling loop runs on the calling (main) thread, so
widget updates happen exactly where and when they're flushed to the
notebook front end - the notebook cell that calls ``.watch()``/``.run()``
is simply busy until the fit finishes (or the loop is interrupted).

Rendering backend
------------------
This module uses :class:`plotly.graph_objects.FigureWidget` to get
live-updating plots in a running notebook. Depending on the installed
version of plotly, this requires either the ``anywidget`` package (plotly
>= 6) or the ``ipywidgets`` package (older plotly) to be installed; neither
is a hard dependency of pyGSTi. A clear error is raised (only when actually
attempting to display a figure) if neither is available.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import glob as _glob
import json as _json
import os as _os
import pathlib as _pathlib
import pickle as _pickle
import re as _re
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
import time as _time
import warnings as _warnings
from datetime import datetime as _datetime

import numpy as _np

from pygsti.report import livemetrics as _livemetrics

_ITERATION_FILE_RE = _re.compile(r'_iteration_(\d+)\.json$')


class LiveGSTMonitorError(RuntimeError):
    """Raised for LiveGSTMonitor usage errors (e.g. a child fitting process failing)."""
    pass


def _terminate_child_process(proc, grace_period=5.0):
    """
    Terminate a child `subprocess.Popen` process, escalating from a polite
    `SIGTERM` to a forceful `SIGKILL` if it doesn't exit within
    `grace_period` seconds. Safe to call on a process that has already
    exited (a no-op in that case).
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_period)
    except _subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _build_figure_widget():
    """
    Construct an (empty) 1x2 `plotly.graph_objects.FigureWidget`: a
    goodness-of-fit-vs-iteration panel on the left, and a gate-eigenvalue
    (complex plane) panel on the right.

    Raises
    ------
    LiveGSTMonitorError
        If plotly cannot construct a `FigureWidget` because neither
        ``anywidget`` (plotly >= 6) nor ``ipywidgets`` (older plotly) is
        installed.
    """
    try:
        import plotly.graph_objects as _go
        from plotly.subplots import make_subplots as _make_subplots
    except ImportError as e:
        raise LiveGSTMonitorError(
            "LiveGSTMonitor requires plotly to be installed.") from e

    fig = _make_subplots(
        rows=1, cols=2,
        subplot_titles=("Goodness of fit vs. circuit depth (provisional - pre-gauge-opt)",
                        "Gate eigenvalues, latest depth (provisional - pre-gauge-opt)"))

    fig.add_trace(_go.Scatter(x=[], y=[], mode='lines+markers', name='p-value',
                              hovertemplate='iteration %{x}<br>p-value=%{y:.3g}<extra></extra>'),
                 row=1, col=1)
    # Reference line: a commonly-used "fit looks bad" p-value threshold.
    fig.add_hline(y=0.05, line=dict(color='firebrick', dash='dash', width=1),
                 annotation_text='p=0.05', row=1, col=1)
    fig.update_yaxes(title_text='p-value', range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text='iteration (circuit-depth stage)', row=1, col=1)

    # Unit-circle reference for the eigenvalue plot.
    theta = _np.linspace(0, 2 * _np.pi, 200)
    fig.add_trace(_go.Scatter(x=_np.cos(theta), y=_np.sin(theta), mode='lines',
                              line=dict(color='lightgray', dash='dot'),
                              hoverinfo='skip', showlegend=False, name='unit circle'),
                 row=1, col=2)
    fig.update_yaxes(title_text='Im(eigenvalue)', scaleanchor='x2', row=1, col=2)
    fig.update_xaxes(title_text='Re(eigenvalue)', row=1, col=2)

    fig.update_layout(title='LiveGSTMonitor - no data yet', showlegend=True, width=1000, height=450)

    try:
        return _go.FigureWidget(fig)
    except ImportError as e:
        raise LiveGSTMonitorError(
            "LiveGSTMonitor could not construct a live-updating plotly figure. "
            "This requires an extra package to be installed alongside plotly: "
            "'anywidget' for plotly>=6, or 'ipywidgets' for older plotly versions.") from e


class LiveGSTMonitor:
    """
    Watches a GST checkpoint directory and renders a live-updating,
    gauge-invariant summary of fit progress.

    Parameters
    ----------
    checkpoint_prefix : str or Path
        The checkpoint path/name prefix, i.e. the same value that was (or
        will be) passed as `checkpoint_path` to
        :meth:`~pygsti.protocols.gst.GateSetTomography.run` (or
        :meth:`~pygsti.protocols.gst.StandardGST.run`). Checkpoint files are
        expected at ``f'{checkpoint_prefix}_iteration_{i}.json'``.

    mode : str, optional
        For a `StandardGST` run's checkpoints (which fit multiple named
        "modes", e.g. 'full TP', 'CPTPLND', as child protocols): which
        mode's checkpoints to monitor. Only needed if the checkpoint files
        contain more than one GST mode; see
        :func:`pygsti.report.livemetrics.unwrap_gst_checkpoint_state`.

    poll_interval : float, optional (default 2.0)
        How many seconds to sleep between successive checks of the
        checkpoint directory.

    n_iterations : int, optional
        The total number of circuit-depth iterations the fit is expected to
        run for, if known (e.g. ``len(data.edesign.circuit_lists)``). When
        given, :meth:`watch` automatically stops once this many iterations
        have completed. If not given, :meth:`watch` polls until manually
        interrupted (e.g. via a notebook "stop" button / KeyboardInterrupt)
        or an optional `timeout` elapses.
    """

    def __init__(self, checkpoint_prefix, mode=None, poll_interval=2.0, n_iterations=None):
        self.checkpoint_prefix = str(checkpoint_prefix)
        self.mode = mode
        self.poll_interval = poll_interval
        self.n_iterations = n_iterations

        self._fig = None
        self._last_rendered_iter = -1
        self._last_update_time = None

    # ------------------------------------------------------------------
    # Checkpoint discovery / safe reading
    # ------------------------------------------------------------------
    def _discover_iteration_indices(self):
        """
        Return a sorted list of iteration indices for which a checkpoint
        file currently exists on disk (regardless of whether that file has
        finished being written).
        """
        pattern = f'{self.checkpoint_prefix}_iteration_*.json'
        indices = []
        for path in _glob.glob(pattern):
            m = _ITERATION_FILE_RE.search(path)
            if m is not None:
                indices.append(int(m.group(1)))
        return sorted(indices)

    def _read_state(self, index, n_retries=3, retry_delay=0.1):
        """
        Attempt to JSON-parse the checkpoint file for `index`, retrying a
        few times (with a short delay) in case the writer is still in the
        middle of writing it (writes are not currently atomic - see
        `pygsti.baseobjs.nicelyserializable.NicelySerializable.write` - so a
        reader can transiently observe a partially-written file for a file
        that was *just* created).

        Returns
        -------
        dict or None
            The parsed checkpoint state, or None if it could not be parsed
            after all retries (e.g. it's still being written).
        """
        path = f'{self.checkpoint_prefix}_iteration_{index}.json'
        for attempt in range(n_retries):
            try:
                return _livemetrics.load_checkpoint_state(path)
            except (_json.JSONDecodeError, OSError):
                if attempt < n_retries - 1:
                    _time.sleep(retry_delay)
        return None

    def _read_latest_safe_state(self):
        """
        Determine the highest iteration index that can be safely read, and
        return `(index, state)` for it (or `(None, None)` if no checkpoint
        is available yet).

        Safe-read rule: checkpoint files are written once, to a new
        filename, and never rewritten - so if
        ``f'{prefix}_iteration_{i+1}.json'`` exists, then
        ``f'{prefix}_iteration_{i}.json'`` is guaranteed to be a completely-
        written file. For the single highest-index file currently on disk
        (which might still be mid-write, since there's no subsequent file to
        prove otherwise), we fall back to a parse-retry strategy instead.
        """
        indices = self._discover_iteration_indices()
        if not indices:
            return None, None

        highest = indices[-1]
        # Every index below the highest is guaranteed-complete; only the
        # highest one might still be mid-write.
        state = self._read_state(highest)
        if state is not None:
            return highest, state

        if len(indices) > 1:
            # The highest-index file isn't parseable (yet); fall back to the
            # previous (guaranteed-complete) one.
            second_highest = indices[-2]
            state = self._read_state(second_highest, n_retries=1)
            if state is not None:
                return second_highest, state

        return None, None

    # ------------------------------------------------------------------
    # Widget construction/update
    # ------------------------------------------------------------------
    def _ensure_figure(self):
        if self._fig is None:
            self._fig = _build_figure_widget()
        return self._fig

    def _update_figure(self, index, state):
        metrics = _livemetrics.extract_live_metrics(state, mode=self.mode)
        gof_history = metrics['gof_history']
        eigenvalues = metrics['eigenvalues']

        fig = self._ensure_figure()
        with fig.batch_update():
            gof_trace = fig.data[0]
            gof_trace.x = [entry['iteration'] for entry in gof_history]
            gof_trace.y = [entry['pvalue'] for entry in gof_history]
            gof_trace.customdata = [
                (entry['chi2k_distributed_qty'], entry['degrees_of_freedom'], entry['objfn_description'])
                for entry in gof_history]
            gof_trace.hovertemplate = (
                'iteration %{x}<br>p-value=%{y:.3g}<br>chi2k/2dLogL=%{customdata[0]:.4g}'
                '<br>dof=%{customdata[1]}<br>%{customdata[2]}<extra></extra>')

            # Rebuild the eigenvalue scatter traces. Indices 0 and 1 are the
            # persistent p-value trace and unit-circle reference; drop
            # everything after that (the previous poll's eigenvalue traces)
            # before adding freshly-computed ones. `fig.data = ...` only
            # supports assigning a permutation of a *subset* of the current
            # traces, so trimming to the first two traces this way (rather
            # than trying to splice in new ones directly) is required before
            # calling `add_traces` to append the new eigenvalue traces back
            # in, correctly wired up to the second subplot's axes.
            fig.data = fig.data[:2]
            import plotly.graph_objects as _go
            new_traces = []
            for lbl, evals in eigenvalues.items():
                evals = _np.asarray(evals)
                new_traces.append(_go.Scatter(
                    x=_np.real(evals), y=_np.imag(evals), mode='markers',
                    name=str(lbl), marker=dict(size=9),
                    hovertemplate=f'{lbl}<br>' + 're=%{x:.4g}<br>im=%{y:.4g}<extra></extra>'))
            if new_traces:
                fig.add_traces(new_traces, rows=[1] * len(new_traces), cols=[2] * len(new_traces))

            self._last_update_time = _datetime.now()
            n_str = str(self.n_iterations) if self.n_iterations is not None else '?'
            fig.layout.title = (
                f"LiveGSTMonitor - iteration {index}"
                + (f" of {n_str}" if self.n_iterations is not None else "")
                + f" - last updated {self._last_update_time.strftime('%H:%M:%S')}"
                + " [PROVISIONAL: pre-gauge-optimization]")

    def display(self):
        """
        Display (or re-display) this monitor's figure widget, e.g. in a
        Jupyter notebook cell. Safe to call before any data has arrived
        (shows an empty/placeholder figure).

        Returns
        -------
        plotly.graph_objects.FigureWidget
        """
        fig = self._ensure_figure()
        try:
            from IPython.display import display as _ipy_display
            _ipy_display(fig)
        except ImportError:
            # Not running in an environment with IPython display support;
            # just return the figure for the caller to render themselves.
            pass
        return fig

    def poll_once(self):
        """
        Check the checkpoint directory once and update the figure if a new
        iteration has completed since the last check.

        Returns
        -------
        int or None
            The iteration index that was rendered as a result of this poll
            (whether newly discovered or already-current), or None if no
            checkpoint data is available yet.
        """
        index, state = self._read_latest_safe_state()
        if index is not None and index > self._last_rendered_iter:
            self._update_figure(index, state)
            self._last_rendered_iter = index
        return index

    # ------------------------------------------------------------------
    # Blocking polling loops
    # ------------------------------------------------------------------
    def watch(self, timeout=None):
        """
        Attach to (an already-running, or already-completed) job's
        checkpoint directory and block, polling and updating the live
        figure, until the fit appears to be done (i.e.
        `n_iterations` completed iterations have been observed, if
        `n_iterations` was given), `timeout` seconds have elapsed, or the
        call is interrupted (e.g. via KeyboardInterrupt in a notebook).

        This does not launch or control the fit itself - use this to
        monitor a job launched separately (locally, via MPI, or via a
        cluster scheduler) that shares a filesystem with wherever this is
        called from.

        Parameters
        ----------
        timeout : float, optional
            Maximum number of seconds to poll for, regardless of whether
            `n_iterations` has been reached. If None (the default) and
            `n_iterations` was also not given at construction, this polls
            indefinitely until interrupted.

        Returns
        -------
        int
            The last iteration index observed (-1 if none was ever seen).
        """
        self.display()
        start_time = _time.time()
        try:
            while True:
                self.poll_once()

                done = (self.n_iterations is not None
                       and self._last_rendered_iter >= self.n_iterations - 1)
                timed_out = (timeout is not None) and (_time.time() - start_time) > timeout
                if done or timed_out:
                    break
                _time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            pass
        return self._last_rendered_iter

    def run(self, protocol, data, run_kwargs=None, python_executable=None, timeout=None):
        """
        Launch `protocol.run(data, checkpoint_path=self.checkpoint_prefix,
        **run_kwargs)` as a single, non-MPI child process, then block,
        watching and rendering its progress the same way `watch` does,
        until the child process finishes (or `timeout` elapses).

        For a multi-rank (MPI) run, launch the fit yourself via
        :meth:`~pygsti.protocols.protocol.Protocol.run_mpi` /
        :meth:`~pygsti.protocols.protocol.Protocol.stage_slurm` and use
        :meth:`watch` to monitor it instead - this method intentionally
        stays free of any dependency on `mpi4py`.

        Parameters
        ----------
        protocol : Protocol
            The (unrun) protocol to execute, e.g. a
            `GateSetTomography`/`StandardGST` instance.

        data : ProtocolData
            The input data to run the protocol on.

        run_kwargs : dict, optional
            Extra keyword arguments forwarded to `protocol.run` in the child
            process (e.g. `simulator`, `optimizers`). `checkpoint_path` and
            `disable_checkpointing` are set automatically and should not be
            included here.

        python_executable : str, optional
            Path to the Python interpreter to use for the child process.
            Defaults to `sys.executable` (the interpreter currently running).

        timeout : float, optional
            If given, stop waiting for the child process after this many
            seconds even if it hasn't finished, and terminate it (see
            Returns below). Note this also determines the lifetime of a
            temporary working directory the child process depends on (see
            Notes), so the child *cannot* simply be left running past this
            point.

        Returns
        -------
        ProtocolResults or None
            The final results of the run, once the child process completes
            normally. If `timeout` is reached, or this call is interrupted
            (e.g. via KeyboardInterrupt in a notebook) before the child
            finishes, the child process is terminated (`SIGTERM`, escalating
            to `SIGKILL` if it doesn't exit promptly) and `None` is
            returned (or the `KeyboardInterrupt` is re-raised, respectively).

        Notes
        -----
        The child process reads its input (data/protocol/run_kwargs) from,
        and writes its final results back to, a temporary working directory
        that only lives as long as this call. Consequently the child process
        can never be left running unsupervised past the end of this call -
        doing so would orphan a process that's about to have the directory
        it depends on deleted out from under it. This is why, unlike
        `Protocol.run_mpi`'s `persistent_dir` option, there is no way to
        detach from a `run()`-launched child and keep it going in the
        background; use `Protocol.run_mpi`/`stage_slurm` (which use
        persistent working directories) plus `watch()` if you need that.
        """
        run_kwargs = dict(run_kwargs) if run_kwargs is not None else {}
        for bad_kwarg in ('checkpoint_path', 'checkpoint', 'disable_checkpointing'):
            if bad_kwarg in run_kwargs:
                raise ValueError(
                    f"run_kwargs should not include '{bad_kwarg}' - LiveGSTMonitor.run "
                    "sets this automatically.")
        run_kwargs['checkpoint_path'] = self.checkpoint_prefix
        run_kwargs['disable_checkpointing'] = False

        checkpoint_path = _pathlib.Path(self.checkpoint_prefix)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with _tempfile.TemporaryDirectory() as artifact_dir:
            artifact_dir = _pathlib.Path(artifact_dir)
            data.write(str(artifact_dir))
            protocol_dir = artifact_dir / 'protocol'
            protocol.write(str(protocol_dir))

            kwargs_path = artifact_dir / 'volatile_run_kwargs.pkl'
            with open(kwargs_path, 'wb') as f:
                _pickle.dump(run_kwargs, f)

            runner_path = artifact_dir / 'live_monitor_runner.py'
            runner_script = (
                "import pickle\n"
                "import pygsti\n"
                f"data = pygsti.io.read_data_from_dir({str(artifact_dir)!r})\n"
                f"protocol = pygsti.io.read_protocol_from_dir({str(protocol_dir)!r})\n"
                f"with open({str(kwargs_path)!r}, 'rb') as _f:\n"
                "    _kwargs = pickle.load(_f)\n"
                "results = protocol.run(data, **_kwargs)\n"
                f"results.write({str(artifact_dir)!r}, data_already_written=True)\n"
            )
            runner_path.write_text(runner_script)

            python_executable = python_executable or _sys.executable
            # Ensure the child process can `import pygsti` (and any other
            # currently-importable packages) even in a development setup
            # where pygsti is only importable because the parent process's
            # working directory happens to be on sys.path (e.g. via the `''`
            # sys.path entry `python -c`/interactive sessions add), rather
            # than being a properly `pip install`-ed package. We locate the
            # directory containing the `pygsti` package explicitly (instead
            # of just forwarding `sys.path`, which may contain a bare `''`
            # entry that only resolves correctly if the child's *working
            # directory* also happens to match) so this works regardless of
            # the child process's cwd. Note: pygsti's existing
            # `Protocol.run_mpi` subprocess-launching code has this same
            # implicit assumption and does not do this - this is a small,
            # self-contained robustness improvement local to this new
            # spawn-and-watch code path.
            import pygsti as _pygsti_pkg
            pygsti_root = str(_pathlib.Path(_pygsti_pkg.__file__).resolve().parent.parent)

            child_env = dict(_os.environ)
            existing_pythonpath = child_env.get('PYTHONPATH', '')
            other_sys_path_entries = _os.pathsep.join(p for p in _sys.path if p and p != pygsti_root)
            child_env['PYTHONPATH'] = _os.pathsep.join(
                entry for entry in (pygsti_root, other_sys_path_entries, existing_pythonpath) if entry)

            proc = _subprocess.Popen([python_executable, str(runner_path)], env=child_env)

            self.display()
            start_time = _time.time()
            timed_out = False
            interrupted = False
            try:
                while proc.poll() is None:
                    self.poll_once()
                    if timeout is not None and (_time.time() - start_time) > timeout:
                        timed_out = True
                        break
                    _time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                if timed_out or interrupted:
                    # The child process depends on `artifact_dir` (this
                    # method's temporary working directory) for both its
                    # inputs and its final results write, and that directory
                    # is about to be deleted when the `with` block below
                    # exits - so the child cannot simply be left running past
                    # this point (doing so would orphan it and it would fail
                    # as soon as it tried to use the now-deleted directory).
                    # Terminate it now, while the directory (and thus a
                    # graceful shutdown) is still possible.
                    _terminate_child_process(proc)
                # Pick up any final checkpoint state written right before exit
                # (or right before termination, for the timed-out/interrupted
                # cases - checkpoints are written to `self.checkpoint_prefix`,
                # which is independent of `artifact_dir` and persists either way).
                self.poll_once()

            if interrupted:
                _warnings.warn(
                    "LiveGSTMonitor.run: interrupted before the child process "
                    "finished; it has been terminated.")
                raise KeyboardInterrupt()

            if timed_out:
                _warnings.warn(
                    "LiveGSTMonitor.run: timeout reached before the child process "
                    "finished; it has been terminated.")
                return None

            if proc.returncode != 0:
                raise LiveGSTMonitorError(
                    f"The child GST fitting process exited with a non-zero return "
                    f"code ({proc.returncode}).")

            from pygsti import io as _io
            return _io.read_results_from_dir(str(artifact_dir), name=protocol.name)
