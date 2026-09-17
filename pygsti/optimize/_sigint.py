"""
Helper for installing Python's default SIGINT handler in long-running optimization routines.
"""
import os as _os
import signal as _signal
import threading as _threading


def install_sigint_handler():
    """
    Install Python's default SIGINT handler (raises KeyboardInterrupt) when called from the
    main thread and the environment variable PYGSTI_NO_CUSTOMLM_SIGINT is not set.

    Why this exists: when pyGSTi is launched as a background subprocess (e.g. under PBS or
    SLURM), SIGINT may be inherited as SIG_IGN from the parent process, making Ctrl-C
    ineffective during long GST optimizations.  This function restores the expected
    interactive behavior.

    The main-thread guard prevents a ValueError from signal.signal() when pyGSTi is imported
    from worker threads (pytest-xdist, Dask, MPI sub-communicators, etc.).  Callers may also
    suppress this entirely by setting PYGSTI_NO_CUSTOMLM_SIGINT in the environment.
    """
    if 'PYGSTI_NO_CUSTOMLM_SIGINT' not in _os.environ:
        if _threading.current_thread() is _threading.main_thread():
            _signal.signal(_signal.SIGINT, _signal.default_int_handler)
