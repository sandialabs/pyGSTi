"""
Defines the VerbosityPrinter class, used for logging output.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math as _math  # used for digit formatting
import sys as _sys
from contextlib import contextmanager as _contextmanager
from copy import deepcopy as _dc

from pygsti.baseobjs import _compatibility as _compat
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation


def _num_digits(n):
    return int(_math.log10(n)) + 1 if n > 0 else 1

# This function isn't a part of the public interface, instead it has a wrapper in the VerbosityPrinter class


def _build_progress_bar(iteration, total, bar_length=100, num_decimals=2, fill_char='#',
                        empty_char='-', prefix='Progress:', suffix='Complete', end='\n'):
    """
    Parameters
    ----------
    iteration   - int, required
      current iteration
    total       - int, required  :
      total iterations
    bar_length   - int, optional  :
      character length of bar
    num_decimals - int, optional  :
      precision of progress percent
    fill_char    - str, optional  :
      replaces '#' as the bar-filling character
    empty_char   - str, optional  :
      replaces '-' as the empty-bar character
    prefix      - str, optional  :
      message in front of the bar
    suffix      - str, optional  :
      message after the bar


    Returns
    -------
    formattedString - str:
      python string representing a progress bar
    """
    filledLength = int(round(bar_length * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), num_decimals)
    bar = fill_char * filledLength + empty_char * (bar_length - filledLength)
    # Here, the \r (carriage return) is what replaces the last line that was printed
    carriageReturn = end if iteration == total else '\r'
    formattedString = '%s [%s] %s%s %s%s' % (prefix, bar, percents, '%', suffix, carriageReturn)
    return formattedString

# Another hidden function for providing verbose progress output


def _build_verbose_iteration(iteration, total, prefix, suffix, end):
    digits = _num_digits(total)
    return '%s Iter %s of %s %s: %s' % (prefix, str(iteration + 1).zfill(digits), total, suffix, end)

#############################################################################################################
#                                    The VerbosityPrinter Class itself                                      #
#############################################################################################################


# The class responsible for optionally logging output
class VerbosityPrinter(object):
    """
    Class responsible for logging things to stdout or a file.

    Controls verbosity and can print progress bars. ex:

    >>> VerbosityPrinter(1)

    would construct a printer that printed out messages of level one or higher
    to the screen.

    >>> VerbosityPrinter(3, 'output.txt')

    would construct a printer that sends verbose output to a text file

    The static function :meth:`create_printer` will construct a printer from
    either an integer or an already existing printer.  it is a static method
    of the VerbosityPrinter class, so it is called like so:

    >>> VerbosityPrinter.create_printer(2)

    or

    >>> VerbostityPrinter.create_printer(VerbosityPrinter(3, 'output.txt'))

    ``printer.log('status')`` would log 'status' if the printers verbosity was
    one or higher. ``printer.log('status2', 2)`` would log 'status2' if the
    printer's verbosity was two or higher

    ``printer.error('something terrible happened')`` would ALWAYS log
    'something terrible happened'. ``printer.warning('something worrisome
    happened')`` would log if verbosity was one or higher - the same as a
    normal status.

    Both printer.error and printer.warning will prepend 'ERROR: ' or 'WARNING:
    ' to the message they are given.  Optionally, printer.log() can also
    prepend 'Status_n' to the message, where n is the message level.

    Logging of progress bars/iterations:

    >>> with printer_instance.progress_logging(verbosity):
    >>>     for i, item in enumerate(data):
    >>>         printer.show_progress(i, len(data))
    >>>         printer.log(...)

    will output either a progress bar or iteration statuses depending on the
    printer's verbosity

    Parameters
    ----------
    verbosity : int
        How verbose the printer should be.

    filename : str, optional
        Where to put output (If none, output goes to screen)

    comm : mpi4py.MPI.Comm or ResourceAllocation, optional
        Restricts output if the program is running in parallel  (By default,
        if the rank is 0, output is sent to screen, and otherwise sent to commfiles `1, 2, ...`

    warnings : bool, optional
        Whether or not to print warnings

    split : bool, optional
        Whether to split output between stdout and stderr as appropriate, or
        to combine the streams so everything is sent to stdout.

    clear_file : bool, optional
        Whether or not `filename` should be cleared (overwritten) or simply
        appended to.

    Attributes
    ----------
    _comm_path : str
        relative path where comm files (outputs of non-root ranks) are stored.

    _comm_file_name : str
        root filename for comm files (outputs of non-root ranks).

    _comm_file_ext : str
        filename extension for comm files (outputs of non-root ranks).
    """

    # Rules for handling comm --This is a global variable-- (technically) it should probably only be set once, at the
    # beginning of the program
    _comm_path = ''
    # The name of the generated files, e.g. 'comm_output'. '' means don't output to comm files.  Must also be set
    _comm_file_name = ''
    _comm_file_ext = '.txt'

    def _create_file(self, filename):
        with open(filename, 'w') as newFile:
            newFile.close()

    def _get_comm_file(self, comm_id):
        if len(VerbosityPrinter._comm_file_name) == 0: return ''
        return '%s%s%s%s' % (VerbosityPrinter._comm_path,
                             VerbosityPrinter._comm_file_name,
                             comm_id,
                             VerbosityPrinter._comm_file_ext)

    # The printer is initialized with a set verbosity, and an optional filename.
    # If a filename is not provided, VerbosityPrinter writes to stdout
    def __init__(self, verbosity=1, filename=None, comm=None, warnings=True, split=False, clear_file=True):
        '''
        Customize a verbosity printer object

        Parameters
        ----------
        verbosity : int, optional
            How verbose the printer should be.

        filename : str, optional
            Where to put output (If none, output goes to screen)

        comm : mpi4py.MPI.Comm or ResourceAllocation, optional
            Restricts output if the program is running in parallel  (By default,
            if the rank is 0, output is sent to screen, and otherwise sent to `commfiles 1, 2, ...`

        warnings : bool, optional
            Whether or not to print warnings
        '''
        if isinstance(comm, _ResourceAllocation): comm = comm.comm
        if comm:
            if comm.Get_rank() != 0 and not filename:  # A filename will override the default comm behavior
                filename = self._get_comm_file(comm.Get_rank())
        self.verbosity = verbosity
        self.filename = filename
        if filename is not None and len(filename) > 0 and clear_file:
            self._create_file(filename)
        self._comm = comm
        self.progressLevel = 0  # Used for queuing output while a progress bar is being shown
        self._delayQueue = []
        self._progressStack = []
        self._progressParamsStack = []
        self.warnings = warnings
        self.extra_indents = 0  # Used for nested calls
        self.defaultVerbosity = 1
        self.recorded_output = None
        self.split = split

    def clone(self):
        """
        Instead of deepcopy, initialize a new printer object and feed it some select deepcopied members

        Returns
        -------
        VerbosityPrinter
        """
        p = VerbosityPrinter(self.verbosity, self.filename, self._comm, self.warnings, self.split, clear_file=False)

        p.defaultVerbosity = self.defaultVerbosity
        p.progressLevel = self.progressLevel
        p.extra_indents = self.extra_indents
        p.recorded_output = self.recorded_output

        p._delayQueue = _dc(self._delayQueue)  # deepcopy
        p._progressStack = _dc(self._progressStack)
        p._progressParamsStack = _dc(self._progressParamsStack)
        return p

    # Function for converting between interfaces:
    # Accepts either a verbosity level (integer) or a pre-constructed VerbosityPrinter
    @staticmethod
    def create_printer(verbosity, comm=None):
        """
        Function for converting between interfaces

        Parameters
        ----------
        verbosity : int or VerbosityPrinter object, required:
            object to build a printer from

        comm : mpi4py.MPI.Comm object, optional
            Comm object to build printers with. !Will override!

        Returns
        -------
        VerbosityPrinter :
            The printer object, constructed from either an integer or another printer
        """
        if _compat.isint(verbosity):
            printer = VerbosityPrinter(verbosity, comm=comm)
        else:
            if isinstance(comm, _ResourceAllocation): comm = comm.comm
            printer = verbosity.clone()  # deepcopy the printer object if it has been passed as a verbosity
            printer._comm = comm  # override happens here
        return printer

    def __add__(self, other):
        '''
        Increase the verbosity of a VerbosityPrinter
        '''
        p = self.clone()
        p.verbosity += other
        p.extra_indents -= other
        return p

    def __sub__(self, other):
        '''
        Decrease the verbosity of a VerbosityPrinter
        '''
        p = self.clone()
        p.verbosity -= other
        p.extra_indents += other
        return p

    def __getstate__(self):
        #Return the state (for pickling) -- *don't* pickle Comm object
        to_pickle = self.__dict__.copy()
        del to_pickle['_comm']  # one *cannot* pickle Comm objects
        return to_pickle

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self._comm = None  # initialize to None upon unpickling

    # Used once a file has been created - open the file whenever a message needs to be sent (rather than opening it for
    # the entire program)
    def _append_to(self, filename, message):
        with open(filename, 'a') as output:
            output.write(message)  # + '\n')

    # Hidden function for deciding what to do with our output
    def _put(self, message, flush=True, stderr=False):
        if self.filename is None:  # Handles the case where comm is None or comm is rank 0
            if stderr:
                print(message, end='', file=_sys.stderr)
            else:
                print(message, end='')
        elif len(self.filename) > 0:
            if self.split:
                if stderr:
                    print(message, end='', file=_sys.stderr)
                else:
                    print(message, end='')
            self._append_to(self.filename, message)
        if flush:
            _sys.stdout.flush()

    # Hidden function for recording output to memory
    def _record(self, typ, level, message):
        if self.recorded_output is not None:
            global_level = level + self.extra_indents
            self.recorded_output.append((typ, global_level, message))

    # special function reserved for logging errors
    def error(self, message):
        """
        Log an error to the screen/file

        Parameters
        ----------
        message : str
            the error message

        Returns
        -------
        None
        """
        self._put('\nERROR: %s\n' % message, stderr=True)
        self._record("ERROR", 0, '\nERROR: %s\n' % message)

    # special function reserved for logging warnings

    def warning(self, message):
        """
        Log a warning to the screen/file if verbosity > 1

        Parameters
        ----------
        message : str
            the warning message

        Returns
        -------
        None
        """
        if self.warnings:
            self._put('\nWARNING: %s\n' % message, stderr=True)
            self._record("WARNING", 0, '\nWARNING: %s\n' % message)

    def log(self, message, message_level=None, indent_char='  ', show_statustype=False, do_indent=True,
            indent_offset=0, end='\n', flush=True):
        """
        Log a status message to screen/file.

        Determines whether the message should be printed based on current verbosity setting,
        then sends the message to the appropriate output

        Parameters
        ----------
        message : str
            the message to print (or log)

        message_level : int, optional
            the minimum verbosity level at which this level is printed.

        indent_char : str, optional
            what constitutes an "indent" (messages at higher levels are indented more
            when `do_indent=True`).

        show_statustype : bool, optional
            if True, prepend lines with "Status Level X" indicating the `message_level`.

        do_indent : bool, optional
            whether messages at higher message levels should be indented.  Note that if
            this is False it may be helpful to set `show_statustype=True`.

        indent_offset : int, optional
            an additional number of indentations to add, on top of any due to the
            message level.

        end : str, optional
            the character (or string) to end message lines with.

        flush : bool, optional
            whether stdout should be flushed right after this message is printed
            (this avoids delays in on-screen output due to buffering).

        Returns
        -------
        None
        """
        if message_level is None:
            message_level = self.defaultVerbosity
        if message_level <= self.verbosity:
            indent = (indent_char * (message_level - 1 + indent_offset
                                     + self.extra_indents)) if do_indent else ''
            # message_level-1 so no indent at verbosity == 1
            statusType = 'Status Level %s:' % message_level if show_statustype else ''
            if end == '\n':
                #Special case where we process a message containing newlines
                formattedMessage = '\n'.join(['%s%s%s' % (indent, statusType, m)
                                              for m in str(message).split('\n')]) + end
            else:
                formattedMessage = '%s%s%s%s' % (indent, statusType, message, end)

            if self.progressLevel > 0 and self.filename is None:
                self._delayQueue.append(indent_char + 'INVALID LEVEL: ' + formattedMessage)
            else:
                self._put(formattedMessage, flush=flush)
                self._record("LOG", message_level, formattedMessage)

    def _progress_bar(self, iteration, total, bar_length, num_decimals, fill_char, empty_char, prefix, suffix, indent):
        progressBar = ''
        # 'self.progressLevel == 1' disallows nested progress bars !!!
        unnested = self.progressLevel == 1
        if unnested:
            progressBar = _build_progress_bar(iteration, total, bar_length, num_decimals,
                                              fill_char, empty_char, prefix, suffix)
            progressBar = indent + progressBar
        return progressBar

    def _verbose_iteration(self, iteration, total, prefix, suffix, verbose_messages, indent, end):
        iteration = _build_verbose_iteration(iteration, total, prefix, suffix, end)
        iteration = indent + iteration
        for verboseMessage in verbose_messages:
            iteration += (indent + verboseMessage + '\n')
        return iteration

    def __str__(self):
        return 'Printer Object: Progress Level: %s Verbosity %s Indents %s' \
            % (self.progressLevel, self.verbosity, self.extra_indents)

    @_contextmanager
    def verbosity_env(self, level):
        """
        Create a temporary environment with a different verbosity level.

        This is context manager, controlled using Python's with statement:

            >>> with printer.verbosity_env(2):
                    printer.log('Message1') # printed at verbosity level 2
                    printer.log('Message2') # printed at verbosity level 2

        Parameters
        ----------
        level : int
            the verbosity level of the environment.
        """
        original = self.defaultVerbosity
        try:
            self.defaultVerbosity = level
            yield
        finally:
            self.defaultVerbosity = original

    @_contextmanager
    def progress_logging(self, message_level=1):
        """
        Context manager for logging progress bars/iterations.

        (The printer will return to its normal, unrestricted state when the progress logging has finished)

        Parameters
        ----------
        message_level : int, optional
            progress messages will not be shown until the verbosity level reaches `message_level`.
        """
        try:
            self._progressStack.append(message_level)
            self._progressParamsStack.append(None)
            if self.verbosity == message_level:
                self.progressLevel += 1
            yield
        finally:
            self._end_progress()

    # A wrapper for show_progress that only works if verbosity is above a certain value (Status by default)
    def show_progress(self, iteration, total, bar_length=50, num_decimals=2, fill_char='#',
                      empty_char='-', prefix='Progress:', suffix='', verbose_messages=None, indent_char='  ', end='\n'):
        """
        Displays a progress message (to be used within a `progress_logging` block).

        Parameters
        ----------
        iteration : int
            the 0-based current iteration -- the interation number this message is for.

        total : int
            the total number of iterations expected.

        bar_length : int, optional
            the length, in characters, of a text-format progress bar (only used when the
            verbosity level is exactly equal to the `progress_logging` message level.

        num_decimals : int, optional
            number of places after the decimal point that are displayed in progress
            bar's percentage complete.

        fill_char : str, optional
            replaces '#' as the bar-filling character

        empty_char : str, optional
            replaces '-' as the empty-bar character

        prefix : str, optional
            message in front of the bar

        suffix : str, optional
            message after the bar

        verbose_messages : list, optional
            A list of strings to display after an initial "Iter X of Y" line when
            the verbosity level is higher than the `progress_logging` message level
            and so more verbose messages are shown (and a progress bar is not).  The
            elements of `verbose_messages` will occur, one per line, after the initial
            "Iter X of Y" line.

        indent_char : str, optional
            what constitutes an "indentation".

        end : str, optional
            the character (or string) to end message lines with.

        Returns
        -------
        None
        """
        if verbose_messages is None:
            verbose_messages = []
        indent = indent_char * (self._progressStack[-1] - 1 + self.extra_indents)
        # -1 so no indent at verbosity == 1

        # Print a standard progress bar if its verbosity matches ours,
        # Otherwise, Print verbose iterations if our verbosity is higher
        # Build either the progress bar or the verbose iteration status
        progress = ''
        if self.verbosity == self._progressStack[-1] and self.filename is None:
            progress = self._progress_bar(iteration, total, bar_length, num_decimals,
                                          fill_char, empty_char, prefix, suffix, indent)
            self._progressParamsStack[-1] = (iteration, total, bar_length,
                                             num_decimals, fill_char, empty_char,
                                             prefix, suffix, indent)
        elif self.verbosity > self._progressStack[-1]:
            progress = self._verbose_iteration(iteration, total, prefix, suffix, verbose_messages, indent, end)
            self._record("LOG", self._progressStack[-1], progress)

        self._put(progress)  # send the progress logging to either file or stdout

    # must be explicitly called when the progress (e.g. loop) is done:
    #  This allows for early exits
    def _end_progress(self):
        if self.progressLevel > 0:
            last_progress_params = self._progressParamsStack.pop()
            if self.verbosity == self._progressStack.pop():
                if last_progress_params is not None:
                    (iteration, total, barLength,
                     numDecimals, fillChar, emptyChar,
                     prefix, suffix, indent) = last_progress_params
                    progress = self._progress_bar(iteration + 1, total, barLength,
                                                  numDecimals, fillChar, emptyChar,
                                                  prefix, suffix, indent)
                    self._put(progress)  # send the progress logging to either file or stdout
                    self._record("LOG", self.verbosity, progress)

                # Show the statuses that were queued while the progressBar was active
                for item in self._delayQueue:
                    print(item)
                del self._delayQueue[:]
                self.progressLevel -= 1

    def start_recording(self):
        """
        Begins recording the output (to memory).

        Begins recording (in memory) a list of `(type, verbosityLevel, message)`
        tuples that is returned by the next call to :meth:`stop_recording`.

        Returns
        -------
        None
        """
        self.recorded_output = []

    def is_recording(self):
        """
        Returns whether this VerbosityPrinter is currently recording.

        Returns
        -------
        bool
        """
        return bool(self.recorded_output is not None)

    def stop_recording(self):
        """
        Stops recording and returns recorded output.

        Stops a "recording" started by :meth:`start_recording` and returns the
        list of `(type, verbosityLevel, message)` tuples that have been recorded
        since then.

        Returns
        -------
        list
        """
        recorded = self.recorded_output
        self.recorded_output = None  # always "stop" recording
        return recorded


########################################################################################################################
#                                Demonstration of how the VerbosityPrinter class is used                               #
########################################################################################################################

# Some basic demonstrations of how to use the printer class with an arbitrary function

'''
if __name__ == "__main__":
    import threading
    import time

    def demo(verbosity):
        # usage of the show_progress function
        printer = VerbosityPrinter.create_printer(verbosity)
        data    = range(10)
        with printer.progress_logging(2):
            for i, item in enumerate(data):
                printer.show_progress(i, len(data)-1,
                          verbose_messages=['%s gates' % i], prefix='--- GST (', suffix=') ---')
                time.sleep(.05)

    def nested_demo(verbosity):
        printer = VerbosityPrinter.create_printer(verbosity)
        printer.warning('Beginning demonstration of the verbosityprinter class. This could go wrong..')
        data    = range(10)
        with printer.progress_logging(1):
            for i, item in enumerate(data):
                printer.show_progress(i, len(data)-1,
                      verbose_messages=['%s circuits' % i], prefix='-- IterativeGST (', suffix=') --')
                if i == 5:
                    printer.error('The iterator is five. This caused an error, apparently')
                demo(printer - 1)

    print('\nTersest: \n')
    nested_demo(0)

    print('\nTerse: \n')
    nested_demo(1)

    print('\nStandard: \n')
    nested_demo(2)

    print('\nVerbose: \n')
    nested_demo(3)

    print('\nMost Verbose: \n')
    nested_demo(4)
    # Create four threads of different verbosities, each of which write output to their own file

    threads = []
    for i in range(4):
        # Each thread is started with a printer that is assigned a different verbosity
        printer = VerbosityPrinter(i, 'output%s.txt' % i)
        t = threading.Thread(target=demo, args = [printer])
        t.daemon = True
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()
'''
