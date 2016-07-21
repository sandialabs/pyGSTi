from __future__ import division, print_function, absolute_import, unicode_literals
from   copy        import deepcopy as _dc
from contextlib    import contextmanager as _contextmanager
import sys         as _sys
import math        as _math # used for digit formatting

def _num_digits(n):
    return int(_math.log10(n)) + 1 if n > 0 else 1

# This function isn't a part of the public interface, instead it has a wrapper in the VerbosityPrinter class
def _build_progress_bar (iteration, total, barLength = 100, numDecimals=2, fillChar='#',
                    emptyChar='-', prefix='Progress:', suffix='Complete', end='\n'):
    """
    Used for building a progress bar as a string
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        barLength   - Optional  : character length of bar (Int)
        numDecimals - Optional  : precision of progress percent (Int)
        fillChar    - Optional  : replaces '#' as the bar-filling character (Str)
        emptyChar   - Optional  : replaces '-' as the empty-bar character (Str)
        prefix      - Optional  : message in front of the bar
        suffix      - Optional  : message after the bar
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), numDecimals)
    bar             = fillChar * filledLength + emptyChar * (barLength - filledLength)
    # Here, the \r (carriage return) is what replaces the last line that was printed
    carriageReturn  = end if iteration == total else '\r'
    formattedString = '%s [%s] %s%s %s%s' % (prefix, bar, percents, '%', suffix, carriageReturn)
    return formattedString

# Another hidden function for providing verbose progress output
def _build_verbose_iteration(iteration, total, prefix, suffix, end):
    digits = _num_digits(total)
    return '%s Iter %s of %s %s: %s' % (prefix, str(iteration).zfill(digits), total, suffix, end)

#############################################################################################################
#                                    The VerbosityPrinter Class itself                                      #
#############################################################################################################


# The class responsible for optionally logging output
class VerbosityPrinter():
    '''
    Class responsible for logging things to stdout or a file. Controls verbosity and can print progress bars
    ex: VerbosityPrinter(1) would construct a printer that printed out messages of level one or higher to the screen
        VerbosityPrinter(3, 'output.txt') would construct a printer that sends verbose output to a text file

        the static function 'build_printer' will construct a printer from either an integer or an already existing printer.
        it is a staticmethod of the VerbosityPrinter class, so it is called like so:
            VerbosityPrinter.build_printer(2) or VerbostityPrinter.build_printer(VerbosityPrinter(3, 'output.txt'))

      printer.log('status')     would log 'status' if the printers verbosity was one or higher
      printer.log('status2', 2) would log 'status2' if the printer's verbosity was two or higher

      printer.error('something terrible happened') would ALWAYS log 'something terrible happened'
      printer.warning('something worrisome happened') would log if verbosity was one or higher - the same as a normal status

      Both printer.error and printer.warning will prepend 'ERROR: ' or 'WARNING: ' to the message they are given.
      Optionally, printer.log() can also prepend 'Status_n' to the message, where n is the message level.

      Logging of progress bars/iterations:

      with printer_instance.progress_logging(verbosity):
        for i, item in enumerate(data):
          printer.show_progress(i, len(data))
          printer.log(...)

      Will output either a progress bar or iteration statuses depending on the printer's verbosity
    '''

    # Rules for handling comm --This is a global variable-- (technically) it should probably only be set once, at the beginning of the program
    _commPath     = ''
    _commFileName = 'comm_output' # The name of the generated files. Must also be set
    _commFileExt  = '.txt'

    def _create_file(self, filename):
        with open(filename, 'w') as newFile:
            newFile.close()

    def _get_comm_file(self, comm_id):
        return '%s%s%s%s' % (VerbosityPrinter._commPath, VerbosityPrinter._commFileName, comm_id, VerbosityPrinter._commFileExt)

    # The printer is initialized with a set verbosity, and an optional filename.
    # If a filename is not provided, VerbosityPrinter writes to stdout
    def __init__(self, verbosity, filename=None, comm=None):
        if comm != None:
            if comm.Get_rank() != 0 and filename == None: # A filename will override the default comm behavior
                filename = self._get_comm_file(comm.Get_rank())
        self.verbosity = verbosity
        self.filename  = filename
        if filename != None:
            self._create_file(filename)
        self._comm            = comm
        self.progressLevel    = 0 # Used for queuing output while a progress bar is being shown
        self._delayQueue      = []
        self._progressStack   = []

    # Alias deepcopy, convert it to a method
    def clone(self):
        return _dc(self)

    # Function for converting between interfaces:
    # Accepts either a verbosity level (integer) or a pre-constructed VerbosityPrinter
    @staticmethod
    def build_printer(verbosity, comm=None):
        if type(verbosity) == int:  #isinstance(verbosity,int)
            return VerbosityPrinter(verbosity, comm=comm)
        else:
            return verbosity.clone() # deepcopy the printer object if it has been passed as a verbosity

    # Basic rules for increasing or decreasing verbosity with arithmetic operators
    def __add__(self, other):
        p = self.clone()
        p.verbosity += other
        return p

    def __sub__(self, other):
        p = self.clone()
        p.verbosity -= other
        return p

    # Helper Function for safely appending a message to a file
    def _append_to(self, filename, message):
        with open(filename, 'a') as output:
            output.write(message + '\n')

    # Hidden function for deciding what to do with our output
    def _put(self, message, flush=True):
        if self.filename == None: # handles the case where comm exists, as comm will create a filename
            if self._comm == None or self._comm.Get_rank() == 0:
                print(message, end='')
                if flush:
                    _sys.stdout.flush()
        else:
            self._append_to(self.filename, message)

    # special function reserved for logging errors
    def error(self, message):
        self._put('\nERROR: %s\n' % message)

    # special function reserved for logging warnings
    def warning(self, message):
        if self.verbosity > 0:
            self._put('\nWARNING: %s\n' % message)

    # Function for sending messages to the desired output
    # Determines whether the message should be printed based on current verbosity setting,
    # then sends the message to the appropriate output
    def log(self, message, messageLevel=1, indentChar='  ', showStatustype=False, doIndent=True, indentOffset=0, end='\n', flush=True):
        if messageLevel <= self.verbosity:
            indent = (indentChar * (messageLevel+indentOffset)) if doIndent else ''
            statusType = 'Status Level %s:' % messageLevel if showStatustype else ''
            formattedMessage = '%s%s%s%s' % (indent, statusType, message, end)

            if self.progressLevel > 0 and self.filename == None:
                self._delayQueue.append(indentChar + 'INVALID LEVEL: ' + formattedMessage)
            else:
                self._put(formattedMessage, flush=flush)

    def _progress_bar(self, iteration, total, barLength, numDecimals, fillChar, emptyChar, prefix, suffix, indent):
        progressBar = ''
        # 'self.progressLevel == 1' disallows nested progress bars !!!
        unnested = self.progressLevel == 1
        if unnested:
            progressBar =  _build_progress_bar(iteration, total, barLength, numDecimals,
                                                         fillChar, emptyChar, prefix, suffix)
            progressBar = indent + progressBar
        return progressBar

    def _verbose_iteration(self, iteration, total, prefix, suffix, verboseMessages, indent, end):
        iteration =  _build_verbose_iteration(iteration, total, prefix, suffix, end)
        iteration = indent + iteration
        for verboseMessage in verboseMessages:
            iteration += (indent + verboseMessage + '\n')
        return iteration

    def __str__(self):
        return 'Printer Object: Progress Level: %s Verbosity %s' % (self.progressLevel, self.verbosity)

    @_contextmanager
    def progress_logging(self, messageLevel):
        self._progressStack.append(messageLevel)
        if self.verbosity == messageLevel:
            self.progressLevel += 1
        yield
        self._end_progress()

    # A wrapper for show_progress that only works if verbosity is above a certain value (Status by default)
    def show_progress(self, iteration, total, messageLevel=1, barLength = 50, numDecimals=2, fillChar='#',
                    emptyChar='-', prefix='Progress:', suffix='', verboseMessages=[], indentChar='  ', end='\n'):

        """
        Used for building a progress bar as a string
        @params:
        iteration       - Required  : current iteration (Int)
        total           - Required  : total iterations (Int)
        messageLevel    - Optional  : verbosity level at which the bar is printed (Int)
        barLength       - Optional  : character length of bar (Int)
        numDecimals     - Optional  : precision of progress percent (Int)
        fillChar        - Optional  : replaces '#' as the bar-filling character (Str)
        emptyChar       - Optional  : replaces '-' as the empty-bar character (Str)
        prefix          - Optional  : message in front of the bar
        suffix          - Optional  : message after the bar
        verboseMessages - Optional  : list of messages that are printed out at the same indentation as the progress bar
        indentChar      - Optional  : sets the level of indentation of the bar
        """
        indent = indentChar * self._progressStack[-1]

        # Print a standard progress bar if its verbosity matches ours,
        # Otherwise, Print verbose iterations if our verbosity is higher
        # Build either the progress bar or the verbose iteration status
        progress = ''
        if self.verbosity == self._progressStack[-1] and self.filename == None:
            progress = self._progress_bar(iteration, total, barLength, numDecimals, fillChar, emptyChar, prefix, suffix, indent)
        elif self.verbosity >  self._progressStack[-1]:
            progress = self._verbose_iteration(iteration, total, prefix, suffix, verboseMessages, indent, end)

        self._put(progress) # send the progress logging to either file or stdout

    # must be explicitly called when the progress (e.g. loop) is done:
    #  This allows for early exits
    def _end_progress(self):
        if self.progressLevel > 0:
            if self.verbosity == self._progressStack.pop():
                # Show the statuses that were queued while the progressBar was active
                for item in self._delayQueue:
                    print(item)
                del self._delayQueue[:]
                self.progressLevel -= 1


########################################################################################################################################
#                                Demonstration of how the VerbosityPrinter class is used                                               #
########################################################################################################################################

# Some basic demonstrations of how to use the printer class with an arbitrary function

'''
# Some basic demonstrations of how to use the printer class with an arbitrary function
if __name__ == "__main__":
    import threading
    import time

    def demo(verbosity):
    # usage of the show_progress function
    printer = VerbosityPrinter.build_printer(verbosity)
    data    = range(10)
        with printer.progress_logging(2):
        for i, item in enumerate(data):
            printer.show_progress(i, len(data)-1, messageLevel=2,
                      verboseMessages=['%s gates' % i], prefix='--- GST (', suffix=') ---')
            time.sleep(.05)

    def nested_demo(verbosity):
        printer = VerbosityPrinter.build_printer(verbosity)
        printer.warning('Beginning demonstration of the verbosityprinter class. This could go wrong..')
    data    = range(10)
        with printer.progress_logging(1):
        for i, item in enumerate(data):
            printer.show_progress(i, len(data)-1, messageLevel=1,
                  verboseMessages=['%s gate strings' % i], prefix='-- IterativeGST (', suffix=') --')
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
