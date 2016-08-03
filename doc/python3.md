# Python 3

Currently, the repository is compatible with both python 2.7 and 3.5+ through the use of `from __future__` statements, which brings python 3 style printing, importing, unicode literals, and division into python 2.7
  
If the code is being ported entirely to python 3 (or higher), these `from __future` statements can be removed
However, there are some deprecations between python2.7 and 3 that need to be addressed
For example, running the command `./lint.py deprecated-method` will show that there is currently one method (get_arg_spec, in `formatter.py`) that is set to be deprecated. However, the function doesn't exist in python 3, but its replacement doesn't exist in 2.7! So, if updating entirely to 3, this can be fixed. 
  
Running the full test suite will also indicate that some behaviors have been deprecated with other modules. For example, numpy arrays cannot be accessed by a floating point number, and the old style of opening files without a context manager will generally result in a warning. These can be silenced, but It's probably best to change them!
