"""Defines calculation "representation" objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

# This is just a "loader" module which either loads the (fast)
# Cython/C++ version or the (slower) Python version of the 
# collection of objects and functions termed a "representation library".

# The bit of logic involved is that we want to display a slowness
# warning only when the user actually *uses* the slower python
# routines, and not immediately upon importing the python version.
# To acheive this, instead of importing '*' directly, we decorate
# the functions as they are imported.  

#Maybe in the FUTURE we'll remove the decorator action once the warning
#  has been issued?

try:
    #If we can import the fast stuff, great!
    from .fastreplib import *

except ImportError:

    #FUTURE: Could try this as a secondary option (but creates side-effects...)
    #import pyximport
    #pyximport.install(setup_args={'include_dirs': _np.get_include()})
    #from .fastreplib import *
    
    import os as _os
    import warnings as _warnings

    val = _os.environ.get('PYGSTI_SLOW_WARNING',None)

    if val in ("0","False","FALSE","false","No","no","NO"):
        #No warnings desired - just import the slow version
        from .slowreplib import *
    elif val in ("1","True","TRUE","true","Yes","yes","YES"):
        #Warning directly upon import
        _warnings.warn(
                ("\n\nFalling back to slower pure-python routines b/c Cython extensions are unavailable.\n"
                 "You haven't necessarily used any slow routines yet - you're seeing this\n"
                 "warning because you've set the environment variable \"PYGSTI_SLOW_WARNING=1\"\n"))
        from .slowreplib import *
    else:
        #Import the slow stuff, decorated so we display a warning when
        # any of it is used.
        import sys as _sys
        import inspect as _inspect
        import functools as _functools
        this_module = _sys.modules[__name__]
        
        def decorate(fn):
            @_functools.wraps(fn)
            def new_fn(*args, **kwargs):
                _warnings.warn(
                    ("\nUsing slower pure-python routines because Cython extensions are unavailable.\n"
                     "This is OK, but your code will take longer to run.  If you want the faster\n"
                     "version, try (re-)installing via pip after you have Cython:\n"
                     "\n"
                     "    pip install cython    # get Cython\n" 
                     "    pip install pygsti    # if you use pygsti as an installed library OR \n"
                     "    pip install -e .      # if you use a local cloned tree (from the root pyGSTi/ dir)\n"
                     "\n"
                     "Instead of the final line above, you can also run:\n"
                     "\n"
                     "python setup.py build_ext --inplace\n"
                     "\n"
                     "from your local pyGSTi root directory to build the Cython extensions in your\n"
                     "local tree.  Finally, if you don't care about pyGSTi being slower and just\n"
                     "want to be rid of this message, you can set the environment variable:\n"
                     "\"PYGSTI_SLOW_WARNING=0\" to disable this message, OR\n"
                     "\"PYGSTI_SLOW_WARNING=1\" to enable a much shorter message that is always\n"
                     "  displayed (to remind you to build the extensions sometime later)"))
                return fn(*args, **kwargs)
            return new_fn
        
        from . import slowreplib
        for name,val in _inspect.getmembers(slowreplib):
            if name.startswith("_"): continue # don't process anything 'hidden'
            if _inspect.isfunction(val):
                val = decorate(val)
                setattr(this_module, name, val) # imports name to this module
            elif _inspect.isclass(val):
                val.__init__ = decorate(val.__init__)
                setattr(this_module, name, val) # imports name to this module
