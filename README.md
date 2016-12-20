********************************************************************************
  pyGSTi 0.9 
********************************************************************************

[![Build Status](https://travis-ci.org/pyGSTio/pyGSTi.svg?branch=beta)](https://travis-ci.org/pyGSTio/pyGSTi)

Overview:
--------
This is the root directory of the pyGSTi software, which is a Python
 implementation of Gate Set Tomography.  pyGSTi free software, licensed
 under the GNU General Public License (GPL).  Copyright and license information
 can be found in ``license.txt``, and the GPL itself in ``COPYING``.


Getting Started:
---------------
pyGSTi is written entirely in Python, and so there's no compiling necessary.
The first step is to install Python (and
[Jupyter notebook](http://jupyter.org/) is highly recommended) if you haven't
already.  In order to use pyGSTi you need to tell your Python distribution
where pyGSTi lives, which can be done in one of two ways:

* User-Only Installation

     To install pyGSTi for the current user, run ``python install_locally.py``

    This adds the current pyGSTi path to Python's list of search paths, and
    doesn't require administrative access.  Typically you want to do this if
    you've cloned the pyGSTi GitHub repository and want any changes you make to
    your local file to have effect when you ``import pygsti`` from Python.
    You'd also want to use this option if you'd like long-term access the
    tutorial notebook files in the ``jupyter_notebooks`` directory under this
    one, since this means you'll be keeping this directory around anyway.

* System-Wide Installation

  To install pyGSTi for all users, run: ``python setup.py install``

  This installs the pyGSTi Python package into one of the Python distribution's
  search directories.  **This typically requires administrative privileges**,
  and is the way most Python packages are installed.  Installing this way has
  the advantage that it makes the package available to all users and then
  allows you to move or delete the directory you're installing from.  If you
  don't use this method **you must not delete this directory** so long as you
  want to use pyGSTi.

  Reasons you may **not** want to use this installation method are 
  
  - pyGSTi comes with (Jupyter notebook) tutorials that you may want to
    access for weeks and years to come (i.e. you plan to *keep* this
    pyGSTi directory around for a while).
  - you've cloned the pyGSTi repository and want this local set of files
    to be the one Python uses when you ``import pygsti``.

Using pyGSTi
----------

After you've installed pyGSTi, you should be able to import the 
`pygsti` Python package.  The next thing to do is take a look at
the tutorials, which you do by:

* Changing to the notebook directory, by running:
    ``cd jupyter_notebooks/Tutorials/``

* Start up the Jupyter notebook server by running:
  ``jupyter notebook``

The Jupyter server should open up your web browser to the server root, from
where you can start the first "00" tutorial notebook.  (Note that the key
command to execute a cell within the Jupyter notebook is ``Shift+Enter``, not
just ``Enter``.)

Congratulations!  You're up and running with pyGSTi!


Documentation
-------------
Online documentation is hosted on [Read the Docs](http://pygsti.readthedocs.io).
Instructions for building the documentation locally are contained in the file
`doc/build/howToBuild.txt`.




Questions?
----------
For help and support with pyGSTi, please contact the authors at
pygsti@sandia.gov.





Brief description of the directory structure:
--------------------------------------------
```
doc/  :  Directory containing the HTML documentation and other
         reference documents.

jupyter_notebooks/  : Parent directory of all Jupyter notebook tutorial
		      files included with pyGSTi.  It is convenient to
		      start an Jupyter notebook server from within this
		      directory.

packages/  :  Directory containing the (one) package provided by pyGSTi.
	             All of the Python source code lies here.

tests/  :  Directory containing pyGSTi unit tests.

install_locally.py :  A Python script that sets up the software to run 
                      from its current location within the file system.  This
                      essentially adds the packages directory to the list of
                      directories Python uses to search for packages in, so
                      that ``import pygsti`` will work for any Python script,
                      regardless of its location in the file system.  This
                      action is local to the user running the script, and does
                      not copy any files to any system directories, making it a
                      good option for a user without administrative access.

COPYING  :  A text version of the GNU General Public License.

license.txt  :  A text file giving copyright and licensing information
	        about pyGSTi.
```
