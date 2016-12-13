from __future__ import division, print_function, absolute_import, unicode_literals
import pickle as _pickle
import matplotlib
import matplotlib.pyplot as _plt
import tempfile          as _tempfile
import shutil            as _shutil

from contextlib import contextmanager

import os

@contextmanager
def temp_backend(backend):
    oldBackend = matplotlib.get_backend()
    _plt.switch_backend(backend)
    yield # User code goes here
    _plt.switch_backend(oldBackend)

class ReportFigure(object):
    """
    Encapsulates a single report figure.

    Essentially a matplotlib figure, but with better persistence.  In
    particular, the figure can be held in memory or on disk and saved
    in different formats independently of the state of the "matplotlib
    cloud".
    """

    def __init__(self, axes, extraInfo=None, imageformats=['png', 'pdf']):
        """
        Create a new ReportFigure.

        Parameters
        ----------
        axes : Axes
           Matplotlib axes of the figure.

        extraInfo : object
           Any extra information you want to
           associate with this figure.
        """

        self.extraInfo   = extraInfo
        self.pickledAxes = _pickle.dumps(axes)

        self.tempFileDict = { }

        for fmt in imageformats:
            tf = _tempfile.TemporaryFile()
            _plt.savefig(tf, format=fmt, bbox_extra_artists=(axes,),
                         bbox_inches='tight') #need extra artists otherwise
            tf.seek(0) #rewind to beginning of file
            self.tempFileDict[fmt] = bytearray(tf.read())
            tf.close()


    def _save_axes(self, filename):
        axes = _pickle.loads(self.pickledAxes)
          #this creates a new (current) figure in matplotlib
        curFig = _plt.gcf() # gcf == "get current figure"
        curFig.callbacks.callbacks = {}
          # initialize fig's CallbackRegistry, which doesn't
          # unpickle properly in matplotlib 1.5.1 (bug?)
        _plt.savefig(filename, bbox_extra_artists=(axes,),
                     bbox_inches='tight') #need extra artists otherwise
                                          #axis labels get clipped
        _plt.close(curFig) # closes the figure created by unpickling

    def save_to(self, filename):
        if filename is not None and len(filename) > 0:
            try:
                self._save_axes(filename)
            except Exception as e:
                print('Initial unpickling of figure failed, trying to use cached formats')
                ext = os.path.splitext(filename)[1][1:] # remove .
                print(ext)
                if ext in self.tempFileDict:
                    tf = _tempfile.NamedTemporaryFile()
                    tf.write(self.tempFileDict[ext])
                    _shutil.copy2(tf.name, filename)
                    tf.close()
                else:
                    print('Extension not in cached files and unpickling failed. Trying experimental backend switching. Your machine may catch fire..')
                    # Subprocess didn't work without auxillary file (cannot import parent module '') -> we weren't desperate enough to do the auxillary file, so no promises that works either
                    # multiprocessing didn't work (it used the same matplotlib import/backend :( )
                    # tempfile with jpeg didn't work (we needed diverse formats)
                    try:
                        with temp_backend('agg'):
                            self._save_axes(filename)
                    except:
                        raise ValueError("ReportFigure unpickling error!  This " +
                                         "could be caused by using matplotlib or " +
                                         "pylab magic functions ('%pylab inline' or " +
                                         "'%matplotlib inline') within an iPython " +
                                         "notebook, so if you used either of these " +
                                         "please remove it and all should be well.")

    def set_extra_info(self, extraInfo):
        self.extraInfo = extraInfo

    def get_extra_info(self):
        return self.extraInfo

    def check(self):
        pass
        #axes = _pickle.loads(self.pickledAxes) #pylint: disable=unused-variable
          #this creates a new (current) figure in matplotlib
        #curFig = _plt.gcf() # gcf == "get current figure"
        #curFig.callbacks.callbacks = {} # initialize fig's CallbackRegistry...
        #_plt.close(curFig) # closes the figure created by unpickling
