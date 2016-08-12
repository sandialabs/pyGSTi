from __future__ import division, print_function, absolute_import, unicode_literals
import pickle            as _pickle
import matplotlib.pyplot as _plt
import tempfile          as _tempfile
import shutil            as _shutil


class ReportFigure(object):
    """
    Encapsulates a single report figure.

    Essentially a matplotlib figure, but with better persistence.  In
    particular, the figure can be held in memory or on disk and saved
    in different formats independently of the state of the "matplotlib
    cloud".
    """

    def __init__(self, axes, extraInfo=None):
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
        # This file will be implicitly closed when garbage collected
        self.tempSave = _tempfile.NamedTemporaryFile()

        curFig = _plt.gcf() # gcf == "get current figure"
        curFig.callbacks.callbacks = {}
        _plt.savefig(self.tempSave, bbox_extra_artists=(axes,),
                     bbox_inches='tight') #need extra artists otherwise
                                          #axis labels get clipped
        _plt.close(curFig) # closes the figure

        self.extraInfo = extraInfo

    def save_to(self, filename):
        if filename is not None and len(filename) > 0:
            _shutil.copy2(self.tempSave.name, filename)

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
