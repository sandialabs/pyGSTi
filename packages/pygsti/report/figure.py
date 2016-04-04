import pickle as _pickle
import matplotlib.pyplot as _plt


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
        self.pickledAxes = _pickle.dumps(axes)
        self.extraInfo = extraInfo

    def save_to(self, filename):
        if filename is not None and len(filename) > 0:
            try:
                axes = _pickle.loads(self.pickledAxes) 
                  #this creates a new (current) figure in matplotlib
                curFig = _plt.gcf() # gcf == "get current figure"
                curFig.callbacks.callbacks = {} 
                  # initialize fig's CallbackRegistry, which doesn't
                  # unpickle properly in matplotlib 1.5.1 (bug?)
            except:
                raise ValueError("ReportFigure unpickling error!  This " +
                                 "could be caused by using matplotlib or " +
                                 "pylab magic functions ('%pylab inline' or " +
                                 "'%matplotlib inline') within an iPython " +
                                 "notebook, so if you used either of these " +
                                 "please remove it and all should be well.")
            _plt.savefig(filename, bbox_extra_artists=(axes,),
                         bbox_inches='tight') #need extra artists otherwise
                                              #axis labels get clipped
            _plt.close(curFig) # closes the figure created by unpickling

    def set_extra_info(self, extraInfo):
        self.extraInfo = extraInfo

    def get_extra_info(self):
        return self.extraInfo
    
    def check(self):
        axes = _pickle.loads(self.pickledAxes) 
          #this creates a new (current) figure in matplotlib
        curFig = _plt.gcf() # gcf == "get current figure"
        curFig.callbacks.callbacks = {} # initialize fig's CallbackRegistry...
        _plt.close(curFig) # closes the figure created by unpickling
