from .formatter_helpers import *
from .parameterized_formatter import ParameterizedFormatter
 
# Formatter class that requires a scratchDirectory from an instance of FormatSet for saving figures to
class FigureFormatter(ParameterizedFormatter):
    '''
    Helper class that utilizes a scratchDir variable to render figures
    '''
    def __init__(self, extension='.png', formatstring='%s%s%s%s'):
        '''
        Parameters
        ---------
        extension : string, optional. extension of the figure's image
        formatstring : string, optional. Normally formatted with W, H, scratchDir, filename
        '''
        super(FigureFormatter, self).__init__(no_format, ['scratchDir'])
        self.extension    = extension
        self.formatstring = formatstring

    # Override call method of Parameterized formatter
    def __call__(self, figInfo, specs):
        fig, name, W, H = figInfo
        scratchDir = self.specs['scratchDir']
        if len(scratchDir) > 0: #empty scratchDir signals not to output figure
            fig.save_to(_os.path.join(scratchDir, name + self.extension))
        return self.formatstring % (W, H, scratchDir,
                                    name + self.extension)
