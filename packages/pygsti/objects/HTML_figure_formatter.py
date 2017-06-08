from .formatting_helpers import *
from .parameterized_formatter import _ParameterizedFormatter

#Special formatter added for now for new HTML reports
class _HTMLFigureFormatter(_ParameterizedFormatter):
    '''
    Helper class that utilizes a scratchDir variable to render figures
    '''
    def __init__(self):
        '''
        Create a new HTMLFigureFormatter
        '''
        super(_HTMLFigureFormatter, self).__init__(no_format, ['resizable','autosize'])

    # Override call method of Parameterized formatter
    def __call__(self, fig):
        render_out = fig.render("html",
                                resizable="handlers only" if self.specs['resizable'] else False,
                                autosize=self.specs['autosize'])
        return render_out #a dictionary with 'html' and 'js' keys
        #return "<script>\n %(js)s \n</script>\n" % render_out + \
        #    "%(html)s" % render_out
    #OLD: <div class='relwrap'><div class='abswrap'> </div></div>
