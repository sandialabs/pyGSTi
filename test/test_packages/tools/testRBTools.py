from ..testutils import BaseTestCase, compare_files, temp_files
import pygsti
import unittest

class RBBaseTestCase(BaseTestCase):
    def test_rb_tools(self):
        #DEPRECATED since addition of extras/rb
        #ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        #val = pygsti.rb_decay(0.1,0.1,0.1)
        #self.assertAlmostEqual(val, 0.1039800665)
        #decay = pygsti.rb_decay_rate(ds,showPlot=False,xlim=(0,10),ylim=(0,10),
        #                             saveFigPath=temp_files + "/RBdecay.png")
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
