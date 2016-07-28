from ..testutils import BaseTestCase
import pygsti
import unittest

class RBBaseTestCase(BaseTestCase):
    def test_rb_tools(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="../cmp_chk_files/analysis.dataset")
        val = pygsti.rb_decay(0.1,0.1,0.1)
        self.assertAlmostEqual(val, 0.1039800665)
        decay = pygsti.rb_decay_rate(ds,showPlot=False,xlim=(0,10),ylim=(0,10),
                                     saveFigPath="../temp_test_files/RBdecay.png")

if __name__ == '__main__':
    unittest.main(verbosity=2)
