

class GaugeOptMethodBase(object):
    def setUp(self):
        super(GaugeOptMethodBase, self).setUp()
        self.options = dict(
            verbosity=0,
            check_jac=False,
            tol = 1e-5
        )