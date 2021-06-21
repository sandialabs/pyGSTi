import pygsti.models as models
from pygsti.algorithms import grammatrix as gm
from pygsti.data import DataSet
from ..util import BaseCase


class DataSetInstance(object):
    def setUp(self):
        super(DataSetInstance, self).setUp()
        self.ds = DataSet(outcome_labels=[('0',), ('1',)])
        self.ds.add_count_dict(('Gx', 'Gx'), {('0',): 40, ('1',): 60})
        self.ds.add_count_dict(('Gx', 'Gy'), {('0',): 40, ('1',): 60})
        self.ds.add_count_dict(('Gy', 'Gx'), {('0',): 40, ('1',): 60})
        self.ds.add_count_dict(('Gy', 'Gy'), {('0',): 40, ('1',): 60})
        self.ds.done_adding_data()


class GramMatrixTester(DataSetInstance, BaseCase):
    def test_get_max_gram_basis(self):
        basis = gm.max_gram_basis(('Gx', 'Gy'), self.ds)
        self.assertEqual(basis, [('Gx',), ('Gy',)])

    def test_max_gram_rank_and_evals(self):
        model = models.create_explicit_model([('Q0',)], ['Gx', 'Gy'],
                                        ["X(pi/4,Q0)", "Y(pi/4,Q0)"])
        rank, evals, tgt_evals = gm.max_gram_rank_and_eigenvalues(self.ds, model)
        self.assertEqual(rank, 1)
