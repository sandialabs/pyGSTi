import pickle

from ..util import BaseCase

from pygsti.io import jsoncodec
from pygsti.baseobjs import label


class LabelBase:
    def testLabels(self):
        labels = []
        labels.append(label.Label('Gx', 0))  # a LabelTup
        labels.append(label.Label('Gx', (0, 1)))  # a LabelTup
        labels.append(label.Label(('Gx', 0, 1)))  # a LabelTup
        labels.append(label.Label('Gx'))  # a LabelStr
        labels.append(label.Label('Gx', None))  # still a LabelStr
        labels.append(label.Label([('Gx', 0), ('Gy', 0)]))  # a LabelTupTup of LabelTup objs
        labels.append(label.Label((('Gx', None), ('Gy', None))))  # a LabelTupTup of LabelStr objs
        labels.append(label.Label([('Gx', 0)]))  # just a LabelTup b/c only one component
        labels.append(label.Label([label.Label('Gx'), label.Label('Gy')]))  # a LabelTupTup of LabelStrs
        labels.append(label.Label(label.Label('Gx')))  # Init from another label

        for l in labels:
            native = l.tonative()
            # TODO assert correctness
            from_native = label.Label(native)
            self.assertEqual(from_native, l)

            s = pickle.dumps(l)
            l2 = pickle.loads(s)
            self.assertEqual(type(l), type(l2))

            j = jsoncodec.encode_obj(l, False)
            l3 = jsoncodec.decode_obj(j, False)
            self.assertEqual(type(l), type(l3))
