import unittest

from ..util import BaseCase

from pygsti.circuits.circuitparser import slowcircuitparser

try:
    from pygsti.circuits.circuitparser import fastcircuitparser
    _FASTCIRCUITPARSER_LOADED = True
except ImportError:
    _FASTCIRCUITPARSER_LOADED = False

class CircuitParserBase():
    @classmethod
    def setUpClass(cls):
        cls.test_cases = [
            ("{}", ()),
            ("{}^127", ()),
            ("{}^0002", ()),
            ("G1", ('G1',)),
            ("G1G2G3", ('G1', 'G2', 'G3')),
            ("G1(G2)G3", ('G1', 'G2', 'G3')),
            ("G1(G2)^3G3", ('G1', 'G2', 'G2', 'G2', 'G3')),
            ("G1(G2G3)^2", ('G1', 'G2', 'G3', 'G2', 'G3')),
            ("G1*G2*G3", ('G1', 'G2', 'G3')),
            ("G1^02", ('G1', 'G1')),
            ("G1*((G2G3)^2G4G5)^2G7", ('G1', 'G2', 'G3', 'G2', 'G3',
                                    'G4', 'G5', 'G2', 'G3', 'G2', 'G3', 'G4', 'G5', 'G7')),
            ("G1(G2^2(G3G4)^2)^2", ('G1', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4', 'G2', 'G2', 'G3', 'G4', 'G3', 'G4')),
            ("G_my_xG_my_y", ('G_my_x', 'G_my_y')),
            ("G_my_x*G_my_y", ('G_my_x', 'G_my_y')),
            ("GsG___", ('Gs', 'G___'))
        ]
        cls.fail_cases =  [
            "FooBar",
            "G1G2^2^2",
            "(G1"
        ]
    
    def setUp(self):
        self.parser = None
    
    def test_parse_circuit(self):
        for string, expected in self.test_cases:
            results, line_labels, occurrence, compilable_indices = self.parser.parse_circuit(string, create_subcircuits=True, integerize_sslbls=True)
            assert line_labels is None
            assert occurrence is None
            assert compilable_indices is None
            flat_results = [lbl for item in results for lbl in item.expand_subcircuits()]
            for result_label, expected_label in zip(flat_results, expected):
                assert result_label == expected_label
    
    def test_parse_circuit_raises_on_syntax_error(self):
        for string in self.fail_cases:
            with self.assertRaises(ValueError):
                self.parser.parse_circuit(string, create_subcircuits=True, integerize_sslbls=True)

class SlowParser(CircuitParserBase, BaseCase):
    def setUp(self):
        self.parser = slowcircuitparser

@unittest.skipUnless(_FASTCIRCUITPARSER_LOADED, "`pygsti.io.fastcircuitparser` not built")
class FastParser(CircuitParserBase, BaseCase):
    def setUp(self):
        self.parser = fastcircuitparser
