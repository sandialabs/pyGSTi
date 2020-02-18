import unittest
from nose.tools import raises

from ..util import BaseCase

from pygsti.io.circuitparser import slowcircuitparser

try:
    from pygsti.io.circuitparser import fastcircuitparser
    _FASTCIRCUITPARSER_LOADED = True
except ImportError:
    _FASTCIRCUITPARSER_LOADED = False


def _test_circuit_parser(parser):

    def test_parse_circuit(string, expected):
        results, line_labels = parser.parse_circuit(string, create_subcircuits=True, integerize_sslbls=True)
        assert line_labels is None
        flat_results = [lbl for item in results for lbl in item.expand_subcircuits()]
        for result_label, expected_label in zip(flat_results, expected):
            assert result_label == expected_label

    test_cases = [
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

    for string, expected in test_cases:
        yield test_parse_circuit, string, expected

    @raises(ValueError)
    def test_parse_circuit_raises_on_syntax_error(string):
        parser.parse_circuit(string, create_subcircuits=True, integerize_sslbls=True)

    fail_cases = [
        "FooBar",
        "G1G2^2^2",
        "(G1"
    ]

    for string in fail_cases:
        yield test_parse_circuit_raises_on_syntax_error, string


def test_slow_circuit_parser():
    yield from _test_circuit_parser(slowcircuitparser)


@unittest.skipUnless(_FASTCIRCUITPARSER_LOADED, "`pygsti.io.fastcircuitparser` not built")
def test_fast_circuit_parser():
    yield from _test_circuit_parser(fastcircuitparser)
