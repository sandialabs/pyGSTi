from ..util import BaseCase
from ..algorithms import fixtures

from pygsti.objects import Circuit, Label

import pygsti.construction.circuitconstruction as cc


class CircuitConstructionTester(BaseCase):
    def test_simple_gatestrings(self):
        #The workhorse function is cc.create_circuit_list, which executes its positional arguments within a nested
        #loop given by iterable keyword arguments.  That's a mouthful, so let's look at a few examples:
        As = [('a1',), ('a2',)]
        Bs = [('b1',), ('b2',)]

        def rep2(x):
            return x + x

        def asserter(x):
            assert(False)

        def samestr(x):
            return "Gx"  # to test string processing

        def sametup(x):
            return "Gx"  # to test string processing

        list0 = cc.create_circuit_list("")
        self.assertEqual(list0, cc.circuit_list([()]))  # special case: get the empty operation sequence

        list1 = cc.create_circuit_list("a", a=As)
        self.assertEqual(list1, cc.circuit_list(As))

        list2 = cc.create_circuit_list("a+b", a=As, b=Bs, order=['a', 'b'])
        self.assertEqual(list2, cc.circuit_list([('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1'), ('a2', 'b2')]))

        list3 = cc.create_circuit_list("a+b", a=As, b=Bs, order=['b', 'a'])
        self.assertEqual(list3, cc.circuit_list([('a1', 'b1'), ('a2', 'b1'), ('a1', 'b2'), ('a2', 'b2')]))

        list4 = cc.create_circuit_list("R(a)+c", a=As, c=[('c',)], R=rep2, order=['a', 'c'])
        self.assertEqual(list4, cc.circuit_list([('a1', 'a1', 'c'), ('a2', 'a2', 'c')]))

        list5 = cc.create_circuit_list("Ast(a)", a=As, Ast=asserter)
        self.assertEqual(list5, [])  # failed assertions cause item to be skipped

        list6 = cc.create_circuit_list("SS(a)", a=As, SS=samestr)
        self.assertEqual(list6, cc.circuit_list([('Gx',), ('Gx',)]))  # strs => parser => Circuits

        list7 = cc.circuit_list(list1)
        self.assertEqual(list7, list1)

        with self.assertRaises(ValueError):
            cc.circuit_list([{'foo': "Bar"}])  # cannot convert dicts to Circuits...

    def test_fiducials_germ_gatestrings(self):
        fids = cc.circuit_list([('Gf0',), ('Gf1',)])
        germs = cc.circuit_list([('G0',), ('G1a', 'G1b')])

        gateStrings1 = cc.create_circuit_list("f0+germ*e+f1", f0=fids, f1=fids,
                                              germ=germs, e=2, order=["germ", "f0", "f1"])
        expected1 = ["Gf0(G0)^2Gf0",
                     "Gf0(G0)^2Gf1",
                     "Gf1(G0)^2Gf0",
                     "Gf1(G0)^2Gf1",
                     "Gf0(G1aG1b)^2Gf0",
                     "Gf0(G1aG1b)^2Gf1",
                     "Gf1(G1aG1b)^2Gf0",
                     "Gf1(G1aG1b)^2Gf1"]
        self.assertEqual([x.str for x in gateStrings1], expected1)

        gateStrings2 = cc.create_circuit_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                              germ=germs, N=3, T=cc.repeat_and_truncate,
                                              order=["germ", "f0", "f1"])
        expected2 = ["Gf0G0G0G0Gf0",
                     "Gf0G0G0G0Gf1",
                     "Gf1G0G0G0Gf0",
                     "Gf1G0G0G0Gf1",
                     "Gf0G1aG1bG1aGf0",
                     "Gf0G1aG1bG1aGf1",
                     "Gf1G1aG1bG1aGf0",
                     "Gf1G1aG1bG1aGf1"]
        self.assertEqual([x.str for x in gateStrings2], expected2)

        gateStrings3 = cc.create_circuit_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                              germ=germs, N=3,
                                              T=cc.repeat_with_max_length,
                                              order=["germ", "f0", "f1"])
        expected3 = ["Gf0(G0)^3Gf0",
                     "Gf0(G0)^3Gf1",
                     "Gf1(G0)^3Gf0",
                     "Gf1(G0)^3Gf1",
                     "Gf0(G1aG1b)Gf0",
                     "Gf0(G1aG1b)Gf1",
                     "Gf1(G1aG1b)Gf0",
                     "Gf1(G1aG1b)Gf1"]
        self.assertEqual([x.str for x in gateStrings3], expected3)

    def test_truncate_methods(self):
        self.assertEqual(cc.repeat_and_truncate(('A', 'B', 'C'), 5), ('A', 'B', 'C', 'A', 'B'))
        self.assertEqual(cc.repeat_with_max_length(('A', 'B', 'C'), 5), ('A', 'B', 'C'))
        self.assertEqual(cc.repeat_count_with_max_length(('A', 'B', 'C'), 5), 1)

    def test_repeat_methods(self):
        mdl = Circuit(('Gx', 'Gx', 'Gy'))

        gs2 = cc.repeat(mdl, 2)
        self.assertEqual(gs2, Circuit(('Gx', 'Gx', 'Gy', 'Gx', 'Gx', 'Gy')))

        gs3 = cc.repeat_with_max_length(mdl, 7)
        self.assertEqual(gs3, Circuit(('Gx', 'Gx', 'Gy', 'Gx', 'Gx', 'Gy')))

        gs4 = cc.repeat_and_truncate(mdl, 4)
        self.assertEqual(gs4, Circuit(('Gx', 'Gx', 'Gy', 'Gx')))

        gs5 = cc.repeat_remainder_for_truncation(mdl, 4)
        self.assertEqual(gs5, Circuit(('Gx',)))

    def test_simplify(self):
        s = "{}Gx^1Gy{}Gz^1"
        self.assertEqual(cc.simplify_str(s), "GxGyGz")

        s = "{}Gx^1(Gy)^2{}Gz^1"
        self.assertEqual(cc.simplify_str(s), "Gx(Gy)^2Gz")

        s = "{}{}^1{}"
        self.assertEqual(cc.simplify_str(s), "{}")

    def test_circuit_list_accessors(self):
        expected_allStrs = set(cc.circuit_list(
            [(), ('Gx',), ('Gy',), ('Gx', 'Gx'), ('Gx', 'Gy'), ('Gy', 'Gx'), ('Gy', 'Gy')]))
        allStrs = cc.list_all_circuits(('Gx', 'Gy'), 0, 2)
        self.assertEqual(set(allStrs), expected_allStrs)

        allStrs = list(cc.gen_all_circuits(('Gx', 'Gy'), 0, 2))
        self.assertEqual(set(allStrs), expected_allStrs)

        expected_onelenStrs = set(cc.circuit_list(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gy', 'Gx'), ('Gy', 'Gy')]))
        onelenStrs = cc.list_all_circuits_onelen(('Gx', 'Gy'), 2)
        self.assertEqual(set(onelenStrs), expected_onelenStrs)

        randStrs = cc.list_random_circuits_onelen(('Gx', 'Gy', 'Gz'), 2, 3)
        self.assertEqual(len(randStrs), 3)
        self.assertTrue(all([len(s) == 2 for s in randStrs]))
        # TODO should assert correctness beyond this

        partialStrs = cc.list_partial_strings(('G1', 'G2', 'G3'))
        self.assertEqual(partialStrs, [(), ('G1',), ('G1', 'G2'), ('G1', 'G2', 'G3')])

    def test_translate_circuit_list(self):
        orig_list = cc.circuit_list(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy'), ('Gi',)]
        )

        list0 = cc.translate_circuit_list(orig_list, None)
        self.assertEqual(list0, orig_list)

        list1 = cc.translate_circuit_list(orig_list, {Label('Gx'): (Label('Gx2'),), Label('Gy'): (Label('Gy'),)})
        expected_list1 = cc.circuit_list(
            [('Gx2', 'Gx2'), ('Gx2', 'Gy'), ('Gx2', 'Gx2', 'Gx2'), ('Gy', 'Gy'), ('Gi',)]
        )
        self.assertEqual(list1, expected_list1)

        list2 = cc.translate_circuit_list(
            orig_list,
            {Label('Gi'): (Label('Gx'), Label('Gx'), Label('Gx'), Label('Gx'))}
        )
        expected_list2 = cc.circuit_list(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy'), ('Gx', 'Gx', 'Gx', 'Gx')]
        )
        self.assertEqual(list2, expected_list2)

    def test_compose_alias_dicts(self):
        aliasDict1 = {'A': ('B', 'B')}
        aliasDict2 = {'B': ('C', 'C')}
        aliasDict3 = cc.compose_alias_dicts(aliasDict1, aliasDict2)
        self.assertEqual(aliasDict3, {'A': ('C', 'C', 'C', 'C')})

    def test_manipulate_circuit(self):
        sequenceRules = [
            (('A', 'B'), ('A', 'B\'')),
            (('B', 'A'), ('B\'\'', 'A')),
            (('C', 'A'), ('C', 'A\'')),
            (('B', 'C'), ('B', 'C\'')),
            (('D',), ('E',)),
            (('A', 'A'), ('A', 'B', 'C',))]

        result = cc.manipulate_circuit(tuple('BAB'), sequenceRules)
        self.assertEqual(result, ("B''", "A", "B'"))

        result = cc.manipulate_circuit(tuple('ABA'), sequenceRules)
        self.assertEqual(result, ("A", "B'", "A"))

        result = cc.manipulate_circuit(tuple('CAB'), sequenceRules)
        self.assertEqual(result, ("C", "A'", "B'"))

        result = cc.manipulate_circuit(tuple('ABC'), sequenceRules)
        self.assertEqual(result, ("A", "B'", "C'"))

        result = cc.manipulate_circuit(tuple('DD'), sequenceRules)
        self.assertEqual(result, ("E", "E"))

        result = cc.manipulate_circuit(tuple('AA'), sequenceRules)
        self.assertEqual(result, ("A", "B", "C"))

        result = cc.manipulate_circuit(tuple('AAAA'), sequenceRules)
        self.assertEqual(result, ("A", "B", "C", "B", "C", "B", "C"))

        results = cc.manipulate_circuit_list([tuple('ABC'), tuple('GHI')], sequenceRules)
        results_trivial = cc.manipulate_circuit_list([tuple('ABC'), tuple('GHI')], None)  # special case
        # TODO assert correctness

    def test_list_strings_lgst_can_estimate(self):
        strs = cc.list_strings_lgst_can_estimate(fixtures.ds, fixtures.fiducials, fixtures.fiducials)
        # TODO assert correctness
