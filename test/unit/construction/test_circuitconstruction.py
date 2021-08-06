import pygsti.circuits.circuitconstruction as cc
import pygsti.data.datasetconstruction as dc
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
from ..util import BaseCase


class CircuitConstructionTester(BaseCase):
    def test_simple_gatestrings(self):
        #The workhorse function is cc.create_circuits, which executes its positional arguments within a nested
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

        list0 = cc.create_circuits("")
        self.assertEqual(list0, cc.to_circuits([()]))  # special case: get the empty operation sequence

        list1 = cc.create_circuits("a", a=As)
        self.assertEqual(list1, cc.to_circuits(As))

        list2 = cc.create_circuits("a+b", a=As, b=Bs, order=['a', 'b'])
        self.assertEqual(list2, cc.to_circuits([('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1'), ('a2', 'b2')]))

        list3 = cc.create_circuits("a+b", a=As, b=Bs, order=['b', 'a'])
        self.assertEqual(list3, cc.to_circuits([('a1', 'b1'), ('a2', 'b1'), ('a1', 'b2'), ('a2', 'b2')]))

        list4 = cc.create_circuits("R(a)+c", a=As, c=[('c',)], R=rep2, order=['a', 'c'])
        self.assertEqual(list4, cc.to_circuits([('a1', 'a1', 'c'), ('a2', 'a2', 'c')]))

        list5 = cc.create_circuits("Ast(a)", a=As, Ast=asserter)
        self.assertEqual(list5, [])  # failed assertions cause item to be skipped

        list6 = cc.create_circuits("SS(a)", a=As, SS=samestr)
        self.assertEqual(list6, cc.to_circuits([('Gx',), ('Gx',)]))  # strs => parser => Circuits

        list7 = cc.to_circuits(list1)
        self.assertEqual(list7, list1)

        with self.assertRaises(ValueError):
            cc.to_circuits([{'foo': "Bar"}])  # cannot convert dicts to Circuits...

    def test_fiducials_germ_gatestrings(self):
        fids = cc.to_circuits([('Gf0',), ('Gf1',)])
        germs = cc.to_circuits([('G0',), ('G1a', 'G1b')])

        #Ensure string reps are computed so circuit addition produces nice string reps (that we expect below)
        [germ.str for germ in germs]
        [c.str for c in fids]

        gateStrings1 = cc.create_circuits("f0+germ*e+f1", f0=fids, f1=fids,
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

        gateStrings2 = cc.create_circuits("f0+T(germ,N)+f1", f0=fids, f1=fids,
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

        gateStrings3 = cc.create_circuits("f0+T(germ,N)+f1", f0=fids, f1=fids,
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

        gs5 = cc._repeat_remainder_for_truncation(mdl, 4)
        self.assertEqual(gs5, Circuit(('Gx',)))

    def test_simplify(self):
        s = "{}Gx^1Gy{}Gz^1"
        self.assertEqual(cc._simplify_circuit_string(s), "GxGyGz")

        s = "{}Gx^1(Gy)^2{}Gz^1"
        self.assertEqual(cc._simplify_circuit_string(s), "Gx(Gy)^2Gz")

        s = "{}{}^1{}"
        self.assertEqual(cc._simplify_circuit_string(s), "{}")

    def test_circuit_list_accessors(self):
        expected_allStrs = set(cc.to_circuits(
            [(), ('Gx',), ('Gy',), ('Gx', 'Gx'), ('Gx', 'Gy'), ('Gy', 'Gx'), ('Gy', 'Gy')]))
        allStrs = cc.list_all_circuits(('Gx', 'Gy'), 0, 2)
        self.assertEqual(set(allStrs), expected_allStrs)

        allStrs = list(cc.iter_all_circuits(('Gx', 'Gy'), 0, 2))
        self.assertEqual(set(allStrs), expected_allStrs)

        expected_onelenStrs = set(cc.to_circuits(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gy', 'Gx'), ('Gy', 'Gy')]))
        onelenStrs = cc.list_all_circuits_onelen(('Gx', 'Gy'), 2)
        self.assertEqual(set(onelenStrs), expected_onelenStrs)

        randStrs = cc.list_random_circuits_onelen(('Gx', 'Gy', 'Gz'), 2, 3)
        self.assertEqual(len(randStrs), 3)
        self.assertTrue(all([len(s) == 2 for s in randStrs]))
        # TODO should assert correctness beyond this

        partialStrs = cc.list_partial_circuits(('G1', 'G2', 'G3'))
        self.assertEqual(partialStrs, [(), ('G1',), ('G1', 'G2'), ('G1', 'G2', 'G3')])

    def test_translate_circuit_list(self):
        orig_list = cc.to_circuits(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy'), ('Gi',)]
        )

        list0 = cc.translate_circuits(orig_list, None)
        self.assertEqual(list0, orig_list)

        list1 = cc.translate_circuits(orig_list, {Label('Gx'): (Label('Gx2'),), Label('Gy'): (Label('Gy'),)})
        expected_list1 = cc.to_circuits(
            [('Gx2', 'Gx2'), ('Gx2', 'Gy'), ('Gx2', 'Gx2', 'Gx2'), ('Gy', 'Gy'), ('Gi',)]
        )
        self.assertEqual(list1, expected_list1)

        list2 = cc.translate_circuits(
            orig_list,
            {Label('Gi'): (Label('Gx'), Label('Gx'), Label('Gx'), Label('Gx'))}
        )
        expected_list2 = cc.to_circuits(
            [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy'), ('Gx', 'Gx', 'Gx', 'Gx')]
        )
        self.assertEqual(list2, expected_list2)

    def test_compose_alias_dicts(self):
        aliasDict1 = {'A': ('B', 'B')}
        aliasDict2 = {'B': ('C', 'C')}
        aliasDict3 = cc._compose_alias_dicts(aliasDict1, aliasDict2)
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

        results = cc.manipulate_circuits([tuple('ABC'), tuple('GHI')], sequenceRules)
        self.assertEqual(results, [("A", "B'", "C'"), tuple('GHI')])
        results_trivial = cc.manipulate_circuits([tuple('ABC'), tuple('GHI')], None)  # special case
        self.assertEqual(results_trivial, [tuple('ABC'), tuple('GHI')])

    def test_list_strings_lgst_can_estimate(self):
        model = std.target_model()
        fids = std.fiducials[:3]
        germs = std.germs[:3]

        # Construct full set
        circuit_list = []
        for f1 in fids:
            for f2 in fids:
                for g in germs:
                    circuit_list.append(f1 + g + f2)
        
        ds = dc.simulate_data(model, circuit_list, 1)
        estimatable_germs = cc.list_circuits_lgst_can_estimate(ds, fids, fids)
        self.assertEqual(set(germs), set(estimatable_germs))
    
        # Add germ with incomplete fiducials
        circuit_list.append(fids[0] + germs[1] + germs[2] + fids[1])

        ds = dc.simulate_data(model, circuit_list, 1)
        estimatable_germs = cc.list_circuits_lgst_can_estimate(ds, fids, fids)
        self.assertEqual(set(germs), set(estimatable_germs))

        # Asymmetric fiducials
        fids2 = fids[:2]
        circuit_list = []
        for f1 in fids:
            for f2 in fids2:
                for g in germs:
                    circuit_list.append(f1 + g + f2)
        
        ds = dc.simulate_data(model, circuit_list, 1)
        estimatable_germs = cc.list_circuits_lgst_can_estimate(ds, fids, fids2)
        self.assertEqual(set(germs), set(estimatable_germs))
    
        
