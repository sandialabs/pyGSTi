from pygsti.models.localnoisemodel import _SimpleCompLayerRules, LocalNoiseModel as _LocalNoiseModel
from pygsti.baseobjs.label import Label, LabelTup, LabelTupTup
from pygsti.modelmembers.operations import opfactory as _opfactory



class SingleQuditGateEquivalenceClassesLayerRules(_SimpleCompLayerRules):
    """
    Submodel which assumes that you have a set of qubits for which you trust the action of a single
    qubit gate equally for all qubits within the set.
    """

    def __init__(self, qubit_labels, implicit_idle_mode, singleq_idle_layer_labels, global_idle_layer_label):

        super().__init__(qubit_labels, implicit_idle_mode, singleq_idle_layer_labels, global_idle_layer_label)

    def operation_layer_operator(self, model, layerlbl: Label, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        LinearOperator
        """

        if layerlbl in caches['complete-layers']:
            return caches['complete-layers'][layerlbl]
        
        if isinstance(layerlbl, LabelTupTup):
            # This could be a multiple qubit gate or multiple single qubit gates.

            group = []
            changed = False
            
            for op in layerlbl:
                assert isinstance(op, LabelTup)
                qubits_used = op.qubits
                if op.num_qubits == 1:
                    if model._qubits_to_equiv_qubit[qubits_used[0]] != qubits_used[0]:
                        new_label = Label(op.name, model._qubits_to_equiv_qubit[qubits_used[0]], op.time, *op.args)

                        changed = True
                        group.append(new_label)
                    else:
                        group.append(op)
                else:
                    group.append(op)

            if changed:
                new_args = None if layerlbl.args == () else layerlbl.args
                new_time = 0.0 if layerlbl.time == None else layerlbl.time
                new_label = Label(group)
            else:
                new_label = layerlbl

            # Get the operator
            if new_label in caches['complete-layers']:
                caches['complete-layers'][layerlbl] = caches['complete-layers'][new_label]
                return caches['complete-layers'][new_label]
            else:

                answer = super().operation_layer_operator(model, new_label, caches)
                caches['complete-layers'][new_label] = answer
                caches['complete-layers'][layerlbl] = answer
                return answer


        elif isinstance(layerlbl, LabelTup):

            qubits_used = layerlbl.qubits
            if layerlbl.num_qubits == 1:
                if model._qubits_to_equiv_qubit[qubits_used[0]] != qubits_used[0]:
                    new_args = None if layerlbl.args == () else layerlbl.args
                    new_time = 0.0 if layerlbl.time == None else layerlbl.time
                    new_label = Label(layerlbl.name, model._qubits_to_equiv_qubit[qubits_used[0]], new_time, new_args)

                    # Get the operator
                    if new_label in caches['complete-layers']:
                        caches['complete-layers'][layerlbl] = caches['complete-layers'][new_label]
                        return caches['complete-layers'][new_label]
                    else:

                        answer = super().operation_layer_operator(model, new_label, caches)
                        caches['complete-layer'][new_label] = answer
                        caches['complete-layer'][layerlbl] = answer
                        return answer

        return super().operation_layer_operator(model, layerlbl, caches)


class EquivalentClassesLocalNoiseModel(_LocalNoiseModel):

    def __init__(self, qubit_to_equivalent_qubit_for_single_qgates: dict, processor_spec, gatedict, prep_layers=None, povm_layers=None, evotype="default",
                simulator="auto", on_construction_error='raise',
                independent_gates=False, ensure_composed_gates=False, implicit_idle_mode="none"):


        super().__init__(processor_spec, gatedict, prep_layers, povm_layers, evotype, simulator,
                         on_construction_error, independent_gates, ensure_composed_gates, implicit_idle_mode)
        
        # Now we need to reset the layer rules to use the Equivalent class rules.

        old_rules = self._layer_rules

        new_rules = SingleQuditGateEquivalenceClassesLayerRules( old_rules.qubit_labels, old_rules.implicit_idle_mode,
                                                                old_rules.single_qubit_idle_layer_labels, old_rules.global_idle_layer_label)
        
        self._layer_rules = new_rules
        self._qubits_to_equiv_qubit = qubit_to_equivalent_qubit_for_single_qgates
        self._reinit_opcaches() # Clear the caches for using the new rules.
