""" Data comparison section """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.section import Section as _Section
from pygsti.data import DataComparator as _DataComparator
from pygsti.tools.mpitools import distribute_indices as _distribute_indices


class DataComparisonSection(_Section):
    _HTML_TEMPLATE = 'tabs/DataComparison.html'

    def render(self, workspace, results=None, dataset_labels=None, embed_figures=True, comm=None, **kwargs):
        #initialize a new "dataset comparison switchboard"
        dscmp_switchBd = workspace.Switchboard(
            ["Dataset1", "Dataset2"],
            [dataset_labels, dataset_labels],
            ["buttons", "buttons"], [0, 1],
            use_loadable_items=embed_figures
        )
        dscmp_switchBd.add("dscmp", (0, 1))
        dscmp_switchBd.add("dscmp_circuits", (0,))
        dscmp_switchBd.add("refds", (0,))
        for d1, dslbl1 in enumerate(dataset_labels):
            dscmp_switchBd.dscmp_circuits[d1] = results[dslbl1].circuit_lists['final']
            dscmp_switchBd.refds[d1] = results[dslbl1].dataset  # only used for #of spam labels below

        dsComp = dict()
        all_dsComps = dict()
        indices = []
        for i in range(len(dataset_labels)):
            for j in range(len(dataset_labels)):
                indices.append((i, j))
        if comm is not None:
            _, indexDict, _ = _distribute_indices(indices, comm)
            rank = comm.Get_rank()
            for k, v in indexDict.items():
                if v == rank:
                    d1, d2 = k
                    dslbl1 = dataset_labels[d1]
                    dslbl2 = dataset_labels[d2]

                    ds1 = results[dslbl1].dataset
                    ds2 = results[dslbl2].dataset
                    dsc = _DataComparator([ds1, ds2], ds_names=[dslbl1, dslbl2])
                    dsc.run()  # to perform processing
                    dsComp[(d1, d2)] = dsc
            dicts = comm.gather(dsComp, root=0)
            if rank == 0:
                for d in dicts:
                    for k, v in d.items():
                        d1, d2 = k
                        dscmp_switchBd.dscmp[d1, d2] = v
                        all_dsComps[(d1, d2)] = v
        else:
            for d1, d2 in indices:
                dslbl1 = dataset_labels[d1]
                dslbl2 = dataset_labels[d2]
                ds1 = results[dslbl1].dataset
                ds2 = results[dslbl2].dataset
                dsc = _DataComparator([ds1, ds2], ds_names=[dslbl1, dslbl2])
                dsc.run()  # to perform processing
                all_dsComps[(d1, d2)] = dsc
                dscmp_switchBd.dscmp[d1, d2] = all_dsComps[(d1, d2)]

        return {
            'dscmpSwitchboard': dscmp_switchBd,
            **super().render(
                workspace, all_dscomps=all_dsComps,
                ds_switchboard=dscmp_switchBd,
                dataset_labels=dataset_labels,
                embed_figures=embed_figures, comm=comm, **kwargs
            )
        }

    @_Section.figure_factory(4)
    def dataset_comparison_summary(workspace, switchboard=None, dataset_labels=None, all_dscomps=None, **kwargs):
        return workspace.DatasetComparisonSummaryPlot(
            dataset_labels, all_dscomps
        )

    @_Section.figure_factory(4)
    def dataset_comparison_histogram(workspace, switchboard=None, ds_switchboard=None, comm=None, bgcolor='white',
                                     **kwargs):
        return workspace.ColorBoxPlot(
            'dscmp', ds_switchboard.dscmp_circuits, ds_switchboard.refds,
            None, dscomparator=ds_switchboard.dscmp, typ='histogram',
            comm=comm, bgcolor=bgcolor
        )

    @_Section.figure_factory(4)
    def dataset_comparison_box_plot(workspace, switchboard=None, ds_switchboard=None, comm=None, bgcolor='white',
                                    **kwargs):
        return workspace.ColorBoxPlot(
            'dscmp', ds_switchboard.dscmp_circuits, ds_switchboard.refds,
            None, dscomparator=ds_switchboard.dscmp, comm=comm,
            bgcolor=bgcolor
        )
