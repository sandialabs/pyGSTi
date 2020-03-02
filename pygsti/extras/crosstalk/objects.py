#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""Functions for crosstalk detection from time-stamped data"""

import numpy as _np


class CrosstalkResults(object):

    def __init__(self):

        #--------------------------#
        # --- Input quantities --- #
        #--------------------------#

        self.name = None
        self.data = None
        self.pygsti_ds = None
        self.number_of_regions = None
        self.settings = None
        self.number_of_datapoints = None
        self.number_of_columns = None
        self.confidence = None

        #----------------------------#
        # --- Derived quantities --- #
        #----------------------------#

        self.skel = None
        self.sep_set = None
        self.graph = None
        self.node_labels = None
        self.setting_indices = None
        self.cmatrix = None
        self.crosstalk_detected = None
        self.is_edge_ct = None
        self.edge_weights = None
        self.edge_tvds = None
        self.max_tvds = None
        self.median_tvds = None
        self.max_tvd_explanations = None

    def any_crosstalk_detect(self):

        if self.crosstalk_detected:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence))
            print("Result: The 'no crosstalk' hypothesis *is* rejected.")
        else:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence))
            print("Result: The 'no crosstalk' hypothesis is *not* rejected.")

    def plot_crosstalk_matrices(self, figsize=(15, 3), savepath=None):
        """

        """

        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_crosstalk_matrix(...) requires you to install matplotlib")
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(sum(self.settings)
                                                       + self.number_of_regions + 6, self.number_of_regions + 4))
        fig.subplots_adjust(wspace=2, hspace=2)

        # common arguments to imshow
        kwargs = dict(
            origin='lower', interpolation='nearest', vmin=0, vmax=1, aspect='equal', cmap='YlOrBr')

        settings_and_regions = _np.zeros((sum(self.settings), self.number_of_regions))
        regions_and_regions = _np.zeros((self.number_of_regions, self.number_of_regions))

        def _setting_range(x):
            return range(
                self.setting_indices[x],
                self.setting_indices[x + 1] if x < (self.number_of_regions - 1) else self.number_of_columns
            )

        for idx, edge in enumerate(self.graph.edges()):
            source = edge[0]
            dest = edge[1]

            # edge between two outcomes
            if source < self.number_of_regions and dest < self.number_of_regions:
                regions_and_regions[source, dest] = self.max_tvds[idx]

            # edge between an outcome and a setting
            if source < self.number_of_regions and dest >= self.number_of_regions:
                if dest not in range(self.setting_indices[source],
                                     (self.setting_indices[(source + 1)] if source < (self.number_of_regions - 1)
                                      else self.number_of_columns)):
                    settings_and_regions[dest - self.number_of_regions, source] = self.max_tvds[idx]

            # edge between an outcome and a setting
            if source >= self.number_of_regions and dest < self.number_of_regions:
                if source not in range(self.setting_indices[dest],
                                       (self.setting_indices[(dest + 1)] if dest < (self.number_of_regions - 1)
                                        else self.number_of_columns)):
                    settings_and_regions[source - self.number_of_regions, dest] = self.max_tvds[idx]

        ax1.imshow(_np.transpose(settings_and_regions), **kwargs)
        _plt.setp(ax1, xticks=_np.arange(0, sum(self.settings), 1),
                  xticklabels=[self.node_labels[k] for k in range(self.number_of_regions, self.number_of_columns)],
                  yticks=_np.arange(0, self.number_of_regions, 1),
                  yticklabels=_np.arange(0, self.number_of_regions, 1).astype('str'))

        dividers = [sum(self.settings[:k]) - 0.5 for k in range(1, self.number_of_regions)]
        for i in range(len(dividers)):
            ax1.axvline(dividers[i], color='k')

        ax1.set_xlabel('Settings')
        ax1.set_ylabel('Region outcomes')
        ax1.set_title('Crosstalk between Region outcomes and settings')

        im = ax2.imshow(regions_and_regions, **kwargs)
        _plt.setp(ax2, xticks=_np.arange(0, self.number_of_regions, 1),
                  xticklabels=_np.arange(0, self.number_of_regions, 1).astype('str'),
                  yticks=_np.arange(0, self.number_of_regions, 1),
                  yticklabels=_np.arange(0, self.number_of_regions, 1).astype('str'))
        ax2.set_xlabel('Region outcomes')
        ax2.set_ylabel('Region outcomes')
        ax2.set_title('Crosstalk between Region outcomes')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        if savepath is not None:
            _plt.savefig(savepath, bbox_inches='tight')
        else:
            _plt.show()

    def plot_crosstalk_dag(self, savepath=None):
        """

        """

        try:
            import networkx as _nx
        except ImportError:
            raise ValueError("plot_crosstalk_dag(...) requires you to install networkx")

        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_crosstalk_dag(...) requires you to install matplotlib")
        # fig = _plt.figure(figsize=(sum(self.settings)+2,6), facecolor='white')
        fig = _plt.figure(facecolor='white')
        ax = fig.add_subplot(1, 1, 1)

        if self.name is not None:
            title = 'Crosstalk DAG for dataset ' + self.name + '. Confidence level ' + str(self.confidence)
        else:
            title = 'Crosstalk DAG for dataset. Confidence level ' + str(self.confidence)

        # set positions for each node in graph
        G = self.graph
        pos = {}
        # settings are distributed along y=1 line
        pos.update((n, (n - self.number_of_regions, 1)) for n in range(self.number_of_regions, self.number_of_columns))

        # results are distributed along y=3 line
        for region in range(self.number_of_regions):
            num_settings_before = sum(self.settings[0:region])
            num_settings = self.settings[region]

            if num_settings == 1:
                pos.update({region: (num_settings_before, 3)})
            else:
                pos.update({region: (num_settings_before + (num_settings - 1) / 2, 3)})

        # node colors
        settings_color = 'xkcd:light grey'
        outcomes_color = 'xkcd:light violet'

        # draw graph nodes
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_regions), node_size=1000,
                                node_color=outcomes_color, node_shape='o', alpha=0.4, ax=ax)
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_regions, self.number_of_columns), node_size=1000,
                                node_color=settings_color, node_shape='s', alpha=0.4, ax=ax)

        label_posns = self.get_offset_label_posns(pos)

        _nx.draw_networkx_labels(G, pos=label_posns, labels=self.node_labels, ax=ax)

        def float_formatter(x): return "%.4f" % x

        # draw graph edge, with ones indicating crosstalk in red
        for idx, edge in enumerate(self.graph.edges()):
            if self.is_edge_ct[idx]:
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='r', ax=ax)
                label = {}
                label[edge] = float_formatter(_np.max(self.edge_tvds[idx]))
                _nx.draw_networkx_edge_labels(G, pos, edge_labels=label, label_pos=0.2, ax=ax)
            else:
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='b', ax=ax)

        # insert plot title
        _plt.title(title, fontsize=17, y=3)

        # expand axis limits to make sure node labels are visible
        ylims = ax.get_ylim()
        ax.set_ylim((ylims[0] - 0.2, ylims[1] + 0.2))
        xlims = ax.get_xlim()
        ax.set_xlim((xlims[0] - 0.2, xlims[1] + 0.2))

        # don't display axis
        _plt.axis('off')

        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()

    def plot_crosstalk_graph(self, savepath=None):
        """

        """

        try:
            import networkx as _nx
        except ImportError:
            raise ValueError("plot_crosstalk_graph(...) requires you to install networkx")

        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_crosstalk_graph(...) requires you to install matplotlib")
        # fig = _plt.figure(figsize=(sum(self.settings)+2,6), facecolor='white')
        fig = _plt.figure(facecolor='white')
        ax = fig.add_subplot(1, 1, 1)

        if self.name is not None:
            title = 'Crosstalk graph for dataset ' + self.name + '. Confidence level ' + str(self.confidence)
        else:
            title = 'Crosstalk graph for dataset. Confidence level ' + str(self.confidence)

        # set positions for each node in graph
        G = self.skel
        pos = {}
        # settings are distributed along y=1 line
        pos.update((n, (n - self.number_of_regions, 1)) for n in range(self.number_of_regions, self.number_of_columns))

        # results are distributed along y=3 line
        for region in range(self.number_of_regions):
            num_settings_before = sum(self.settings[0:region])
            num_settings = self.settings[region]

            if num_settings == 1:
                pos.update({region: (num_settings_before, 3)})
            else:
                pos.update({region: (num_settings_before + (num_settings - 1) / 2, 3)})

        # node colors
        settings_color = 'xkcd:light grey'
        outcomes_color = 'xkcd:light violet'

        # draw graph nodes
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_regions), node_size=1000,
                                node_color=outcomes_color, node_shape='o', alpha=0.4, ax=ax)
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_regions, self.number_of_columns), node_size=1000,
                                node_color=settings_color, node_shape='s', alpha=0.4, ax=ax)

        label_posns = self.get_offset_label_posns(pos)

        _nx.draw_networkx_labels(G, pos=label_posns, labels=self.node_labels, ax=ax)

        def float_formatter(x): return "%.4f" % x

        # draw graph edge, with ones indicating crosstalk in red
        for idx, edge in enumerate(self.graph.edges()):
            if self.is_edge_ct[idx]:
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='r', ax=ax)
                label = {}
                label[edge] = float_formatter(_np.max(self.edge_tvds[idx]))
                _nx.draw_networkx_edge_labels(G, pos, edge_labels=label, label_pos=0.2)
            else:
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='b', ax=ax)

        # insert plot title
        _plt.title(title, fontsize=17)

        # expand axis limits to make sure node labels are visible
        ylims = ax.get_ylim()
        ax.set_ylim((ylims[0] - 0.2, ylims[1] + 0.2))
        xlims = ax.get_xlim()
        ax.set_xlim((xlims[0] - 0.2, xlims[1] + 0.2))

        # don't display axis
        _plt.axis('off')

        if savepath is not None:
            _plt.savefig(savepath, bbox_inches='tight')
        else:
            _plt.show()

    def show_crosstalk_table(self, precision=5, savepath=None):
        """

        """

        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("show_crosstalk_table(...) requires you to install matplotlib")
        #fig = _plt.figure(facecolor='white')
        #ax = fig.add_subplot(1, 1, 1)

        if self.name is not None:
            title = 'Crosstalk table for dataset ' + self.name + '. Confidence level ' + str(self.confidence) + '\n'
        else:
            title = 'Crosstalk table for dataset. Confidence level ' + str(self.confidence) + '\n'

        columns = ('Node 1', 'Node 2', 'Max TVD', 'Median TVD')

        cell_text = []
        for idx, edge in enumerate(self.graph.edges()):
            if self.is_edge_ct[idx]:
                source = edge[0]
                dest = edge[1]

                print(idx)
                print(self.max_tvds)

                # edge between two outcomes
                if source < self.number_of_regions and dest < self.number_of_regions:
                    cell_text.append([r'R$_{%d}$' % source,
                                      r'R$_{%d}$' % dest,
                                      _np.around(self.max_tvds[idx], decimals=precision),
                                      _np.around(self.median_tvds[idx], decimals=precision)])

                # edge between an outcome and a setting (source is outcome, dest is setting)
                if source < self.number_of_regions and dest >= self.number_of_regions:
                    if dest not in range(self.setting_indices[source],
                                         (self.setting_indices[(source + 1)] if source < (self.number_of_regions - 1)
                                          else self.number_of_columns)):

                        region, setting_number = self.get_setting_region_and_number(dest)

                        cell_text.append([r'R$_{%d}$' % source,
                                          r'S$_{%d}^{(%d)}$' % (region, setting_number),
                                          _np.around(self.max_tvds[idx], decimals=precision),
                                          _np.around(self.median_tvds[idx], decimals=precision)])
                        # (dest-self.setting_indices[region])),

                # edge between an outcome and a setting (source is setting, dest is outcome)
                if source >= self.number_of_regions and dest < self.number_of_regions:
                    if source not in range(self.setting_indices[dest],
                                           (self.setting_indices[(dest + 1)] if dest < (self.number_of_regions - 1)
                                            else self.number_of_columns)):

                        region, setting_number = self.get_setting_region_and_number(source)

                        cell_text.append([r'S$_{%d}^{(%d)}$' % (region, setting_number),
                                          r'R$_{%d}$' % dest,
                                          _np.around(self.max_tvds[idx], decimals=precision),
                                          _np.around(self.median_tvds[idx], decimals=precision)])
                        #(source-self.setting_indices[region])),

        thetable = _plt.table(cellText=cell_text, colLabels=columns, loc='center')

        thetable.auto_set_font_size(False)
        thetable.set_fontsize(14)
        thetable.scale(1.7, 1.7)
        # insert plot title
        _plt.title(title, fontsize=17)

        # expand axis limits to make sure node labels are visible
#        ylims = ax.get_ylim()
#        ax.set_ylim((ylims[0]-0.2, ylims[1]+0.2))
#        xlims = ax.get_xlim()
#        ax.set_xlim((xlims[0]-0.2, xlims[1]+0.2))

        # don't display axis
        _plt.axis('off')

        if savepath is not None:
            _plt.savefig(savepath, bbox_inches='tight')
        else:
            _plt.show()

    def get_offset_label_posns(self, pos):
        """
            From https://stackoverflow.com/questions/11946005/label-nodes-outside-with-minimum-overlap-with-other-nodes-edges-in-networkx?
        """  # noqa: E501

        label_ratio = 1.0 / 20.0
        pos_labels = {}

        G = self.graph

        # For each node in the Graph
        for aNode in G.nodes():

            # Get the node's position from the layout
            x, y = pos[aNode]

            # Get the node's neighbourhood
            N = G[aNode]

            # Find the centroid of the neighbourhood. The centroid is the average of the Neighbourhood's node's x and y
            # coordinates respectively.
            # Please note: This could be optimised further

            cx = sum(map(lambda x: pos[x][0], N)) / len(pos)
            cy = sum(map(lambda x: pos[x][1], N)) / len(pos)

            # Get the centroid's 'direction' or 'slope'. That is, the direction TOWARDS the centroid FROM aNode.
            slopeY = (y - cy)
            slopeX = (x - cx)
            # Position the label at some distance along this line. Here, the label is positioned at about 1/8th of the
            # distance.

            pos_labels[aNode] = (x + slopeX * label_ratio, y + slopeY * label_ratio)

        return pos_labels

    def get_setting_region_and_number(self, idx):
        """
            For a graph node with index idx that is a setting, work out the region it belongs to, and its number
            as a setting in that region
        """
        if idx < self.number_of_regions:
            # this is a result, not a setting, so the region is just idx
            region = idx
            setting_number = 0
            print("Warning: Idx corresponds to a result node index")
        else:

            # compute the region number and setting number for idx node
            for region in range(self.number_of_regions):
                if idx in range(self.setting_indices[region],
                                (self.setting_indices[(region + 1)] if region < (self.number_of_regions - 1)
                                 else self.number_of_columns)):
                    break
                setting_number = idx - self.setting_indices[region]

        return region, setting_number

    def show_tvd_explanations(self):

        g = self.graph

        for idx, edge in enumerate(g.edges()):

            source = edge[0]
            dest = edge[1]

            if self.is_edge_ct[idx] == 1:
                if idx in self.max_tvd_explanations.keys():
                    if (source >= self.number_of_regions) and (dest < self.number_of_regions):
                        region, setting_number = self.get_setting_region_and_number(source)

                        print("------ Edge (S_{}^({}) to R_{}) ------".format(region, setting_number, dest))
                        print(self.max_tvd_explanations[idx])
                    else:
                        print(("Source is not setting and destination is not results. "
                               "Not sure why there is a TVD explanation here"))
