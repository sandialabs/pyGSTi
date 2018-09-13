from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for crosstalk detection from time-stamped data"""

import numpy as _np

class CrosstalkResults(object):

    def __init__(self):
        
        #--------------------------#
        # --- Input quantities --- #
        #--------------------------#
        
        self.name = None
        self.data = None
        self.number_of_qubits = None
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

        

    def any_crosstalk_detect(self):
        
        if self.crosstalk_detected:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence)) 
            print("Result: The 'no crosstalk' hypothesis *is* rejected.")
        else:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence))
            print("Result: The 'no crosstalk' hypothesis is *not* rejected.")
    
    
    def plot_crosstalk_matrices(self, figsize=(15,3), savepath=None):
        """

        """
       
        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_crosstalk_matrix(...) requires you to install matplotlib")
        from mpl_toolkits.axes_grid1 import make_axes_locatable
 #       import matplotlib.ticker as ticker
 #       from mpl_toolkits.axes_grid.parasite_axes import SubplotHost


        # fig = _plt.figure()
        # ax1 = SubplotHost(fig, 1,2,1)
        # ax2 = SubplotHost(fig, 1,2,2)
        # fig.add_subplot(ax1)
        # fig.add_subplot(ax2)

        fig, (ax1, ax2) = _plt.subplots(1,2,figsize=(sum(self.settings)+self.number_of_qubits+6, self.number_of_qubits+4))
        fig.subplots_adjust(wspace=2, hspace=2)

        if self.name is not None:
            title = 'Crosstalk matrices for dataset '+self.name+'. Confidence level '+str(self.confidence)
        else:
            title = 'Crosstalk matrices for dataset. Confidence level '+str(self.confidence)

        # common arguments to imshow
        kwargs = dict(
            origin='lower', interpolation='nearest', vmin=0, vmax=1, aspect='equal', cmap='YlOrBr')

        settings_and_qubits = _np.zeros((sum(self.settings), self.number_of_qubits))
        qubits_and_qubits = _np.zeros((self.number_of_qubits, self.number_of_qubits))

        for idx, edge in enumerate(self.graph.edges()) :
            source = edge[0]
            dest = edge[1]

            # edge between two outcomes
            if source < self.number_of_qubits and dest < self.number_of_qubits:
                qubits_and_qubits[source, dest] = _np.max(self.edge_tvds[idx])

            # edge between an outcome and a setting
            if source < self.number_of_qubits and dest >= self.number_of_qubits:
                if dest not in range(self.setting_indices[source], (self.setting_indices[(source + 1)] if source < (self.number_of_qubits - 1) else self.number_of_columns)):
                    settings_and_qubits[dest-self.number_of_qubits, source] = _np.max(self.edge_tvds[idx])

            # edge between an outcome and a setting
            if source >= self.number_of_qubits and dest < self.number_of_qubits:
                if source not in range(self.setting_indices[dest], (self.setting_indices[(dest + 1)] if dest < (self.number_of_qubits - 1) else self.number_of_columns)):
                    settings_and_qubits[source-self.number_of_qubits, dest] = _np.max(self.edge_tvds[idx])

        ax1.imshow(settings_and_qubits, **kwargs)
        _plt.setp(ax1, xticks=_np.arange(0, sum(self.settings), 1),
                 xticklabels= [self.node_labels[k] for k in range(self.number_of_qubits,self.number_of_columns)],
                 yticks=_np.arange(0, self.number_of_qubits, 1),
                 yticklabels=_np.arange(0, self.number_of_qubits, 1).astype('str'))


        dividers = [sum(self.settings[:k])-0.5 for k in range(1,self.number_of_qubits)]
        for i in range(len(dividers)) :
            ax1.axvline(dividers[i], color='k')

        ax1.set_xlabel('Settings')
        ax1.set_ylabel('Qubit outcomes')
        ax1.set_title('Crosstalk between qubit outcomes and settings')

  #       ax1a = ax1.twiny()
  #       offset = 0, -25  # Position of the second axis
  #       new_axisline = ax1a.get_grid_helper().new_fixed_axis
  #       ax1a.axis["bottom"] = new_axisline(loc="bottom", axes=ax1a, offset=offset)
  #       ax1a.axis["top"].set_visible(False)
  #
  #       dividers.insert(0,0.0)
  #       dividers.append(sum(self.settings)+0.5)
  #       ax1a.set_xticks(dividers)
  #       ax1a.xaxis.set_major_formatter(ticker.NullFormatter())
  #       ax1a.xaxis.set_minor_locator(ticker.FixedLocator([0.3, 0.8]))
  #       ax1a.xaxis.set_minor_formatter(ticker.FixedFormatter(['mammal', 'reptiles']))

        im = ax2.imshow(qubits_and_qubits, **kwargs)
        _plt.setp(ax2, xticks=_np.arange(0, self.number_of_qubits, 1),
                 xticklabels=_np.arange(0, self.number_of_qubits, 1).astype('str'),
                 yticks=_np.arange(0, self.number_of_qubits, 1),
                 yticklabels=_np.arange(0, self.number_of_qubits, 1).astype('str'))
        ax2.set_xlabel('Qubit outcomes')
        ax2.set_ylabel('Qubit outcomes')
        ax2.set_title('Crosstalk between qubit outcomes')


        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical' )

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
        _plt.figure(figsize=(sum(self.settings)+2,6))

        if self.name is not None:
            title = 'Crosstalk graph for dataset ' + self.name + '. Confidence level ' + str(self.confidence)
        else:
            title = 'Crosstalk graph for dataset. Confidence level ' + str(self.confidence)

        # set positions for each node in graph
        G = self.graph
        pos = {}
        # settings are distributed along y=1 line
        pos.update( (n, (n-self.number_of_qubits, 1)) for n in range(self.number_of_qubits, self.number_of_columns) )

        # results are distributed along y=3 line
        for qubit in range(self.number_of_qubits) :
            num_settings_before = sum(self.settings[0:qubit])
            num_settings = self.settings[qubit]

            if num_settings == 1 :
                pos.update( {qubit: (num_settings_before, 3)} )
            else :
                pos.update( {qubit: (num_settings_before + (num_settings-1)/2, 3)} )

        # node colors
        settings_color = 'xkcd:light grey'
        outcomes_color = 'xkcd:light violet'

        # draw graph nodes
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_qubits), node_size=1000,
                                node_color=outcomes_color, node_shape='o',alpha=0.4)
        _nx.draw_networkx_nodes(G, pos, nodelist=range(self.number_of_qubits, self.number_of_columns), node_size=1000,
                                node_color=settings_color, node_shape='s',alpha=0.4)

        label_posns = self.get_offset_label_posns(pos)

        _nx.draw_networkx_labels(G, pos=label_posns, labels=self.node_labels)

        float_formatter = lambda x: "%.4f" % x

        # draw graph edge, with ones indicating crosstalk in red
        for idx, edge in enumerate(self.graph.edges()) :
            if self.is_edge_ct[idx] :
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='r')
                label = {}
                label[edge] = float_formatter(_np.max(self.edge_tvds[idx]) )
                _nx.draw_networkx_edge_labels(G,pos,edge_labels=label)
            else :
                _nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, alpha=1, edge_color='b')

        _plt.title(title, fontsize=17)
        _plt.axis('off')

        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()

    def get_offset_label_posns(self, pos):
        """
            From https://stackoverflow.com/questions/11946005/label-nodes-outside-with-minimum-overlap-with-other-nodes-edges-in-networkx?
        """

        label_ratio = 1.0 / 20.0
        pos_labels = {}

        G = self.graph

        # For each node in the Graph
        for aNode in G.nodes():

            # Get the node's position from the layout
            x, y = pos[aNode]

            # Get the node's neighbourhood
            N = G[aNode]

            # Find the centroid of the neighbourhood. The centroid is the average of the Neighbourhood's node's x and y coordinates respectively.
            # Please note: This could be optimised further

            cx = sum(map(lambda x: pos[x][0], N)) / len(pos)
            cy = sum(map(lambda x: pos[x][1], N)) / len(pos)

            # Get the centroid's 'direction' or 'slope'. That is, the direction TOWARDS the centroid FROM aNode.
            slopeY = (y - cy)
            slopeX = (x - cx)
            # Position the label at some distance along this line. Here, the label is positioned at about 1/8th of the distance.

            pos_labels[aNode] = (x + slopeX * label_ratio, y + slopeY * label_ratio)

        return pos_labels