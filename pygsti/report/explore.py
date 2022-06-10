from kivy.app import App

from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.splitter import Splitter
from kivy.uix.dropdown import DropDown
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.spinner import Spinner

from kivy.properties import ObjectProperty, StringProperty
from kivy.graphics import Color, Rectangle

from .kivygraph import Graph, MeshLinePlot, BarPlot, MatrixBoxPlotGraph

import pygsti
from pygsti.report import Workspace
from pygsti.protocols.gst import ModelEstimateResults as _ModelEstimateResults
from pygsti.io import read_results_from_dir as _read_results_from_dir


class RootExplorerWidget(BoxLayout):

    def __init__(self, results_dir, **kwargs):
        # make sure we aren't overriding any important functionality
        kwargs['orientation'] = 'vertical'  # self == top 'mode' bar and lower area
        super(RootExplorerWidget, self).__init__(**kwargs)

        top_bar = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        top_bar.add_widget(Button(text="Processor Spec"))
        top_bar.add_widget(Button(text="Experiment Design"))
        top_bar.add_widget(Button(text="Analysis"))

        sidebar = BoxLayout(orientation='vertical') #, size_hint_x=None, width=100)  # ResultsSelectorWidget??
        edesign_selector = ExperimentDesignSelectorWidget(results_dir)
        results_selector = ResultsSelectorWidget(edesign_selector)
        resultdetail_selector = ResultDetailSelectorWidget(results_selector)
        sidebar.add_widget(edesign_selector)
        sidebar.add_widget(results_selector)
        sidebar.add_widget(resultdetail_selector)

        sidebar_splt = Splitter(sizable_from='right', size_hint_x=None, width=100)
        sidebar_splt.add_widget(sidebar)
        sidebar_splt.min_size = 500

        sidebar_splt.max_size = 1000

        dataarea = DataAreaWidget(results_selector, resultdetail_selector)
        #GridLayout(cols=1, rows=1)

        under_top_bar = BoxLayout(orientation='horizontal')
        under_top_bar.add_widget(sidebar_splt)
        under_top_bar.add_widget(dataarea)

        bottom_bar = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        bottom_bar.add_widget(Label(text="Status text goes here..."))

        self.add_widget(top_bar)
        self.add_widget(under_top_bar)
        self.add_widget(bottom_bar)

        ## let's add a Widget to this layout
        #self.add_widget(
        #    Button(
        #        text="Hello World",
        #        size_hint=(.5, .1),
        #        pos_hint={'center_x': .5, 'top': 1.0}))


        #results = self.results_dir.for_protocol['GateSetTomography']  # HACK for now... (for testing) - in future show list of protocols and subdirs...

        #TEST1
        #ws = Workspace()
        ##wstable = ws.SoftwareEnvTable()
        #tbl = wstable.tables[0]
        #out = tbl.render('kivywidget')
        #tblwidget = out['kivywidget']
        #dataarea.data_area.add_widget(tblwidget)

        #TEST2 - kivy plot
        from math import sin
        from kivy.utils import get_color_from_hex as rgb
        graph_theme = {
                'label_options': {
                    'color': rgb('444444'),  # color of tick labels and titles
                    'bold': True},
                'background_color': rgb('f8f8f2'),  # canvas background color
                'tick_color': rgb('808080'),  # ticks and grid
                'border_color': rgb('808080')}  # border drawn around each graph

        #graph = Graph(xlabel='X', ylabel='Y',
        #              x_ticks_major=25, x_ticks_minor=5,
        #              y_ticks_major=1.0, y_ticks_minor=2,
        #              y_grid_label=True, x_grid_label=True, padding=5,
        #              x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1,
        #              **graph_theme)
        #plot = MeshLinePlot(color=[1, 0, 0, 1])
        #plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]

        #graph = Graph(xlabel='X', ylabel='Y',
        #              x_ticks_major=10, x_ticks_minor=2,
        #              y_ticks_major=1.0, y_ticks_minor=2,
        #              y_grid_label=True, x_grid_label=True, padding=5,
        #              x_grid=True, y_grid=True, xmin=-0, xmax=70, ymin=0, ymax=1,
        #              x_grid_labels=["One", "Two", "Three", "4", "5", "6", "7", "-8-", "nine", "ten"],
        #              x_ticks_angle=45,
        #              **graph_theme)
        #plot = BarPlot(color=(0,0,1,1), bar_spacing=.5)
        #graph.add_plot(plot)
        #plot.bind_to_graph(graph)
        #plot.points = [(30, 0.1), (40, 0.2), (50, 0.3), (60, 0.4)]

        #graph = Graph(xlabel='X', ylabel='Y',
        #              x_ticks_major=25, x_ticks_minor=5,
        #              y_ticks_major=1.0, y_ticks_minor=2,
        #              y_grid_label=True, x_grid_label=True, padding=5,
        #              x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1,
        #              **graph_theme)
        #plot = MeshBoxPlot(color=[1, 0, 0, 1])
        #plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]

        #graph.add_plot(plot)
        #dataarea.data_area.add_widget(graph)

        #import numpy as np
        #randmx = np.random.random((6,6))
        #print("Random mx =\n",randmx)
        #graph = MatrixBoxPlotGraph(randmx, xlabel='myX', ylabel='myY', padding=5, background_color=(1,0,0,1),
        #                          x_ticks_major=1.0, y_ticks_major=1.0, x_tick_offset=0.5, y_tick_offset=0.5,
        #                          x_grid_labels=["A", "B", "C", "D", "E", "F"], x_grid_label=True,
        #                          y_grid_labels=list(reversed(["a", "b", "c", "d", "e", "f"])), y_grid_label=True)
        #                          #x_ticks_angle=45,
        #                          #**graph_theme)
        #dataarea.data_area.add_widget(graph)

        
        #TEST3
        #pltwidget = PlotWidget()
        #pltwidget = 


        #self.add_widget(test_widget)
        #self.add_widget(TableWidget(2,2, size_hint=(1.0,0.1 * 2), pos_hint={'x': 0.0, 'top': 0.9}))


class BarPlotWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas:
            self._fbo = Fbo(
                size=self.size, with_stencilbuffer=True)



class HistogramPlotWidget(Widget):
    pass
    

class BoxPlotWidget(Widget):
    pass

class TreeViewLabelWithData(TreeViewLabel):
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data


class ExperimentDesignSelectorWidget(BoxLayout):
    # allows selection of edesign (tree node) and results object (dataset) at tree node.
    selected_results_dir = ObjectProperty(None, allownone=True)

    def __init__(self, root_results_dir, **kwargs):
        super().__init__(**kwargs)
        self.root_results_dir = root_results_dir
        tv = TreeView(root_options=dict(text='Root <filename>'),
                      hide_root=False,
                      indent_level=4)

        def populate_view(results_dir, parent_tree_node):
            for ky in results_dir.keys():
                tree_node = tv.add_node(TreeViewLabelWithData(data=results_dir[ky], text=str(ky), is_open=True),
                                        parent=parent_tree_node)  # OK if None
                populate_view(results_dir[ky], tree_node)

        populate_view(root_results_dir, None)
        tv.bind(_selected_node=self.on_change_selected_node)

        self.add_widget(tv)
        tv.select_node(tv.get_root())

    def on_change_selected_node(self, instance, new_node):
        print("Selected edesign: " + new_node.text)  # (new_node is a TreeViewLabel)
        self.selected_results_dir = new_node.data if hasattr(new_node, 'data') else self.root_results_dir


class ResultsSelectorWidget(BoxLayout):
    # allows selection of edesign (tree node) and results object (dataset) at tree node.
    selected_results = ObjectProperty(None, allownone=True)

    def __init__(self, results_dir_selector_widget, **kwargs):
        super().__init__(**kwargs)
        self.treeview = None
        if results_dir_selector_widget.selected_results_dir is not None:
            self.refresh_tree(results_dir_selector_widget.selected_results_dir)                
        results_dir_selector_widget.bind(selected_results_dir=self.on_change_selected_results_dir)

        #Select the first selectable node
        if self.treeview is not None:
            if len(self.treeview.get_root().nodes) > 0:
                self.treeview.select_node(self.treeview.get_root().nodes[0])

    def refresh_tree(self, current_results_dir):
        tv = TreeView(root_options=dict(text='Results for protocol:'),
                      hide_root=False,
                      indent_level=4)

        def populate_view(results_dir, parent_tree_node):
            for protocol_name, protocol_results in results_dir.for_protocol.items():
                nd = tv.add_node(TreeViewLabelWithData(data=protocol_results, text=str(protocol_name),
                                                       is_open=True), parent=parent_tree_node)  # OK if None

        populate_view(current_results_dir, None)
        tv.bind(_selected_node=self.on_change_selected_node)

        self.treeview = tv
        self.clear_widgets()
        self.add_widget(tv)

    def on_change_selected_node(self, instance, new_node):
        if new_node.is_leaf:
            print("Selected Results: " + new_node.text)  # (new_node is a TreeViewLabel)
            self.selected_results = new_node.data if hasattr(new_node, 'data') else None
        else:
            print("Root result item selected (?)")

    def on_change_selected_results_dir(self, instance, new_results_dir):
        self.refresh_tree(new_results_dir)


class ResultDetailSelectorWidget(BoxLayout):
    estimate_name = StringProperty(None, allownone=True)
    model_name = StringProperty(None, allownone=True)

    # allows selection of model, gaugeopt, etc.
    def __init__(self, results_selector_widget, **kwargs):
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)
        self.results = None  # the current results object
        self.rows = []  # list of horizontal BoxLayouts, one per row
        if results_selector_widget.selected_results is not None:
            self.on_change_selected_results(None, results_selector_widget.selected_results)
        results_selector_widget.bind(selected_results=self.on_change_selected_results)

    def rebuild(self):
        self.clear_widgets()
        for row in self.rows:
            self.add_widget(row)

    def on_change_selected_results(self, instance, new_results_obj):
        self.rows.clear()
        self.results = new_results_obj  # make this the current results object

        if isinstance(new_results_obj, _ModelEstimateResults):
            estimate_row = BoxLayout(orientation='horizontal')
            estimate_keys = list(new_results_obj.estimates.keys())
            estimate_spinner = Spinner(text=estimate_keys[0] if (len(estimate_keys) > 0) else '(none)',
                                       values=estimate_keys, size_hint=(0.6, 1.0))
            estimate_spinner.bind(text=self.on_change_selected_estimate)
            estimate_row.add_widget(Label(text='Estimate:', size_hint=(0.4, 1.0)))
            estimate_row.add_widget(estimate_spinner)
            self.rows.append(estimate_row)
            self.on_change_selected_estimate(None, estimate_keys[0] if (len(estimate_keys) > 0) else '(none)')
        else:
            self.rows.append(Label(text='No details'))
            self.rebuild()

    def on_change_selected_estimate(self, spinner, new_estimate_key):
        #Note: this is only called when self.results is a ModelEstimateResults object
        if len(self.rows) > 1:
            self.remove_widget(self.rows[1])  # remove second row == "Model: ..." row

        self.estimate_name = new_estimate_key
        if new_estimate_key is not None:
            estimate = self.results.estimates[new_estimate_key]
            model_names = list(estimate.models.keys())
        else:
            model_names = []

        model_row = BoxLayout(orientation='horizontal')
        model_spinner = Spinner(text=model_names[0] if (len(model_names) > 0) else '(none)',
                                values=model_names, size_hint=(0.6, 1.0))
        model_spinner.bind(text=self.on_change_selected_model)
        model_row.add_widget(Label(text='Model:', size_hint=(0.4, 1.0)))
        model_row.add_widget(model_spinner)
        self.rows.append(model_row)
        self.rebuild()
        self.on_change_selected_model(None, model_names[0] if (len(model_names) > 0) else None)

    def on_change_selected_model(self, spinner, new_model_name):
        self.model_name = new_model_name  # set property so other layouts can trigger off of?


class DataAreaWidget(BoxLayout):
    # needs menus of all available tables/plots to add (for currently selected results/model/data/gaugeopt, etc)
    def __init__(self, results_selector, resultdetail_selector, **kwargs):
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

        self.results_selector_widget = results_selector
        self.resultdetail_selector_widget = resultdetail_selector

        self.results_selector_widget.bind(selected_results=self.selection_change)
        self.resultdetail_selector_widget.bind(estimate_name=self.selection_change, model_name=self.selection_change)

        add_dropdown = DropDown()  # size_hint=(0.8, 1.0)
        possible_items_to_add = [
            'SpamTable', 'SpamParametersTable', 'GatesTable', 'ChoiTable', 'ModelVsTargetTable',
            'GatesVsTargetTable', 'SpamVsTargetTable', 'ErrgenTable', 'NQubitErrgenTable', 'GateDecompTable',
            'GateEigenvalueTable', 'DataSetOverviewTable', 'FitComparisonTable', 'CircuitTable', 'GatesSingleMetricTable',
            'StandardErrgenTable', 'GaugeOptParamsTable', 'MetadataTable', 'SoftwareEnvTable', 'WildcardBudgetTable',
            'ExampleTable', 'ColorBoxPlot', 'FitComparisonBarPlot', 'FitComparisonBoxPlot']
            # 'GateMatrixPlot', 'MatrixPlot', DatasetComparisonHistogramPlot, RandomizedBenchmarkingPlot
            # GaugeRobustModelTable, GaugeRobustMetricTable, GaugeRobustErrgenTable, ProfilerTable

        for item_name in possible_items_to_add:
            btn = Button(text=item_name, size_hint_y=None, height=40)  # must specify height manually
            btn.bind(on_release=lambda btn: add_dropdown.select(btn.text))  # pressing button selects dropdown
            btn.bind(on_press=lambda btn: self.add_item(btn.text))  # pressing button selects dropdown
            add_dropdown.add_widget(btn)  # add the button inside the dropdown
        add_dropdown_mainbutton = Button(text='Add New', size_hint=(1.0, 1.0))

        # show the dropdown menu when the main button is released
        # note: all the bind() calls pass the instance of the caller (here, the
        # mainbutton instance) as the first argument of the callback (here, dropdown.open.).
        add_dropdown_mainbutton.bind(on_release=lambda instance: add_dropdown.open(instance))
        # NOTE: using on_release=add_dropdown.open doesn't work for some reason, despite tutorial instruction.

        # listen for the selection in the dropdown list and assign the data to the button text.
        add_dropdown.bind(on_select=lambda instance, x: setattr(add_dropdown_mainbutton, 'text', x))

        add_item_bar = BoxLayout(orientation='horizontal', height=40, size_hint_y=None)
        #add_item_bar.add_widget(Label(text='Add new:', size_hint=(0.2, 1.0)))
        add_item_bar.add_widget(add_dropdown_mainbutton)

        self.data_area = StackLayout(orientation='lr-tb')
        self.add_widget(add_item_bar)
        self.add_widget(self.data_area)

    def selection_change(self, instance, value):
        print("Data area noticed a selected results or model change... do something in the future?")

    def add_item(self, item_text):
        print("Adding item ", item_text)

        results = self.results_selector_widget.selected_results
        if isinstance(results, _ModelEstimateResults):
            estimate = results.estimates[self.resultdetail_selector_widget.estimate_name]
            model = estimate.models[self.resultdetail_selector_widget.model_name]
        else:
            estimate = model = None

        ws = Workspace()
        if item_text == 'SpamTable':
            wstable = ws.SpamTable([model])
        elif item_text == 'SoftwareEnvTable':
            wstable = ws.SoftwareEnvTable()
        else:
            wstable = None

        if wstable is not None:
            tbl = wstable.tables[0]
            out = tbl.render('kivywidget')
            tblwidget = out['kivywidget']
            self.data_area.clear_widgets()
            self.data_area.add_widget(tblwidget)
        else:
            print("Cannot create " + item_text + " yet.")
            
        #possible_items_to_add = [
        #    'SpamTable', 'SpamParametersTable', 'GatesTable', 'ChoiTable', 'ModelVsTargetTable',
        #    'GatesVsTargetTable', 'SpamVsTargetTable', 'ErrgenTable', 'NQubitErrgenTable', 'GateDecompTable',
        #    'GateEigenvalueTable', 'DataSetOverviewTable', 'FitComparisonTable', 'CircuitTable', 'GatesSingleMetricTable',
        #    'StandardErrgenTable', 'GaugeOptParamsTable', 'MetadataTable', 'SoftwareEnvTable', 'WildcardBudgetTable',
        #    'ExampleTable', 'ColorBoxPlot', 'FitComparisonBarPlot', 'FitComparisonBoxPlot']

        
        #results = self.results_dir.for_protocol['GateSetTomography']  # HACK for now... (for testing) - in future show list of protocols and subdirs...




class DataTableWidget(FloatLayout):
    pass  # maybe takes a single child and has a remove button?

#class ModelViolationWidget(FloatLayout):


#class TableWidget(GridLayout):
#    def __init__(self, num_rows, num_cols, **kwargs):
#        super(TableWidget, self).__init__(**kwargs)
#        self.rows = num_rows
#        self.cols = num_cols
#        self.padding = 2
#        self.spacing = 5
#
#        for i in range(self.rows):
#            for j in range(self.cols):
#                l = Label(text='%d,%d' % (i, j))
#                l.color = (0,0,0,1)  # black text
#                self.add_widget(l)
#
#        #self.username = TextInput(multiline=False)
#        #self.add_widget(self.username)
#        #self.add_widget(Label(text='password'))
#        #self.password = TextInput(password=True, multiline=False)
#        #self.add_widget(self.password)
#
#        #self.on_size()
#
#    def on_size(self, *args):
#        self.canvas.before.clear()
#        with self.canvas.before:
#            Color(0.5, 0.5, 0.5)  # table background
#            Rectangle(pos=self.pos, size=self.size)


class DataExplorerApp(App):
    def __init__(self, results_dir):  #, test_widget):
        self.results_dir = results_dir        
        #self.test_widget = test_widget
        super().__init__()

    def build(self):
        return RootExplorerWidget(self.results_dir)


def build_dropdown(options, option_height=40, size_hint=(None, None)):
    dropdown = DropDown()  # size_hint=(0.8, 1.0)
    for option_text in options:
        btn = Button(text=option_text, size_hint_y=None, height=option_height)  # must specify height manually
        btn.bind(on_release=lambda btn: dropdown.select(btn.text))  # pressing button selects dropdown
        dropdown.add_widget(btn)  # add the button inside the dropdown
    dropdown_mainbutton = Button(text=options[0] if (len(options) > 0) else '<empty>', size_hint=size_hint)

    # show the dropdown menu when the main button is released
    # note: all the bind() calls pass the instance of the caller (here, the
    # mainbutton instance) as the first argument of the callback (here, dropdown.open.).
    dropdown_mainbutton.bind(on_release=lambda instance: dropdown.open(instance))
    # NOTE: using on_release=dropdown.open doesn't work for some reason, despite tutorial instruction.

    # listen for the selection in the dropdown list and assign the data to the button text.
    dropdown.bind(on_select=lambda instance, x: setattr(dropdown_mainbutton, 'text', x))
    return dropdown, dropdown_mainbutton


if __name__ == '__main__':
    ws = Workspace()
    wstable = ws.SoftwareEnvTable()
    tbl = wstable.tables[0]
    out = tbl.render('kivywidget')
    tblwidget = out['kivywidget']
    #import bpdb; bpdb.set_trace()
    #print(tbl)

    DataExplorerApp(tblwidget).run()
