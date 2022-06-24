import os as _os
from kivy.app import App

from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.splitter import Splitter
from kivy.uix.dropdown import DropDown
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.spinner import Spinner
from kivy.uix.behaviors import DragBehavior
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.modalview import ModalView
from kivy.uix.stencilview import StencilView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock

from kivy.properties import ObjectProperty, StringProperty, ListProperty, DictProperty
from kivy.graphics import Color, Rectangle, Line

from .kivyresize import ResizableBehavior
from .kivygraph import Graph, MeshLinePlot, BarPlot, MatrixBoxPlotGraph

import pygsti
from pygsti.report import Workspace
from pygsti.report.kivywidget import WrappedLabel
from pygsti.protocols.protocol import CircuitListsDesign as _CircuitListsDesign
from pygsti.protocols.estimate import Estimate as _Estimate
from pygsti.protocols.gst import ModelEstimateResults as _ModelEstimateResults
from pygsti.protocols.gst import StandardGSTDesign as _StandardGSTDesign
from pygsti.io import read_results_from_dir as _read_results_from_dir
from pygsti.objectivefns import objectivefns as _objfns


from kivy.core.window import Window
Window.size = (1200, 700)


def set_info_containers(root, sidebar, statusbar):
        """Walk down widget tree from `root` and call `set_info_containers` on applicable children. """
        if hasattr(root, 'set_info_containers'):
            root.set_info_containers(sidebar, statusbar)
        for c in root.children:
            set_info_containers(c, sidebar, statusbar)


class ClickableLabel(Label):
    __events__ = ('on_release', )

    def on_release(self, *largs):
        pass

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dispatch('on_release')
            return True
        super().on_touch_down(touch)


class EdesignLibElement(object):
    def __init__(self, name, protocol_data, root_file_path, path_from_root):
        self.name = name
        self.protocol_data = protocol_data
        self.root_file_path = root_file_path
        self.path_from_root = path_from_root # a list, e.g. ['CombinedDesign1', 'Q0']


class ModelLibElement(object):
    def __init__(self, name, model_name, model, model_container, root_file_path, path_from_root,
                 protocol_name, additional_details=None):
        self.name = name
        self.model = model
        self.model_name = model_name
        self.model_container = model_container
        self.root_file_path = root_file_path
        self.path_from_root = path_from_root # a list, e.g. ['CombinedDesign1', 'Q0']
        self.protocol_name = protocol_name
        self.additional_details = additional_details if (additional_details is not None) else {}


class FixedHeightLabel(Label):
    pass


class RootExplorerWidget(BoxLayout):

    active_figure_container = ObjectProperty(None, allownone=True)

    def __init__(self, results_dir_path=None, **kwargs):
        super().__init__(**kwargs)
        self._initial_results_dir_path = results_dir_path
        Clock.schedule_once(self.after_created, 0)
        self.mode = None
        self.edesign_library = {}
        self.model_library = {}
        self.ws = Workspace(gui_mode='kivy')
        self.default_figure_selector_vals = {}
        self.active_figure_selector_vals = {}

        #Bind to keyboard events
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def change_mode(self, new_mode):
        print("Changing mode from %s to %s" % (self.mode, new_mode))

        def show_sidebar(w):
            w.width = 500
            w.opacity = 1

        def hide_sidebar(w):
            w.width = 0
            w.opacity = 0

        if new_mode == 'Library':
            show_sidebar(self.ids.import_sidebar_splitter)
            show_sidebar(self.ids.library_sidebar_splitter)
            show_sidebar(self.ids.library_info_sidebar_splitter)
            hide_sidebar(self.ids.add_item_sidebar_splitter)
            # TODO hide(?) central area
            hide_sidebar(self.ids.figure_properties_sidebar_splitter)
            hide_sidebar(self.ids.figure_info_sidebar_splitter)

        elif new_mode == 'Create':
            hide_sidebar(self.ids.import_sidebar_splitter)
            hide_sidebar(self.ids.library_sidebar_splitter)
            hide_sidebar(self.ids.library_info_sidebar_splitter)
            show_sidebar(self.ids.add_item_sidebar_splitter)
            # TODO show central area
            show_sidebar(self.ids.figure_properties_sidebar_splitter)
            hide_sidebar(self.ids.figure_info_sidebar_splitter)
            self.set_active_figure_container(None)  # to prompt population of default arg panel

        elif new_mode == 'Analysis':
            hide_sidebar(self.ids.import_sidebar_splitter)
            hide_sidebar(self.ids.library_sidebar_splitter)
            hide_sidebar(self.ids.library_info_sidebar_splitter)
            hide_sidebar(self.ids.add_item_sidebar_splitter)
            # TODO show central area
            hide_sidebar(self.ids.figure_properties_sidebar_splitter)
            show_sidebar(self.ids.figure_info_sidebar_splitter)

        self.mode = new_mode

    def after_created(self, delta_time):
        print("Running post-kv-file creation of root widget.")
        #self.ids.create_tab.add_widget(self.create_add_item_panel(self.ids.center_area))
        self.ids.add_item_sidebar.add_widget(self.create_add_item_panel())
        set_info_containers(self.ids.center_area, self.ids.info_layout, self.ids.status_label)
        if self._initial_results_dir_path:
            results_dir = pygsti.io.read_results_from_dir(self._initial_results_dir_path)
            self.ids.results_dir_selector.root_name = _os.path.basename(self._initial_results_dir_path)
            self.ids.results_dir_selector.root_file_path = self._initial_results_dir_path
            self.ids.results_dir_selector.root_results_dir = results_dir  # automatic if we make self.results_dir a property
        self.change_mode('Library')

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        #print('The key', keycode, 'have been pressed')
        #print(' - text is %r' % text)
        #print(' - modifiers are %r' % modifiers)

        if keycode[1] == 'backspace' and self.active_figure_container is not None:
            self.ids.data_area.remove_widget(self.active_figure_container)
            self.active_figure_container = None
            return True

        # Return True to accept the key. Otherwise, it will be used by the system.
        return False

    def set_active_figure_container(self, active_figure_container):
        if self.active_figure_container is not None:
            self.active_figure_container.deactivate()

        if active_figure_container is not None:
            active_figure_container.activate()
            active_figure_container.capsule.populate_figure_property_panel(self.ids.figure_properties_sidebar)
        else:
            self.populate_figure_property_defaults_panel(self.ids.figure_properties_sidebar)
        self.active_figure_container = active_figure_container

    def populate_figure_property_defaults_panel(self, panel_widget):
        panel_widget.clear_widgets()
        panel_widget.add_widget(Label(text='Defaults', bold=True, size_hint_y=None, height=40))

        all_properties = ['*models', '*model_titles', '*model', '*model_title', '*target_model',
                          '*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl',
                          '*objfn_builder', '*gaugeopt_args', '*estimate_params']

        selector_types = self.selector_types_for_properties(all_properties)
        for typ in selector_types:
            self.add_figure_property_selector(typ, panel_widget,
                                              storage_dict=self.default_figure_selector_vals)

    def selector_types_for_properties(self, property_names):
        prop_set = set(property_names)
        dependencies = {  # include key if any of values (property template names) are present
            '**model': set(['*model', '*model_title', '*models', '*model_titles',
                            '*gaugeopt_args', '*estimate_params']),
            '**target_model': set(['*target_model']),
            '**edesign': set(['*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl']),
            '**objfn_builder': set(['*objfn_builder'])
        }

        selector_types = []
        for sel_typ, prop_names in dependencies.items():
            if prop_set.intersection(prop_names):
                selector_types.append(sel_typ)
        return selector_types

    def add_figure_property_selector(self, typ, panel_widget, storage_dict):
        initial_value = storage_dict.get(typ, None)
        row = BoxLayout(orientation='horizontal')
        if typ == '**model':
            row.add_widget(FixedHeightLabel(text='Model Title'))
            title_input = TextInput(text='', size_hint_y=None, height=40)
            title_input.bind(text=lambda inst, val: storage_dict.__setitem__('**model_title', val))
            storage_dict['**model_title'] = title_input.text
            row.add_widget(title_input)
            panel_widget.add_widget(row)

            row = BoxLayout(orientation='horizontal')
            row.add_widget(FixedHeightLabel(text='Model'))
            model_names = list(self.model_library.keys())
            if initial_value is None:
                initial_value = model_names[0] if (len(model_names) > 0) else '(none)'
            elif initial_value not in model_names:
                initial_value = "REMOVED!"
            spinner = Spinner(text=initial_value, values=model_names, size_hint=(0.6, 1.0))
            spinner.bind(text=lambda inst, val: storage_dict.__setitem__(typ, val))
            storage_dict[typ] = spinner.text

            row.add_widget(spinner)
            panel_widget.add_widget(row)

        elif typ == '**target_model':
            model_names = list(self.model_library.keys())
            row.add_widget(FixedHeightLabel(text='Target Model'))
            if initial_value is None:
                initial_value = model_names[0] if (len(model_names) > 0) else '(none)'
            elif initial_value not in model_names:
                initial_value = "REMOVED!"
            spinner = Spinner(text=initial_value, values=model_names, size_hint=(0.6, 1.0))
            spinner.bind(text=lambda inst, val: storage_dict.__setitem__(typ, val))
            storage_dict[typ] = spinner.text
            row.add_widget(spinner)
            panel_widget.add_widget(row)

        elif typ == '**edesign':
            edesign_names = list(self.edesign_library.keys())
            row.add_widget(FixedHeightLabel(text='Edesign'))
            if initial_value is None:
                initial_value = edesign_names[0] if (len(edesign_names) > 0) else '(none)'
            elif initial_value not in edesign_names:
                initial_value = "REMOVED!"
            spinner = Spinner(text=initial_value, values=edesign_names, size_hint=(0.6, 1.0))
            spinner.bind(text=lambda inst, val: storage_dict.__setitem__(typ, val))
            storage_dict[typ] = spinner.text
            row.add_widget(spinner)
            panel_widget.add_widget(row)
            
        elif typ == '**objfn_builder':
            objfn_builder_names = ['logl', 'chi2', 'from estimate']
            row.add_widget(FixedHeightLabel(text='Objective Fn.'))
            if initial_value is None:
                initial_value = objfn_builder_names[0]
            spinner = Spinner(text=initial_value, values=objfn_builder_names, size_hint=(0.6, 1.0))
            spinner.bind(text=lambda inst, val: storage_dict.__setitem__(typ, val))
            storage_dict[typ] = spinner.text
            row.add_widget(spinner)
            panel_widget.add_widget(row)

        else:
            raise ValueError("Unknown figure property selector type: %s" % str(typ))

    def selector_values_to_creation_args(self, figure_selector_vals):
        creation_args = {}
        #all_properties = ['*models', '*model_titles', '*model', '*model_title', '*target_model',
        #                  '*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl',
        #                  '*objfn_builder', '*gaugeopt_args', '*estimate_params']
        
        for typ, val in figure_selector_vals.items():
            if typ == '**model_title':
                if val:
                    creation_args['*model_title'] = val
            if typ == '**model':
                creation_args['*model'] = self.model_library[val].model
                if '*model_title' not in creation_args:
                    creation_args['*model_title'] = self.model_library[val].model_name
                if isinstance(self.model_library[val].model_container, _Estimate):
                    estimate = self.model_library[val].model_container
                    creation_args['*estimate_params'] = estimate.parameters
                    creation_args['*gaugeopt_args'] = estimate.goparameters.get(self.model_library[val].model_name, {})
            if typ == '**target_model':
                creation_args['*target_model'] = self.model_library[val].model
            if typ == '**edesign':
                protocol_data = self.edesign_library[val].protocol_data
                creation_args['*edesign'] = protocol_data.edesign
                creation_args['*dataset'] = protocol_data.dataset
                creation_args['*circuit_list'] = protocol_data.edesign.all_circuits_needing_data                
                creation_args['*circuit_lists'] = protocol_data.edesign.circuit_lists \
                    if isinstance(protocol_data.edesign, _CircuitListsDesign) else None
                if isinstance(protocol_data.edesign, _StandardGSTDesign):
                    creation_args['*maxlengths'] = protocol_data.edesign.maxlengths
                    creation_args['*circuits_by_maxl'] = protocol_data.edesign.circuit_lists
            if typ == '**objfn_builder':
                if val == 'from estimate':
                    if '**model' in figure_selector_vals:
                        k = figure_selector_vals['**model']
                        mdl_container = self.model_library[k].model_container
                        if isinstance(mdl_container, _Estimate):
                            creation_args['*objfn_builder'] = mdl_container.parameters.get(
                                'final_objfn_builder', _objfns.ObjectiveFunctionBuilder.create_from('logl'))
                    if '*objfn_builder' not in creation_args:
                        print("Warning: could not retrieve objective function from estimate -- using logL instead")
                        creation_args['*objfn_builder'] = _objfns.ObjectiveFunctionBuilder.create_from('logl')
                else:
                    creation_args['*objfn_builder'] = _objfns.ObjectiveFunctionBuilder.create_from(val)            

        if '*model' in creation_args and '*models' not in creation_args:
            creation_args['*models'] = [creation_args['*model']]
        if '*model_title' in creation_args and '*model_titles' not in creation_args:
            creation_args['*model_titles'] = [creation_args['*model_title']]

        return creation_args

    def create_add_item_panel(self):

        items_by_category = {
            '-- Model Violation --': ['FitComparisonTable', 'ColorBoxPlot', 'ColorScatterPlot', 'ColorHistogramPlot',
                                      'FitComparisonBarPlot', 'FitComparisonBoxPlot'],
            '-- G. Inv. Metrics --': ['SpamParametersTable', 'GateEigenvalueTable', 'ModelVsTargetTable',
                                      'WildcardBudgetTable'],
            '-- Metrics --': ['GatesVsTargetTable', 'SpamVsTargetTable', 'SpamTable', 'GatesTable', 'ChoiTable',
                              'ErrgenTable', 'NQubitErrgenTable', 'GateDecompTable', 'GatesSingleMetricTable'],
            '-- Reference --': ['CircuitTable', 'DataSetOverviewTable', 'StandardErrgenTable', 'GaugeOptParamsTable',
                                'MetadataTable', 'SoftwareEnvTable'],
        }
        # 'GateMatrixPlot', 'MatrixPlot', DatasetComparisonHistogramPlot, RandomizedBenchmarkingPlot
        # GaugeRobustModelTable, GaugeRobustMetricTable, GaugeRobustErrgenTable, ProfilerTable

        first_child = None
        ret = Accordion(orientation='vertical', height=1000)  # height= just to try to supress initial warning
        for category_name, item_list in items_by_category.items():
            acc_item = CustomAccordionItem(title=category_name)
            acc_item_layout = BoxLayout(orientation='vertical')
            acc_item.add_widget(acc_item_layout)
            for item_name in item_list:
                btn = Button(text=item_name, size_hint_y=None, height=50)  # must specify height manually
                btn.bind(on_press=lambda btn: self.add_item(btn.text))  # pressing button fires add_item
                acc_item_layout.add_widget(btn)
            acc_item_layout.add_widget(Label(text=''))  # blank variable-height label so buttons are at top of BoxLayout
            ret.add_widget(acc_item)
            if first_child is None: first_child = acc_item

        ret.select(first_child)
        return ret

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        print("TODO: load root: ", path, filename)
        self.dismiss_popup()
        root_path = _os.path.join(path, filename)
        self.results_dir = pygsti.io.read_results_from_dir(root_path)

        self.ids.results_dir_selector.root_name = filename  # no way to input this separately from the popup yet
        self.ids.results_dir_selector.root_file_path = root_path
        self.ids.results_dir_selector.root_results_dir = self.results_dir  # automatic if we make self.results_dir a property

    def import_edesign(self, include_children=False):
        root_name = self.ids.results_dir_selector.root_name
        results_dir_node = self.ids.results_dir_selector.selected_results_dir_node
        root_file_path = self.ids.results_dir_selector.root_file_path
        path_from_root = results_dir_node.path
        pth_for_name = [root_name] + results_dir_node.path if root_name else results_dir_node.path
        name = '.'.join(pth_for_name)  # edesign name

        results_dir = results_dir_node.data
        data = results_dir.data  # a ProtocolData object
        if include_children:
            print("Importing with children isn't implemented yet!!!")
            pass  # TODO: walk down tree adding children too

        btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        btn.bind(state=self.update_library_item_info)
        if name in self.edesign_library:
            print("Edesign %s is already imported." % name); return
        self.edesign_library[name] = EdesignLibElement(name, data, root_file_path, path_from_root)
        self.ids.edesign_library_list.add_widget(btn)
        print("Imported edesign: ", name)

    def import_models(self):
        root_name = self.ids.results_dir_selector.root_name
        results_dir_node = self.ids.results_dir_selector.selected_results_dir_node
        root_file_path = self.ids.results_dir_selector.root_file_path
        path_from_root = results_dir_node.path
        protocol_name = self.ids.results_dir_detail_selector.protocol_name
        additional_detail = self.ids.results_dir_detail_selector.additional_detail
        selected_model_names = self.ids.results_dir_detail_selector.selected_model_names
        pth_for_name = [root_name] + results_dir_node.path if root_name else results_dir_node.path

        results_dir = results_dir_node.data
        results = results_dir.for_protocol[protocol_name]

        if 'estimate_name' in additional_detail:
            estimate = results.estimates[additional_detail['estimate_name']]
            model_container = estimate.models
        else:
            model_container = results.models

        for model_name in selected_model_names:
            name = '.'.join(pth_for_name + [protocol_name] + list(additional_detail.values()) + [model_name])
            model = model_container[model_name]
            if name in self.model_library:
                print("Model %s is already imported." % name); continue
            self.model_library[name] = ModelLibElement(name, model_name, model, model_container,
                                                       root_file_path, path_from_root,
                                                       protocol_name, additional_detail)

            btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
            btn.bind(state=self.update_library_item_info)
            self.ids.model_library_list.add_widget(btn)
                #Label(text=name, size_hint_y=None, height=40))
            print("Importing model: ", model_name)

    def import_all(self):
        print("Importing all models")

    def update_library_item_info(self, togglebtn, val):
        info_area = self.ids.library_info_area
        info_area.clear_widgets()

        def add_info_row(widget, k, v):
            row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            key_lbl = Label(text=str(k), font_size=24, size_hint_y=None, height=40)
            key_anchor = AnchorLayout(anchor_x='center', anchor_y='center'); key_anchor.add_widget(key_lbl)
            row.add_widget(key_anchor)
            value_lbl = WrappedLabel(text=str(v), font_size=24)
            value_lbl.bind(height=lambda _, h: setattr(row, 'height', h))  # so row height adjusts with label
            row.add_widget(value_lbl)
            widget.add_widget(row)

        if val == 'down':
            key = togglebtn.text
            if key in self.edesign_library:
                item = self.edesign_library[key]
                add_info_row(info_area, 'name:', item.name)
                add_info_row(info_area, 'root path:', item.root_file_path)
                add_info_row(info_area, 'path from root:', str(item.path_from_root))
            elif key in self.model_library:
                item = self.model_library[key]
                add_info_row(info_area, 'name:', item.name)
                add_info_row(info_area, 'root path:', item.root_file_path)
                add_info_row(info_area, 'path from root:', str(item.path_from_root))
                add_info_row(info_area, 'protocol:', str(item.protocol_name))
                for k, v in item.additional_details.items():
                    add_info_row(info_area, k, str(v))
            #print("Down from ", togglebtn.text)
        else:
            #print(val, " from ", togglebtn.text)
            pass

    def add_item(self, item_text):
        print("Adding item ", item_text)

        cri = None
        extra_capsule_args = dict(caption=item_text,
                                  info_sidebar=self.ids.info_layout,
                                  status_label=self.ids.status_label,
                                  root_widget=self)

        default_figure_creation_args = self.selector_values_to_creation_args(self.default_figure_selector_vals)
        if item_text == 'SpamTable':
            #wstable = ws.SpamTable(models, titles, 'boxes', cri, False)  # titles?
            figure_capsule = FigureCapsule(self.ws.SpamTable, ['*models', '*model_titles', 'boxes', cri, False],
                                           **extra_capsule_args)

        figure_capsule.fill_args_from_creation_arg_dict(default_figure_creation_args)
        figure_capsule.update_figure_widget(self.ids.data_area)
        #TODO - add rest


    def old_add_item(self, item_text):
        print("OLD Adding item ", item_text)

        resultsdir = self.resultsdir_selector_widget.selected_results_dir
        data = resultsdir.data
        edesign = data.edesign

        results = self.results_selector_widget.selected_results

        circuit_list = edesign.all_circuits_needing_data
        dataset = data.dataset

        if isinstance(edesign, _StandardGSTDesign):
            max_length_list = edesign.maxlengths
            circuits_by_L = edesign.circuit_lists
        else:
            max_length_list = None
            circuits_by_L = None

        if isinstance(results, _ModelEstimateResults):
            estimate = results.estimates[self.resultdetail_selector_widget.estimate_name]
            model = estimate.models[self.resultdetail_selector_widget.model_name]
            target_model = estimate.models['target'] if 'target' in estimate.models else None
            models = [model]
            titles = ['Estimate']
            objfn_builder = estimate.parameters.get(
                'final_objfn_builder', _objfns.ObjectiveFunctionBuilder.create_from('logl'))
            models_by_L = [estimate.models['iteration %d estimate' % i] for i in range(len(max_length_list))] \
                if (max_length_list is not None) else None
            est_lbls_mt = [est_name for est_name in results.estimates if est_name != "Target"]
            est_mdls_mt = [results.estimates[est_name].models.get('final iteration estimate', None)
                           for est_name in est_lbls_mt]
            gaugeopt_args = estimate.goparameters.get(self.resultdetail_selector_widget.model_name, {})
            estimate_params = estimate.parameters
        else:
            estimate = model = target_model = None
            models = titles = []
            objfn_builder = None
            models_by_L = None
            est_lbls_mt = None
            est_mdls_mt = None
            gaugeopt_args = {}
            estimate_params = {}
        cri = None

        ws = Workspace(gui_mode='kivy')
        wstable = None
        wsplot = None
        if item_text == 'SpamTable':
            wstable = ws.SpamTable(models, titles, 'boxes', cri, False)  # titles?
        elif item_text == 'SpamParametersTable':
            wstable = ws.SpamParametersTable(models, titles, cri)
        elif item_text == 'GatesTable':
            wstable = ws.GatesTable(models, titles, 'boxes', cri)
        elif item_text == 'ChoiTable':
            wstable = ws.ChoiTable(models, titles, cri)
        elif item_text == 'ModelVsTargetTable':
            clifford_compilation = None
            wstable = ws.ModelVsTargetTable(model, target_model, clifford_compilation, cri)
        elif item_text == 'GatesVsTargetTable':
            wstable = ws.GatesVsTargetTable(model, target_model, cri)  # wildcard?
        elif item_text == 'SpamVsTargetTable':
            wstable = ws.SpamVsTargetTable(model, target_model, cri)
        elif item_text == 'ErrgenTable':
            wstable = ws.ErrgenTable(model, target_model, cri)  # (more options)
        elif item_text == 'NQubitErrgenTable':
            wstable = ws.NQubitErrgenTable(model, cri)
        elif item_text == 'GateDecompTable':
            wstable = ws.GateDecompTable(model, target_model, cri)
        elif item_text == 'GateEigenvalueTable':
            wstable = ws.GateEigenvalueTable(model, target_model, cri,
                                             display=('evals', 'rel', 'log-evals', 'log-rel'))
        elif item_text == 'DataSetOverviewTable':
            wstable = ws.DataSetOverviewTable(dataset, max_length_list)
        elif item_text == 'SoftwareEnvTable':
            wstable = ws.SoftwareEnvTable()
        elif item_text == 'CircuitTable':
            # wstable = ws.CircuitTable(...)  # wait until we can select circuit list; e.g. germs, fiducials
            print("Wait until better selection methods to create circuit tables...")
        elif item_text == 'GatesSingleMetricTable':
            #metric = 'inf'  # entanglement infidelity
            #wstable = GatesSingleMetricTable(metric, ...)
            print("Wait until better selection methods to create single-item gate metric tables...")
        elif item_text == 'StandardErrgenTable':
            wstable = ws.StandardErrgenTable(model.dim, 'hamiltonian', 'pp')  # not super useful; what about 'stochastic'?
        elif item_text == 'GaugeOptParamsTable':
            wstable = ws.GaugeOptParamsTable(gaugeopt_args)
        elif item_text == 'MetadataTable':
            wstable = ws.MetadataTable(model, estimate_params)
        elif item_text == 'WildcardBudgetTable':
            wstable = ws.WildcardBudgetTable(estimate_params.get("unmodeled_error", None))
        elif item_text == 'FitComparisonTable':
            wstable = ws.FitComparisonTable(max_length_list, circuits_by_L, models_by_L, dataset)
        elif item_text == 'FitComparisonBarPlot':
            wsplot = ws.FitComparisonBarPlot(max_length_list, circuits_by_L, models_by_L, dataset)
        elif item_text == 'FitComparisonBarPlotB':
            wsplot = ws.FitComparisonBarPlot(est_lbls_mt, [circuit_list] * len(est_mdls_mt),
                                             est_mdls_mt, [dataset] * len(est_mdls_mt), objfn_builder)

        elif item_text == 'FitComparisonBoxPlot':
            # used for multiple data sets -- enable this once we get better selection methods
            print("Wait until better selection methods to create fit comparison box plot...")
        elif item_text in ('ColorBoxPlot', 'ColorScatterPlot', 'ColorHistogramPlot'):

            if item_text == 'ColorBoxPlot': plot_type = "boxes"
            elif item_text == "ColorScatterPlot": plot_type = "scatter"
            else: plot_type = "histogram"

            linlog_percentile = 5
            bgcolor = 'white'
            wsplot = ws.ColorBoxPlot(
                objfn_builder, circuit_list,
                dataset, model,  # could use non-gauge-opt model here?
                linlg_pcntle=linlog_percentile / 100, comm=None, bgcolor=bgcolor,
                typ=plot_type)

        else:
            wstable = wsplot = None

        if wstable is not None:
            tbl = wstable.tables[0]
            out = tbl.render('kivywidget', kivywidget_kwargs={'size_hint': (None, None)})
            tblwidget = out['kivywidget']
            #self.data_area.clear_widgets()
            fig = FigureContainer(tblwidget, item_text, size_hint=(None, None))
            set_info_containers(fig, self.ids.info_layout, self.ids.status_label)
            self.ids.data_area.add_widget(fig)
        elif wsplot is not None:
            plt = wsplot.figs[0]
            constructor_fn, kwargs = plt.kivywidget
            natural_size = plt.metadata.get('natural_size', (300, 300))
            kwargs.update({'size_hint': (None, None)})
            pltwidget = constructor_fn(**kwargs)
            pltwidget.size = natural_size
            print("DB: PLOT Initial size = ", natural_size)
            #self.data_area.clear_widgets()
            fig = FigureContainer(pltwidget, item_text, size_hint=(None, None))
            set_info_containers(fig, self.ids.info_layout, self.ids.status_label)
            self.ids.data_area.add_widget(fig)
        else:
            print("Cannot create " + item_text + " yet.")


class TreeViewLabelWithData(TreeViewLabel):
    def __init__(self, path, data, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.data = data


class BorderedBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        if 'border_width' in kwargs:
            self.border_thickness = kwargs['border_thickness']
            del kwargs['border_thickness']
        else:
            self.border_thickness = 4
        super().__init__(**kwargs)

        with self.canvas.after:
            Color(0.5, 0.5, 0.5)
            self.border_rect = Line(points=[], width=self.border_thickness)
        self._update_border()
        self.bind(size=self._update_border, pos=self._update_border)

    def _update_border(self, *args):
        t = self.border_thickness
        x1, y1 = self.x + t, self.y + t
        x2, y2 = self.x + self.width - t, self.y + self.height - t
        self.border_rect.points = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]


class ResultsDirSelectorWidget(BorderedBoxLayout):
    # allows selection of edesign (tree node) and results object (dataset) at tree node.
    root_name = StringProperty('')
    root_file_path = StringProperty('')
    root_results_dir = ObjectProperty(None, allownone=True)
    selected_results_dir_node = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a trigger bound to multiple properties to update based on any number of then changing
        self._trigger_update = Clock.create_trigger(self.update_contents, 0)
        self.bind(root_name=self._trigger_update, root_results_dir=self._trigger_update)

    def update_contents(self, *args):
        tv = TreeView(root_options=dict(text=('From .../' + self.root_name)),
                      hide_root=False,
                      indent_level=4)

        def populate_view(results_dir, parent_tree_node, current_path):
            for ky in results_dir.keys():
                pth = current_path + [ky]
                tree_node = tv.add_node(TreeViewLabelWithData(path=pth, data=results_dir[ky], text=str(ky), is_open=True),
                                        parent=parent_tree_node)  # OK if None
                populate_view(results_dir[ky], tree_node, pth)

        populate_view(self.root_results_dir, None, [self.root_name])
        tv.bind(_selected_node=self.on_change_selected_node)

        self.clear_widgets()
        self.add_widget(tv)
        tv.select_node(tv.get_root())

    def on_change_selected_node(self, instance, new_node):
        print("Selected results dir: " + new_node.text)  # (new_node is a TreeViewLabel)
        self.selected_results_dir_node = new_node if hasattr(new_node, 'data') \
            else TreeViewLabelWithData(path=[], data=self.root_results_dir, text='ROOT', is_open=True)


class ResultsDirDetailSelectorWidget(BorderedBoxLayout):
    results_dir_selector_widget = ObjectProperty(None, allownone=True)
    model_list_widget = ObjectProperty(None, allownone=True)
    protocol_name = StringProperty(None, allownone=True)
    additional_detail = DictProperty({})
    selected_model_names = ListProperty([])  # maybe move this to model list widget?

    # allows selection of model, gaugeopt, etc.
    def __init__(self, **kwargs):
        kwargs['orientation'] = 'vertical'
        kwargs['size_hint_y'] = None
        super().__init__(**kwargs)
        #self.results_dir = None  # the current results object
        #self.results = None  # the current results object
        self.rows = []  # list of horizontal BoxLayouts, one per row

    def on_results_dir_selector_widget(self, inst, val):
        if self.results_dir_selector_widget is None: return
        if self.results_dir_selector_widget.selected_results_dir_node is not None:
            self.on_change_selected_results_dir_node(None, self.results_dir_selector_widget.selected_results_dir_node)
        self.results_dir_selector_widget.bind(selected_results_dir_node=self.on_change_selected_results_dir_node)

    def rebuild(self):
        self.clear_widgets()
        for row in self.rows:
            self.add_widget(row)
        self.height = len(self.rows) * 60

    def on_change_selected_results_dir_node(self, instance, new_results_dir_node):
        self.rows.clear()
        results_dir = new_results_dir_node.data

        protocol_names = list(results_dir.for_protocol.keys())
        protocol_row = BoxLayout(orientation='horizontal')
        protocol_spinner = Spinner(text=protocol_names[0] if (len(protocol_names) > 0) else '(none)',
                                   values=protocol_names, size_hint=(0.6, 1.0))
        protocol_spinner.bind(text=self.on_change_selected_protocol)
        protocol_row.add_widget(Label(text='Protocol:', size_hint=(0.4, 1.0)))
        protocol_row.add_widget(protocol_spinner)
        self.rows.append(protocol_row)
        self.on_change_selected_protocol(None, protocol_names[0] if (len(protocol_names) > 0) else '(none)')

    def on_change_selected_protocol(self, spinner, new_protocol_key):
        self.protocol_name = new_protocol_key
        results_dir = self.results_dir_selector_widget.selected_results_dir_node.data
        results = results_dir.for_protocol[new_protocol_key]
        del self.rows[1:]  # keep only the protocol row
        self.additional_detail.clear()

        if isinstance(results, _ModelEstimateResults):
            estimate_row = BoxLayout(orientation='horizontal')
            estimate_keys = list(results.estimates.keys())
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
        #if len(self.rows) > 1:
        #    self.remove_widget(self.rows[1])  # remove second row == "Model: ..." row

        self.additional_detail['estimate_name'] = new_estimate_key
        self.rebuild()

        if new_estimate_key is not None:
            results_dir = self.results_dir_selector_widget.selected_results_dir_node.data
            estimate = results_dir.for_protocol[self.protocol_name].estimates[new_estimate_key]
            model_names = list(estimate.models.keys())
        else:
            model_names = []

        #model_row = BoxLayout(orientation='horizontal')
        #model_spinner = Spinner(text=model_names[0] if (len(model_names) > 0) else '(none)',
        #                        values=model_names, size_hint=(0.6, 1.0))
        #model_spinner.bind(text=self.on_change_selected_model)
        #model_row.add_widget(Label(text='Model:', size_hint=(0.4, 1.0)))
        #model_row.add_widget(model_spinner)
        if self.model_list_widget is not None:
            self.model_list_widget.clear_widgets()
            for mn in model_names:
                btn = ToggleButton(text=mn, size_hint_y=None, height=40)
                btn.bind(state=self.on_change_selected_models) 
                self.model_list_widget.add_widget(btn)

        #self.on_change_selected_model(None, model_names[0] if (len(model_names) > 0) else None)

    def on_change_selected_models(self, *args):
        self.selected_model_names.clear()
        if self.model_list_widget is not None:
            for btn in self.model_list_widget.children:
                if btn.state == 'down':
                    self.selected_model_names.append(btn.text)


#class ResultsSelectorWidget(BorderedBoxLayout):
#    results_dir_selector_widget = ObjectProperty(None, allownone=True)
#    selected_results = ObjectProperty(None, allownone=True)
#
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self.treeview = None
#
#    def on_results_dir_selector_widget(self, inst, val):
#        if self.results_dir_selector_widget is None: return
#        if self.results_dir_selector_widget.selected_results_dir is not None:
#            self.refresh_tree(self.results_dir_selector_widget.selected_results_dir)                
#        self.results_dir_selector_widget.bind(selected_results_dir=self.on_change_selected_results_dir)
#
#    def refresh_tree(self, current_results_dir):
#        tv = TreeView(root_options=dict(text='Results for protocol:'),
#                      hide_root=False,
#                      indent_level=4)
#
#        def populate_view(results_dir, parent_tree_node):
#            for protocol_name, protocol_results in results_dir.for_protocol.items():
#                tv.add_node(TreeViewLabelWithData(data=protocol_results, text=str(protocol_name),
#                                                  is_open=True), parent=parent_tree_node)  # OK if None
#
#        populate_view(current_results_dir, None)
#        tv.bind(_selected_node=self.on_change_selected_node)
#
#        self.treeview = tv
#        self.clear_widgets()
#        self.add_widget(tv)
#
#    def on_change_selected_node(self, instance, new_node):
#        if new_node.is_leaf:
#            print("Selected Results: " + new_node.text)  # (new_node is a TreeViewLabel)
#            self.selected_results = new_node.data if hasattr(new_node, 'data') else None
#        else:
#            print("Root result item selected (?)")
#
#    def on_change_selected_results_dir(self, instance, new_results_dir):
#        self.refresh_tree(new_results_dir)
#
#        #Select the first selectable node
#        if self.treeview is not None:
#            if len(self.treeview.get_root().nodes) > 0:
#                self.treeview.select_node(self.treeview.get_root().nodes[0])
#
#
#class ResultDetailSelectorWidget(BorderedBoxLayout):
#    results_selector_widget = ObjectProperty(None, allownone=True)
#    estimate_name = StringProperty(None, allownone=True)
#    model_name = StringProperty(None, allownone=True)
#
#    # allows selection of model, gaugeopt, etc.
#    def __init__(self, **kwargs):
#        kwargs['orientation'] = 'vertical'
#        kwargs['size_hint_y'] = None
#        super().__init__(**kwargs)
#        self.results = None  # the current results object
#        self.rows = []  # list of horizontal BoxLayouts, one per row
#
#    def on_results_selector_widget(self, inst, val):
#        if self.results_selector_widget is None: return
#        if self.results_selector_widget.selected_results is not None:
#            self.on_change_selected_results(None, self.results_selector_widget.selected_results)
#        self.results_selector_widget.bind(selected_results=self.on_change_selected_results)
#
#    def rebuild(self):
#        self.clear_widgets()
#        for row in self.rows:
#            self.add_widget(row)
#        self.height = len(self.rows) * 60
#
#    def on_change_selected_results(self, instance, new_results_obj):
#        self.rows.clear()
#        self.results = new_results_obj  # make this the current results object
#
#        if isinstance(new_results_obj, _ModelEstimateResults):
#            estimate_row = BoxLayout(orientation='horizontal')
#            estimate_keys = list(new_results_obj.estimates.keys())
#            estimate_spinner = Spinner(text=estimate_keys[0] if (len(estimate_keys) > 0) else '(none)',
#                                       values=estimate_keys, size_hint=(0.6, 1.0))
#            estimate_spinner.bind(text=self.on_change_selected_estimate)
#            estimate_row.add_widget(Label(text='Estimate:', size_hint=(0.4, 1.0)))
#            estimate_row.add_widget(estimate_spinner)
#            self.rows.append(estimate_row)
#            self.on_change_selected_estimate(None, estimate_keys[0] if (len(estimate_keys) > 0) else '(none)')
#        else:
#            self.rows.append(Label(text='No details'))
#            self.rebuild()
#
#    def on_change_selected_estimate(self, spinner, new_estimate_key):
#        #Note: this is only called when self.results is a ModelEstimateResults object
#        if len(self.rows) > 1:
#            self.remove_widget(self.rows[1])  # remove second row == "Model: ..." row
#
#        self.estimate_name = new_estimate_key
#        if new_estimate_key is not None:
#            estimate = self.results.estimates[new_estimate_key]
#            model_names = list(estimate.models.keys())
#        else:
#            model_names = []
#
#        model_row = BoxLayout(orientation='horizontal')
#        model_spinner = Spinner(text=model_names[0] if (len(model_names) > 0) else '(none)',
#                                values=model_names, size_hint=(0.6, 1.0))
#        model_spinner.bind(text=self.on_change_selected_model)
#        model_row.add_widget(Label(text='Model:', size_hint=(0.4, 1.0)))
#        model_row.add_widget(model_spinner)
#        self.rows.append(model_row)
#        self.rebuild()
#        self.on_change_selected_model(None, model_names[0] if (len(model_names) > 0) else None)
#
#    def on_change_selected_model(self, spinner, new_model_name):
#        self.model_name = new_model_name  # set property so other layouts can trigger off of?


class CenterAreaWidget(BoxLayout, StencilView):
    # needs menus of all available tables/plots to add (for currently selected results/model/data/gaugeopt, etc)
    resultsdir_selector_widget = ObjectProperty(None, allownone=True)
    results_selector_widget = ObjectProperty(None, allownone=True)
    resultdetail_selector_widget = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

    def on_results_selector_widget(self, inst, val):
        self.results_selector_widget.bind(selected_results=self.selection_change)

    def on_resultdetail_selector_widget(self, inst, val):
        self.resultdetail_selector_widget.bind(estimate_name=self.selection_change, model_name=self.selection_change)

    def selection_change(self, instance, value):
        print("Data area noticed a selected results or model change... do something in the future?")



class DataAreaWidget(RelativeLayout):
    root_widget = ObjectProperty(None, allownone=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas:
            Color(0.7, 0.7, 0.7, 1)
            self._bgrect = Rectangle(pos=(0,0), size=self.size)  # note: *relative* coords
        self.bind(size=self._draw)

    def _draw(self, *args):
        self._bgrect.size = self.size

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print("data area touch down")
            if self.root_widget is None:
                return super().on_touch_down(touch)

            #See if touch should active a figure container
            for figc in self.children:  # loop over figure containers
                if figc.collide_point(*touch.pos):
                    print("Figure %s received touch-down event" % figc.title)
                    self.root_widget.set_active_figure_container(figc)
                    break
            else:
                print("no collision with any figure container")
                self.root_widget.set_active_figure_container(None)

            # don't count activation as actual 'processing', so continue on and
            # let super decide whether this event is processed
            return super().on_touch_down(touch)
        else:
            return False


class FigureContainer(DragBehavior, ResizableBehavior, BoxLayout):
    def __init__(self, fig_widget, title, capsule, **kwargs):
        kwargs['orientation'] = 'vertical'
        resize_kwargs = dict(
            resizable_left=False,
            resizable_right=True,
            resizable_up=False,
            resizable_down=True,
            resizable_border=10,
            resizable_border_offset=5)
        ResizableBehavior.__init__(self, **resize_kwargs)
        BoxLayout.__init__(self, **kwargs)
        initial_size = fig_widget.size

        with self.canvas.before:
            Color(0.0, 0.4, 0.4, 0.0)  # Make opaque when debugging - turquoise figure background
            self.bgrect = Rectangle(pos=self.pos, size=self.size)

            self.active_color = Color(0.8, 0.8, 0, 0)  # select / active box color
            self.active_border_thickness = 4
            x1, y1 = self.x, self.y
            x2, y2 = self.x + self.width, self.y + self.height
            pts = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]
            self.activebox = Line(points=pts, width=self.active_border_thickness)

        self.size = initial_size
        self.title = title
        self.content = None
        self.capsule = capsule
        self.add_widget(Label(text=title, bold=True, size_hint_y=None, height=50, color=(0,0,0,1), font_size=18))
        self.set_content(fig_widget)  # sets self.content
        #self.set_cursor_mode(0)

        db = self.drag_border = 2 * (self.resizable_border - self.resizable_border_offset)  # 2 for extra measure
        drag_kwargs = {'drag_rectangle': (self.x + db, self.y + db, self.width - 2 * db, self.height - 2 * db),
                       'drag_timeout': 2000 }  # wait 2 seconds before giving up on drag
        DragBehavior.__init__(self, **drag_kwargs)
        print("initial drag rect = ", self.drag_rectangle)

    def set_content(self, fig_widget):
        fig_widget.size_hint_x = 1.0
        fig_widget.size_hint_y = 1.0
        if self.content is not None:
            self.remove_widget(self.content)

        self.content = fig_widget
        if self.content is not None:
            self.add_widget(fig_widget)

    def _redraw(self):
        self.bgrect.pos = self.pos
        self.bgrect.size = self.size

        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.width, self.y + self.height
        self.activebox.points = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]

    def deactivate(self):
        self.active_color.a = 0.0

    def activate(self):
        self.active_color.a = 1.0

    def on_size(self, *args):
        self._redraw()

    def on_pos(self, *args):
        self._redraw()
        #print("Pos change: ", self.pos, ' drag_rect = ',self.drag_rectangle)
        db = self.drag_border
        self.drag_rectangle = (self.pos[0] + db, self.pos[1] + db, self.size[0] - 2 * db, self.size[1] - 2 * db)

class FigureArgumentSelector(BoxLayout):
    def create_default_args_selectors():
        pass

        
class FigureCapsule(object):

    def __init__(self, creation_fn, args_template, root_widget, caption='', info_sidebar=None, status_label=None):
        self.creation_fn = creation_fn
        self.args_template = args_template
        self.selector_vals = {}
        self.args = []
        self.caption = caption
        self.fig_container = None
        self._info_sidebar = info_sidebar
        self._status_label = status_label
        self.root_widget = root_widget

    def fill_args_from_creation_arg_dict(self, arg_dict):
        self.args = []
        for t in self.args_template:
            if isinstance(t, str) and t.startswith('*'):
                self.args.append(arg_dict[t])
            else:
                self.args.append(t)

    def update_figure_widget(self, data_area):
        from pygsti.report.workspace import WorkspaceTable, WorkspacePlot

        workspace_obj = self.creation_fn(*self.args)
        if isinstance(workspace_obj, WorkspaceTable):
            tbl = workspace_obj.tables[0]
            out = tbl.render('kivywidget', kivywidget_kwargs={'size_hint': (None, None)})
            figwidget = out['kivywidget']

        elif isinstance(workspace_obj, WorkspacePlot):
            plt = workspace_obj.figs[0]
            constructor_fn, kwargs = plt.kivywidget
            natural_size = plt.metadata.get('natural_size', (300, 300))
            kwargs.update({'size_hint': (None, None)})
            figwidget = constructor_fn(**kwargs)
            figwidget.size = natural_size
            print("DB: PLOT Initial size = ", natural_size)
        else:
            raise ValueError("Invalid figure type created: " + str(type(workspace_obj)))

        if self.fig_container is None:
            fig = FigureContainer(figwidget, self.caption, self, size_hint=(None, None))  # Note: capsule and container should probably be one and the same...
            set_info_containers(fig, self._info_sidebar, self._status_label)
            data_area.add_widget(fig)
            self.fig_container = fig
        else:
            figwidget.size = self.fig_container.size
            self.fig_container.set_content(figwidget)

    def populate_figure_property_panel(self, panel_widget):
        panel_widget.clear_widgets()

        panel_widget.add_widget(Label(text=self.caption, bold=True, size_hint_y=None, height=40))
        my_properties = [name for name in self.args_template if (isinstance(name, str) and name.startswith('*'))]
        selector_types = self.root_widget.selector_types_for_properties(my_properties)
        for typ in selector_types:
            self.root_widget.add_figure_property_selector(typ, panel_widget, storage_dict=self.selector_vals)
        btn = Button(text='Update', size_hint_y=None, height=40)
        btn.bind(on_release=self.update_figure)
        panel_widget.add_widget(btn)

    def update_figure(self, **args):
        fig_creation_args = self.root_widget.selector_values_to_creation_args(self.selector_vals)
        self.fill_args_from_creation_arg_dict(fig_creation_args)
        self.update_figure_widget(None)


class CustomAccordionItem(AccordionItem):
    #Overrides _update_title so we don't have to use (deprecated) templates to customize them
    # Basically copied from accordion.py
    def _update_title(self, dt):
        if not self.container_title:
            self._trigger_title()
            return
        c = self.container_title
        c.clear_widgets()
        instance = CustomAccordionTitle(self.title, self, bold=True, font_size=24)
        c.add_widget(instance)


class CustomAccordionTitle(Label):
    """ Mimics the (deprecated) default Kivy template for an accordion title"""
    def __init__(self, text, item, **kwargs):
        from kivy.graphics import PushMatrix, PopMatrix, Translate, Rotate, BorderImage
        super().__init__(text=text, **kwargs)

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.bi = BorderImage(source=item.background_normal if item.collapse else item.background_selected,
                                  pos=self.pos, size=self.size)
            PushMatrix()
            self.t1 = Translate(xy=(self.center_x, self.center_y))
            Rotate(angle= 90 if item.orientation == 'horizontal' else 0, axis=(0, 0, 1))
            self.t2 = Translate(xy=(-self.center_x, -self.center_y))

        with self.canvas.after:
            PopMatrix

        self.bind(pos=self.update, size=self.update)
        item.bind(collapse=lambda inst, v: setattr(self.bi, 'source', inst.background_normal
                                                   if v else inst.background_selected))

    def update(self, *args):
        self.bi.pos = self.pos
        self.bi.size = self.size
        self.t1.xy = (self.center_x, self.center_y)
        self.t2.xy = (-self.center_x, -self.center_y)


def add_row(widget, k, v):
    row = BoxLayout(orientation='horizontal')
    row.add_widget(Label(text=str(k), font_size=20))
    row.add_widget(Label(text=str(v), font_size=20))
    widget.add_widget(row)


class ProcessorSpecModal(ModalView):
    def __init__(self, results_dir, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')

        from pygsti.protocols.gst import HasProcessorSpec as _HasProcessorSpec
        #root_edesign = self.results_dir_selector_widget.root_results_dir.data.edesign
        edesign = results_dir.data.edesign
        pspec = edesign.processor_spec if isinstance(edesign, _HasProcessorSpec) else None
        layout.add_widget(Label(text='Processor Specification', font_size=24, bold=True))

        if pspec is not None:
            add_row(layout, '# of qubits:', pspec.num_qubits)
            add_row(layout, 'gate names:', pspec.gate_names)  # clickable for availability?
            add_row(layout, 'prep names:', pspec.prep_names)
            add_row(layout, 'POVM names:', pspec.povm_names)
        else:
            layout.add_widget(Label(text='(no info)', font_size=18))
        self.add_widget(layout)


#class ProcessorSpecInfoWidget(BoxLayout):
#    def __init__(self, results_dir_selector_widget, **kwargs):
#        kwargs['orientation'] = 'vertical'
#        super().__init__(**kwargs)
#        results_dir_selector_widget.bind(selected_results_dir=self.update_selected)
#        self.results_dir_selector_widget = results_dir_selector_widget
#        self.update_selected(None, None)
#
#    def update_selected(self, inst, val):
#        from pygsti.protocols.gst import HasProcessorSpec as _HasProcessorSpec
#        #root_edesign = self.results_dir_selector_widget.root_results_dir.data.edesign
#        edesign = self.results_dir_selector_widget.selected_results_dir.data.edesign
#        pspec = edesign.processor_spec if isinstance(edesign, _HasProcessorSpec) else None
#        self.clear_widgets()
#        self.add_widget(Label(text='Processor Specification', font_size=24, bold=True))
#        
#        def add_row(k, v):
#            row = BoxLayout(orientation='horizontal')
#            row.add_widget(Label(text=str(k), font_size=20))
#            row.add_widget(Label(text=str(v), font_size=20))
#            self.add_widget(row)
#
#        if pspec is not None:
#            add_row('# of qubits:', pspec.num_qubits)
#            add_row('gate names:', pspec.gate_names)  # clickable for availability?
#            add_row('prep names:', pspec.prep_names)
#            add_row('POVM names:', pspec.povm_names)
#        else:
#            self.add_widget(Label(text='(no info)', font_size=18))


class ExperimentDesignModal(ModalView):
    def __init__(self, results_dir, **kwargs):
        super().__init__(**kwargs)
        
        layout = BoxLayout(orientation='vertical')

        from pygsti.protocols.protocol import CircuitListsDesign, CombinedExperimentDesign, \
            SimultaneousExperimentDesign
        from pygsti.protocols.gst import GateSetTomographyDesign, StandardGSTDesign
        edesign = results_dir.data.edesign

        layout.add_widget(Label(text='Experiment Design', font_size=24, bold=True))
        add_row(layout, 'Type:', str(edesign.__class__.__name__))
        add_row(layout, '# of circuits:', len(edesign.all_circuits_needing_data))
        if isinstance(edesign, CircuitListsDesign):
            add_row(layout, 'Circuit list lengths:', ", ".join(map(str, map(len, edesign.circuit_lists))))
        if isinstance(edesign, CombinedExperimentDesign):
            add_row(layout, 'Sub-designs:', ", ".join(map(str, edesign.keys())))
        if isinstance(edesign, SimultaneousExperimentDesign):
            add_row(layout, 'Sub-designs:', ", ".join(map(str, edesign.keys())))
        if isinstance(edesign,  GateSetTomographyDesign):
            pass  # doesn't add anything beyond a CircuitListsDesign other than a processor spec
        if isinstance(edesign, StandardGSTDesign):
            add_row(layout, '# prep fiducials:', len(edesign.prep_fiducials))
            add_row(layout, '# meas fiducials:', len(edesign.meas_fiducials))
            add_row(layout, '# germs:', len(edesign.germs))
            add_row(layout, 'Max-depths:', ", ".join(map(str, edesign.maxlengths)))
        self.add_widget(layout)


class DatasetModal(ModalView):
    def __init__(self, results_dir, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        ds = results_dir.data.dataset

        layout.add_widget(Label(text='DataSet', font_size=24, bold=True))
        add_row(layout, '# of circuits:', len(ds))
        if ds.has_constant_totalcounts_pertime:
            add_row(layout, 'samples per circuit:', ds.totalcounts_pertime)
        if not ds.has_trivial_timedependence:
            add_row(layout, 'timestamps:', ds.timestamps)
        self.add_widget(layout)

#REMOVE
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

from kivy.lang import Builder


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


load_dialog_kv = \
"""<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser

        BoxLayout:
            size_hint_y: None
            height: 60
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Load"
                on_release: root.load(filechooser.path, filechooser.selection)
"""
Builder.load_string(load_dialog_kv)


class DataExplorerApp(App):
    def __init__(self, initial_results_dir_path):  #, test_widget):
        self.initial_results_dir_path = initial_results_dir_path
        #self.test_widget = test_widget
        super().__init__()

    def build(self):
        return RootExplorerWidget(self.initial_results_dir_path)


#if __name__ == '__main__':
#    DataExplorerApp(tblwidget).run()
