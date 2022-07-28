import os as _os
import json as _json
import inspect as _inspect
import itertools as _itertools

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
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem, TabbedPanelHeader
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.modalview import ModalView
from kivy.uix.stencilview import StencilView
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock

from kivy.properties import ObjectProperty, StringProperty, ListProperty, DictProperty, BoundedNumericProperty
from kivy.properties import AliasProperty
from kivy.graphics import Color, Rectangle, Line

from .kivyresize import ResizableBehavior
from .kivygraph import Graph, MeshLinePlot, BarPlot, MatrixBoxPlotGraph

import pygsti
from pygsti.report import Workspace
from pygsti.report.kivyfrontend import WrappedLabel, TableWidget, LatexWidget
from pygsti.report.kivywidget import KivyWidgetFactory
from pygsti.protocols.protocol import CircuitListsDesign as _CircuitListsDesign
from pygsti.protocols.protocol import ExperimentDesign as _ExperimentDesign
from pygsti.protocols.protocol import ProtocolData as _ProtocolData
from pygsti.protocols.estimate import Estimate as _Estimate
from pygsti.protocols.gst import ModelEstimateResults as _ModelEstimateResults
from pygsti.protocols.gst import StandardGSTDesign as _StandardGSTDesign
from pygsti.io import read_results_from_dir as _read_results_from_dir
from pygsti.io import read_dataset as _read_dataset
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.models import Model as _Model

from kivy.core.window import Window
Window.size = (1200, 700)

#import tkinter as tk
#from tkinter import filedialog


def set_info_containers(root, sidebar, statusbar):
        """Walk down widget tree from `root` and call `set_info_containers` on applicable children. """
        if hasattr(root, 'set_info_containers'):
            root.set_info_containers(sidebar, statusbar)
        for c in root.children:
            set_info_containers(c, sidebar, statusbar)


class MockKernel(object):
    def __init__(self):
        self.globals_dict = {'__builtins__': __builtins__, 'print': print}
        self.reply = None

    def execute(self, code, user_expressions=None):
        exec(code, self.globals_dict)
        print("--> globals after exec = ", self.globals_dict.get('__ret__', None), self.globals_dict.get('__mock_kernel_val__', None))
        self.reply = {'content': {'status': 'ok',
                                  'user_expressions': {}
                                  }
                      }
        if user_expressions is not None:
            for key, expr in user_expressions.items():
                exec('__mock_kernel_val__ = %s' % expr, self.globals_dict)
                val = self.globals_dict['__mock_kernel_val__']
                self.reply['content']['user_expressions'][key] = {'status': 'ok',
                                                                  'data': {'text/plain': str(val)}}

    def get_shell_msg(self, timeout):
        return self.reply


class DataExplorerKernelInterface(object):
    def __init__(self, existing_kernel, debug=False):
        if existing_kernel is not None and debug is False:
            import jupyter_core
            from jupyter_client import BlockingKernelClient
            pth = _os.path.join(jupyter_core.paths.jupyter_runtime_dir(), existing_kernel)
            with open(pth, 'r') as f:
                connection = _json.load(f)
            self.kc = BlockingKernelClient()
            self.kc.load_connection_info(connection)
            self.kc.start_channels()

            pth = _os.path.join(_os.path.dirname(__file__), 'explorerkernelcode.py')
            with open(pth, 'r') as f:
                initial_kernel_code = f.read()
            self.kc.execute(initial_kernel_code)
            reply = self.kc.get_shell_msg(timeout=10)
            print("Received reply 1:\n", reply)
        elif debug is True:
            self.kc = MockKernel()
            pth = _os.path.join(_os.path.dirname(__file__), 'explorerkernelcode.py')
            with open(pth, 'r') as f:
                initial_kernel_code = f.read()
            self.kc.execute(initial_kernel_code)

        else:
            raise NotImplementedError("Need to specify an existing kernel, since we haven't implemented starting one up on our own yet!")

    def __del__(self):
        self.kc.stop_channels()

    def reply_indicates_success(self, reply):
        return bool(reply['content']['status'] == 'ok')

    def extract_user_expression_str(self, reply, user_expression, default_value=None):
        if reply['content']['status'] == 'ok':
            if reply['content']['user_expressions'][user_expression]['status'] == 'ok':
                return reply['content']['user_expressions'][user_expression]['data']['text/plain'].strip("'").replace('\\\\', '\\')
        elif reply['content']['status'] == 'error':
            #Print traceback
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            print("KERNEL ERROR!!!")
            for frame_str in reply['content']['traceback']:
                print(ansi_escape.sub('',frame_str))
        return default_value

    def import_edesign(self, root_name, root_file_path, path_from_root,
                       import_children=False, import_all_models=False):
        #import bpdb; bpdb.set_trace()
        root_file_path = _os.path.abspath(root_file_path)
        self.kc.execute(f"__ret__ = lib.import_edesign('{root_name}', '{root_file_path}', '{_json.dumps(path_from_root)}', '{import_children}', '{import_all_models}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        print("Received reply 2:\n", reply)
        ret_str = self.extract_user_expression_str(reply, 'return_value')
        print("Return Text = ", ret_str)

        #pth_for_name = [root_name] + path_from_root if root_name else path_from_root
        #name = '.'.join(pth_for_name)  # edesign name
        #
        ##btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        ##btn.bind(state=self.update_library_item_info)
        #if name not in self.edesign_library:
        #
        #    data = results_dir.data  # a ProtocolData object
        #    self.edesign_library[name] = EdesignLibElement(name, data.edesign, root_file_path, path_from_root, btn)
        #    #self.ids.edesign_library_list.add_widget(btn)
        #    print("Imported edesign: ", name)
        #
        #    if data.dataset is not None:
        #        #btn2 = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        #        #btn2.bind(state=self.update_library_item_info)
        #        self.dataset_library[name] = DatasetLibElement(name, data.dataset, root_file_path, path_from_root, btn2)
        #        #self.ids.dataset_library_list.add_widget(btn2)
        #        print("Imported dataset: ", name)
        #
        #    if import_all_models:
        #        raise ValueError("TODO UPGRADE!")
        #        self._import_models(root_name, root_file_path, path_from_root, results_dir, 'all', 'all', 'all')
        #else:
        #    print("An e-design with name '%s' is already imported, and will not be clobbered." % name); return
        #
        #if import_children:
        #    for ky in results_dir.keys():
        #        child_path_from_root = path_from_root + [ky]
        #        child_results_dir = results_dir[ky]
        #        self.import_edesign(root_name, root_file_path, child_path_from_root,
        #                            child_results_dir, import_children, import_all_models)

    def import_models(self, root_name, root_file_path, path_from_root, protocol_names,
                      additional_details, model_names):
        root_file_path = _os.path.abspath(root_file_path)
        self.kc.execute((f"__ret__ = lib.import_models('{root_name}','{root_file_path}','{_json.dumps(path_from_root)}',"
                         f"'{_json.dumps(protocol_names)}','{_json.dumps(additional_details)}','{_json.dumps(model_names)}')"),
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        print("Received reply 3:\n", reply)
        ret_str = self.extract_user_expression_str(reply, 'return_value')
        print("Return Text = ", ret_str)

    def import_models_from_files(self, filenames):
        self.kc.execute(f"__ret__ = lib.import_models_from_files('{_json.dumps(filenames)}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        print("Received reply 3:\n", reply)
        ret_str = self.extract_user_expression_str(reply, 'return_value')
        print("Return Text = ", ret_str)

    def import_datasets_from_files(self, filenames):
        self.kc.execute(f"__ret__ = lib.import_datasets_from_files('{_json.dumps(filenames)}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        print("Received reply 3:\n", reply)
        ret_str = self.extract_user_expression_str(reply, 'return_value')
        print("Return Text = ", ret_str)

    def library_item_names(self, category=None):
        if category == 'edesign':
            self.kc.execute(f"__ret__ = json.dumps(list(lib.edesign_library.keys()))",
                            user_expressions={'return_value': '__ret__'})
        elif category == 'dataset':
            self.kc.execute(f"__ret__ = json.dumps(list(lib.dataset_library.keys()))",
                            user_expressions={'return_value': '__ret__'})
        elif category == 'model':
            self.kc.execute(f"__ret__ = json.dumps(list(lib.model_library.keys()))",
                            user_expressions={'return_value': '__ret__'})
        else:
            raise ValueError("Invalid library category: %s" % category)

        reply = self.kc.get_shell_msg(timeout=10)
        print("Received names: ", self.extract_user_expression_str(reply, 'return_value'))
        return _json.loads(self.extract_user_expression_str(reply, 'return_value'))

    def library_item_info(self, category, name):
        if category == 'edesign':
            self.kc.execute(f"__ret__ = json.dumps(lib.edesign_library['{name}'].info)",
                            user_expressions={'return_value': '__ret__'})
        elif category == 'dataset':
            self.kc.execute(f"__ret__ = json.dumps(lib.dataset_library['{name}'].info)",
                            user_expressions={'return_value': '__ret__'})
        elif category == 'model':
            self.kc.execute(f"__ret__ = json.dumps(lib.model_library['{name}'].info)",                        
                            user_expressions={'return_value': '__ret__'})
        else:
            raise ValueError("Invalid library category: %s" % category)
        reply = self.kc.get_shell_msg(timeout=10)
        return _json.loads(self.extract_user_expression_str(reply, 'return_value'))

    def update_library_item_name(self, category, existing_name, new_name):
        self.kc.execute(f"__ret__ = lib.update_item_name('{category}', '{existing_name}', '{new_name}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    def remove_library_item(self, category, name):
        self.kc.execute(f"__ret__ = lib.remove_item('{category}', '{name}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    def read_library(self, json_filename):
        self.kc.execute(f"__ret__ = lib.load_from_file('{json_filename}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    def save_library(self, json_filename):
        self.kc.execute(f"__ret__ = lib.save_to_file('{json_filename}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    def read_analysis(self, json_filename):
        self.kc.execute(f"__ret__ = json.dumps(analysis.load_from_file('{json_filename}'))",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        front_end_info = _json.loads(self.extract_user_expression_str(reply, 'return_value'))
        return front_end_info

    def save_analysis(self, json_filename, frontend_figure_info):
        self.kc.execute(f"__ret__ = analysis.save_to_file('{json_filename}','{_json.dumps(frontend_figure_info)}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    def add_figure(self, tab_name, figure_type, name):
        self.kc.execute(f"__ret__ = analysis.add_figure('{tab_name}', '{figure_type}', '{name}')",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        figure_id_str = self.extract_user_expression_str(reply, 'return_value')
        figure_id = int(figure_id_str) if (figure_id_str != 'None') else None
        return figure_id

    def create_figure_widget(self, figure_id, selector_vals, scale=1.0):
        self.kc.execute(f"analysis.update_figure({figure_id}, '{_json.dumps(selector_vals)}')")
        reply = self.kc.get_shell_msg(timeout=10)
        assert(self.reply_indicates_success(reply)), "Failed to update back-end figure!"

        self.kc.execute(f"__ret__ = analysis.capsules[{figure_id}].create_figure_widget_json()",
                        user_expressions={'figure_json': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        widget_factory_json = _json.loads(self.extract_user_expression_str(reply, 'figure_json'))

        # Build a kivy widget from received figure-widget-factory JSON
        widget_factory = KivyWidgetFactory.from_nice_serialization(widget_factory_json)
        widget = widget_factory.create_widget()
        print("DB: CREATED figure Initial size = ", widget.size, ' pos=', widget.pos)
        return widget

    def figure_property_names(self, figure_id):
        self.kc.execute(f"__ret__ = json.dumps(analysis.capsules[{figure_id}].property_names)",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return _json.loads(self.extract_user_expression_str(reply, 'return_value'))

    def remove_figure(self, figure_id):
        self.kc.execute(f"__ret__ = analysis.remove_figure({figure_id})",
                        user_expressions={'return_value': '__ret__'})
        reply = self.kc.get_shell_msg(timeout=10)
        return bool(self.extract_user_expression_str(reply, 'return_value') == 'True')

    #def get_analysis_item_selector_vals(self, item_id):
    #    pass



class CloseableHeader(TabbedPanelHeader):
    def edit_name_and_order(self, touch):
        if not touch.is_double_tap:
            return
        content = EditTabDialog(ok=self._update_name, cancel=self._dismiss_popup,
                                reorder=lambda direction: self.content.root_widget.move_tab(self, direction),
                                tab_name=self.ids.name.text)
        self._popup = Popup(title="Edit tab", content=content, size_hint=(0.7, 0.5))
        self._popup.open()

    def _update_name(self, new_name):
        self.text = new_name
        #self.ids.name.texture_update()
        #self.width = self.texture_size[0] + 60
        self._dismiss_popup()

    def _dismiss_popup(self):
        if self._popup is not None:
            self._popup.dismiss()
            self._popup = None            


class FixedHeightLabel(Label):
    pass


class ClickableLabel(Label):
    __events__ = ('on_release', )

    def on_release(self, *largs):
        pass

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dispatch('on_release')
            return True
        super().on_touch_down(touch)


class WrappedClickableLabel(WrappedLabel):
    __events__ = ('on_release', )

    def on_release(self, *largs):
        pass

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dispatch('on_release')
            return True
        super().on_touch_down(touch)


class NameDialog(FloatLayout):
    update = ObjectProperty(None)
    cancel = ObjectProperty(None)
    text = StringProperty('<empty>')

    
class RootExplorerWidget(BoxLayout):

    active_figure_container = ObjectProperty(None, allownone=True)
    active_data_area = AliasProperty(lambda self: (self.ids.figure_areas.current_tab.content.ids.data_area
                                                   if ('data_area' in self.ids.figure_areas.current_tab.content.ids)
                                                   else None), None)

    def __init__(self, app_path, initial_kernel_json_name=None, **kwargs):
        super().__init__(**kwargs)
        self._initial_kernel_json_name = initial_kernel_json_name
        self._initial_results_dir_path = 'results_root1'
        Clock.schedule_once(self.after_created, 0)
        self.mode = None
        self.kernel = DataExplorerKernelInterface(existing_kernel=initial_kernel_json_name)
        #self.edesign_library = {}
        #self.dataset_library = {}
        #self.model_library = {}
        #self.ws = Workspace(gui_mode='kivy')
        self.default_figure_selector_vals = {}
        self.active_figure_selector_vals = {}
        self.sidebar_widths = {}
        self.app_path = app_path
        self.library_path = None
        self.analysis_path = None
        self._popup = None  # active popup
        self.positioning_options = {}

        #Bind to keyboard events
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def change_mode(self, new_mode):
        print("Changing mode from %s to %s" % (self.mode, new_mode))

        def show_sidebar(w):
            #REMOVE print("Setting sidebar to ", self.sidebar_widths[id(w)] if (id(w) in self.sidebar_widths) else 500)
            w.width = self.sidebar_widths[id(w)] if (id(w) in self.sidebar_widths) else 500
            w.opacity = 1

        def hide_sidebar(w):
            if w.width > 0:
                #REMOVE print("Saving sidebar size before hiding: ",w.width)
                self.sidebar_widths[id(w)] = w.width
            w.width = 0
            w.opacity = 0

        if new_mode == 'Library':
            show_sidebar(self.ids.import_sidebar_splitter)
            self.ids.library_area.size_hint_x = 1.0
            self.ids.library_area.opacity = 1.0
            show_sidebar(self.ids.library_info_sidebar_splitter)

            hide_sidebar(self.ids.add_item_sidebar_splitter)
            self.ids.figure_areas.size_hint_x = None
            self.ids.figure_areas.width = 0
            self.ids.figure_areas.opacity = 0
            hide_sidebar(self.ids.figure_properties_sidebar_splitter)
            hide_sidebar(self.ids.figure_info_sidebar_splitter)

        elif new_mode == 'Create':
            hide_sidebar(self.ids.import_sidebar_splitter)
            self.ids.library_area.size_hint_x = None
            self.ids.library_area.width = 0
            self.ids.library_area.opacity = 0
            hide_sidebar(self.ids.library_info_sidebar_splitter)

            show_sidebar(self.ids.add_item_sidebar_splitter)
            self.ids.figure_areas.size_hint_x = 1.0
            self.ids.figure_areas.opacity = 1.0
            show_sidebar(self.ids.figure_properties_sidebar_splitter)
            hide_sidebar(self.ids.figure_info_sidebar_splitter)
            self.set_active_figure_container(None)  # to prompt population of default arg panel

        elif new_mode == 'Analysis':
            hide_sidebar(self.ids.import_sidebar_splitter)
            self.ids.library_area.size_hint_x = None
            self.ids.library_area.width = 0
            self.ids.library_area.opacity = 0
            hide_sidebar(self.ids.library_info_sidebar_splitter)

            hide_sidebar(self.ids.add_item_sidebar_splitter)
            self.ids.figure_areas.size_hint_x = 1.0
            self.ids.figure_areas.opacity = 1.0
            hide_sidebar(self.ids.figure_properties_sidebar_splitter)
            show_sidebar(self.ids.figure_info_sidebar_splitter)

        self.mode = new_mode

    def after_created(self, delta_time):
        print("Running post-kv-file creation of root widget.")
        self.ids.add_item_sidebar.add_widget(self.create_add_item_panel())
        set_info_containers(self.ids.figure_areas, self.ids.lower_info_area, self.ids.status_label)

        #Setup initial tab for creating more tabs
        self.add_new_tab_preset_buttons(self.ids.creation_tab_content)
        self.ids.figure_areas.switch_to(self.ids.figure_areas.tab_list[0])  # make sure 1 and only tab is selected
        self.ids.comment_input.bind(text=self.save_comment_text)

        #REMOVE -- DEBUG
        #if self._initial_results_dir_path:
        #    results_dir = pygsti.io.read_results_from_dir(self._initial_results_dir_path)
        #    self.ids.results_dir_selector.root_name = ".../" + _os.path.basename(self._initial_results_dir_path)
        #    self.ids.results_dir_selector.root_file_path = self._initial_results_dir_path
        #    self.ids.results_dir_selector.root_results_dir = results_dir  # automatic if we make self.results_dir a property
        self.change_mode('Library')

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        #print('The key', keycode, 'have been pressed')
        #print(' - text is %r' % text)
        #print(' - modifiers are %r' % modifiers)

        if keycode[1] == 'backspace' and self.active_figure_container is not None:
            if not self.kernel.remove_figure(self.active_figure_container.figure_id):
                print("Warning: figure %d was removed from GUI but not found in back-end kernel!"
                      % self.active_figure_container.figure_id)
            self.active_data_area.remove_widget(self.active_figure_container)
            self.set_active_figure_container(None)
            return True

        if self.active_figure_container is not None and keycode[1] in ('left', 'right', 'up', 'down', '-', '='):
            if keycode[1] == 'left':
                self.active_figure_container.x -= 10
            elif keycode[1] == 'right':
                self.active_figure_container.x += 10
            elif keycode[1] == 'up':
                self.active_figure_container.y += 10
            elif keycode[1] == 'down':
                self.active_figure_container.y -= 10
            elif keycode[1] == '=':
                self.active_figure_container.x -= 5
                self.active_figure_container.y -= 5
                self.active_figure_container.width += 10
                self.active_figure_container.height += 10
            elif keycode[1] == '-':
                self.active_figure_container.x += 5
                self.active_figure_container.y += 5
                self.active_figure_container.width -= 10
                self.active_figure_container.height -= 10

            # Schedule an extra redraw so table lines update correctly
            Clock.schedule_once(lambda dt: self.active_figure_container.content._redraw()
                                if isinstance(self.active_figure_container.content, TableWidget) else None)
            return True

        # Return True to accept the key. Otherwise, it will be used by the system.
        return False

    def set_active_figure_container(self, active_figure_container):
        if self._keyboard is None:
            self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
            self._keyboard.bind(on_key_down=self._on_keyboard_down)

        if self.active_figure_container is not None:
            self.active_figure_container.deactivate()

        if active_figure_container is not None:
            active_figure_container.activate()
            active_figure_container.populate_info_panel(self.ids.upper_info_area)
            active_figure_container.populate_figure_property_panel(self.ids.figure_properties_sidebar)
            self.ids.comment_input.text = active_figure_container.comment
        else:
            self.populate_figure_property_defaults_panel(self.ids.figure_properties_sidebar)
            self.ids.upper_info_area.clear_widgets()
            self.ids.lower_info_area.clear_widgets()
            # (clear comment_input below)
            
        self.active_figure_container = active_figure_container
        
        if active_figure_container is None:
            self.ids.comment_input.text = ''  # *after* setting self.active_figure_container (see save_comment_text)

    def save_comment_text(self, inst, val):
        print("SAVING COMMENT: ", inst, val)
        if self.active_figure_container is not None:
            self.active_figure_container.comment = val

    def populate_figure_property_defaults_panel(self, panel_widget):
        panel_widget.clear_widgets()
        panel_widget.add_widget(FixedHeightLabel(text='Defaults', bold=True, bgcolor=(0, 0.4, 0.6, 1)))

        all_properties = ['*models', '*model_titles', '*model', '*model_title', '*model_dim', '*target_model',
                          '*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl',
                          '*models_by_maxl', '*objfn_builder', '*gaugeopt_args', '*estimate_params', '*unmodeled_error']

        selector_types = self.selector_types_for_properties(all_properties)
        for typ in selector_types:
            self.add_figure_property_selector(typ, panel_widget,
                                              storage_dict=self.default_figure_selector_vals)

        #Add positioning options
        panel_widget.add_widget(FixedHeightLabel(text=''))
        panel_widget.add_widget(FixedHeightLabel(text='Positioning', padding=(10, 10), bgcolor=(0, 0.4, 0.6, 1)))
        row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        row.add_widget(Label(text='Arrange'))
        spinner = Spinner(text=self.positioning_options.get('arrange', "top down"),
                          values=['top down', 'all in center'])
        spinner.bind(text=lambda inst, txt: self.positioning_options.__setitem__('arrange', txt))
        row.add_widget(spinner)
        panel_widget.add_widget(row)

        row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        row.add_widget(Label(text='Scale'))
        inpt = TextInput(text=self.positioning_options.get('scale', "1.0"), multiline=False)
        inpt.bind(text=lambda inst, txt: self.positioning_options.__setitem__('scale', txt))
        row.add_widget(inpt)
        panel_widget.add_widget(row)

    def selector_types_for_properties(self, property_names):
        prop_set = set(property_names)
        dependencies = {  # include key if any of values (property template names) are present
            '**model': set(['*model', '*model_title', '*model_dim', '*models', '*model_titles',
                            '*gaugeopt_args', '*estimate_params', '*unmodeled_error', '*models_by_maxl']),
            '**target_model': set(['*target_model']),
            '**edesign': set(['*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl']),
            '**dataset': set(['*dataset']),
            '**objfn_builder': set(['*objfn_builder'])
        }

        selector_types = []
        for sel_typ, prop_names in dependencies.items():
            if prop_set.intersection(prop_names):
                selector_types.append(sel_typ)
        return selector_types

    def add_figure_property_selector(self, typ, panel_widget, storage_dict):
        initial_value = storage_dict.get(typ, None)

        def to_val(x):
            return x.replace('\n.', '.')

        def to_txt(x):
            num_leading_dots = sum(1 for _ in _itertools.takewhile(lambda b: b == '.', x))
            return x[0:num_leading_dots] + x[num_leading_dots:].replace('.', '\n.')

        lbl_pc = 0.4
        val_pc = 0.6

        if typ == '**model':
            title_input = TextInput(text='', size_hint=(val_pc, None), height=40)
            storage_dict['**model_title'] = title_input.text

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                row.add_widget(FixedHeightLabel(text='Model Title', size_hint_x=lbl_pc))
                title_input.bind(text=lambda inst, val: storage_dict.__setitem__('**model_title', val))
                row.add_widget(title_input)
                panel_widget.add_widget(row)

            model_names = self.kernel.library_item_names(category='model')
            if initial_value is None or initial_value == '(none)':
                initial_value = model_names[0] if (len(model_names) > 0) else '(none)'
            elif initial_value not in model_names and initial_value != '(none)':
                initial_value = "REMOVED!"
            vals = [to_txt(mn) for mn in model_names]
            max_lines = (max([v.count('\n') for v in vals]) + 1) if len(vals) > 0 else 1
            spinner = Spinner(text=to_txt(initial_value), values=vals, size_hint=(val_pc, None),
                              height=max_lines * 50, sync_height=True)
            storage_dict[typ] = to_val(spinner.text)

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=spinner.height)
                anchor = AnchorLayout(anchor_x='left', anchor_y='center', size_hint_x=lbl_pc)
                anchor.add_widget(FixedHeightLabel(text='Model'))
                row.add_widget(anchor)
                row.add_widget(spinner)
                spinner.bind(text=lambda inst, txt: storage_dict.__setitem__(typ, to_val(txt)))
                panel_widget.add_widget(row)

        elif typ == '**target_model':
            model_names = self.kernel.library_item_names(category='model')
            if initial_value is None or initial_value == '(none)':
                initial_value = model_names[0] if (len(model_names) > 0) else '(none)'
            elif initial_value not in model_names and initial_value != '(none)':
                initial_value = "REMOVED!"
            vals = [to_txt(mn) for mn in model_names]
            max_lines = (max([v.count('\n') for v in vals]) + 1) if len(vals) > 0 else 1
            spinner = Spinner(text=to_txt(initial_value), values=vals, size_hint=(val_pc, None),
                              height=max_lines * 50, sync_height=True)
            storage_dict[typ] = to_val(spinner.text)

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=spinner.height)
                anchor = AnchorLayout(anchor_x='left', anchor_y='center', size_hint_x=lbl_pc)
                anchor.add_widget(FixedHeightLabel(text='Target Model'))
                row.add_widget(anchor)
                row.add_widget(spinner)
                spinner.bind(text=lambda inst, txt: storage_dict.__setitem__(typ, to_val(txt)))
                panel_widget.add_widget(row)

        elif typ == '**edesign':
            edesign_names = self.kernel.library_item_names(category='edesign')
            if initial_value is None or initial_value == '(none)':
                initial_value = edesign_names[0] if (len(edesign_names) > 0) else '(none)'
            elif initial_value not in edesign_names and initial_value != '(none)':
                initial_value = "REMOVED!"
            vals = [to_txt(mn) for mn in edesign_names]
            max_lines = (max([v.count('\n') for v in vals]) + 1) if len(vals) > 0 else 1
            spinner = Spinner(text=to_txt(initial_value), values=vals, size_hint=(val_pc, None),
                              height=max_lines * 50, sync_height=True)
            storage_dict[typ] = to_val(spinner.text)

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=spinner.height)
                anchor = AnchorLayout(anchor_x='left', anchor_y='center', size_hint_x=lbl_pc)
                anchor.add_widget(FixedHeightLabel(text='Exp. design'))
                row.add_widget(anchor)
                row.add_widget(spinner)
                spinner.bind(text=lambda inst, txt: storage_dict.__setitem__(typ, to_val(txt)))
                panel_widget.add_widget(row)

        elif typ == '**dataset':
            ds_names = self.kernel.library_item_names(category='dataset')
            if initial_value is None or initial_value == '(none)':
                initial_value = ds_names[0] if (len(ds_names) > 0) else '(none)'
            elif initial_value not in ds_names and initial_value != '(none)':
                initial_value = "REMOVED!"
            vals = [to_txt(mn) for mn in ds_names]
            max_lines = (max([v.count('\n') for v in vals]) + 1) if len(vals) > 0 else 1
            spinner = Spinner(text=to_txt(initial_value), values=vals, size_hint=(val_pc, None),
                              height=max_lines * 50, sync_height=True)
            storage_dict[typ] = to_val(spinner.text)

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=spinner.height)
                anchor = AnchorLayout(anchor_x='left', anchor_y='center', size_hint_x=lbl_pc)
                anchor.add_widget(FixedHeightLabel(text='Data set'))
                row.add_widget(anchor)
                row.add_widget(spinner)
                spinner.bind(text=lambda inst, txt: storage_dict.__setitem__(typ, to_val(txt)))
                panel_widget.add_widget(row)

        elif typ == '**objfn_builder':
            objfn_builder_names = ['logl', 'chi2', 'from estimate']
            if initial_value is None:
                initial_value = objfn_builder_names[0]
            spinner = Spinner(text=initial_value, values=objfn_builder_names, size_hint=(val_pc, None), height=50)
            storage_dict[typ] = spinner.text

            if panel_widget:
                row = BoxLayout(orientation='horizontal', size_hint_y=None, height=spinner.height)
                row.add_widget(FixedHeightLabel(text='Objective Fn.', size_hint_x=lbl_pc))
                row.add_widget(spinner)
                spinner.bind(text=lambda inst, val: storage_dict.__setitem__(typ, val))
                panel_widget.add_widget(row)

        else:
            raise ValueError("Unknown figure property selector type: %s" % str(typ))

    #def selector_values_to_creation_args(self, figure_selector_vals):  ## REVAMP - creation args should just be *names* of kernel variables
    #    creation_args = {}
    #    #all_properties = ['*models', '*model_titles', '*model', '*model_title', '*target_model',
    #    #                  '*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl',
    #    #                  '*objfn_builder', '*gaugeopt_args', '*estimate_params']
    #
    #    for typ, val in figure_selector_vals.items():
    #        if val == '(none)':
    #            continue  # don't populate any creation args
    #
    #        if typ == '**model_title':
    #            if val:
    #                creation_args['*model_title'] = "'" + val + "'"
    #        elif typ == '**model':
    #            creation_args['*model'] = f"lib.edesign_library['{val}'].model"
    #            creation_args['*model_dim'] = f"lib.edesign_library['{val}'].model.dim"
    #            if '*model_title' not in creation_args:
    #                creation_args['*model_title'] = f"lib.model_library['{val}'].model_name"
    #            if isinstance(self.model_library[val].model_container, _Estimate):
    #                estimate = self.model_library[val].model_container
    #                creation_args['*estimate_params'] = estimate.parameters
    #                creation_args['*unmodeled_error'] = estimate.parameters.get("unmodeled_error", None)
    #                creation_args['*gaugeopt_args'] = estimate.goparameters.get(self.model_library[val].model_name, {})
    #                if isinstance(self.model_library[val].model_container.parent.data.edesign, _StandardGSTDesign):
    #                    max_length_list = self.model_library[val].model_container.parent.data.edesign.maxlengths
    #                    creation_args['*models_by_maxl'] = [estimate.models['iteration %d estimate' % i]
    #                                                        for i in range(len(max_length_list))]
    #        elif typ == '**target_model':
    #            creation_args['*target_model'] = self.model_library[val].model
    #        elif typ == '**edesign':
    #            edesign = self.edesign_library[val].edesign
    #            creation_args['*edesign'] = edesign
    #            creation_args['*circuit_list'] = edesign.all_circuits_needing_data                
    #            creation_args['*circuit_lists'] = edesign.circuit_lists \
    #                if isinstance(edesign, _CircuitListsDesign) else None
    #            if isinstance(edesign, _StandardGSTDesign):
    #                creation_args['*maxlengths'] = edesign.maxlengths
    #                creation_args['*circuits_by_maxl'] = edesign.circuit_lists
    #        elif typ == '**dataset':
    #            dataset = self.dataset_library[val].dataset
    #            creation_args['*dataset'] = dataset
    #        elif typ == '**objfn_builder':
    #            if val == 'from estimate':
    #                if '**model' in figure_selector_vals:
    #                    k = figure_selector_vals['**model']
    #                    mdl_container = self.model_library[k].model_container
    #                    if isinstance(mdl_container, _Estimate):
    #                        creation_args['*objfn_builder'] = mdl_container.parameters.get(
    #                            'final_objfn_builder', _objfns.ObjectiveFunctionBuilder.create_from('logl'))
    #                if '*objfn_builder' not in creation_args:
    #                    print("Warning: could not retrieve objective function from estimate -- using logL instead")
    #                    creation_args['*objfn_builder'] = _objfns.ObjectiveFunctionBuilder.create_from('logl')
    #            else:
    #                creation_args['*objfn_builder'] = _objfns.ObjectiveFunctionBuilder.create_from(val)            
    #
    #    if '*model' in creation_args and '*models' not in creation_args:
    #        creation_args['*models'] = [creation_args['*model']]
    #    if '*model_title' in creation_args and '*model_titles' not in creation_args:
    #        creation_args['*model_titles'] = [creation_args['*model_title']]
    #
    #    return creation_args

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
            '-- Presets --': ['Model Violation Overview', 'Model Violation Detail', 'Gauge Inv. Metrics',
                              'Metrics', 'Raw Model Data', 'Reference'],
        }
        # 'GateMatrixPlot', 'MatrixPlot', DatasetComparisonHistogramPlot, RandomizedBenchmarkingPlot
        # GaugeRobustModelTable, GaugeRobustMetricTable, GaugeRobustErrgenTable, ProfilerTable

        first_child = None
        ret = Accordion(orientation='vertical', height=1000)  # height= just to try to supress initial warning
        for category_name, item_list in items_by_category.items():
            acc_item = CustomAccordionItem(title=category_name)
            #acc_item = AccordionItem(title=category_name, title_template='CustomAccordionTitle')
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
        if self._popup is not None:
            self._popup.dismiss()
            self._popup = None

    def show_load_root_dialog(self):
        content = LoadRootDialog(load=self._load_root, cancel=self.dismiss_popup, initial_path=self.app_path)
        self._popup = Popup(title="Choose root pyGSTi analysis directory to import from", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def _load_root(self, path, filenames, name):
        filename = filenames[0]
        print("Loading root pyGSTi directory: ", path, filename)
        self.dismiss_popup()
        root_path = filename  # filenames contain path, so no need for: _os.path.join(path, filename)
        self.results_dir = pygsti.io.read_results_from_dir(root_path)

        self.ids.results_dir_selector.root_name = (".../" + _os.path.basename(filename)) if len(name) == 0 else name
        self.ids.results_dir_selector.root_file_path = root_path
        self.ids.results_dir_selector.root_results_dir = self.results_dir  # automatic if we make self.results_dir a property

    def open_library(self):
        initial_path = self.library_path if (self.library_path is not None) else self.app_path
        content = OpenDialog(load=self._open_library, cancel=self.dismiss_popup,
                             initial_path=initial_path, filters=['*.json'])
        self._popup = Popup(title="Open library file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def _open_library(self, path, filenames):
        filename = filenames[0]
        self.dismiss_popup()

        #Clear existing library (LATER: also clear analyses?)
        self.ids.edesign_library_list.clear_widgets()
        self.ids.dataset_library_list.clear_widgets()
        self.ids.model_library_list.clear_widgets()

        self.ids.library_info_area.clear_widgets()  # also clear library item info area

        if self.kernel.read_library(filename):
            self._refresh_library_lists()
            print("Loaded library from ", filename)
            self.library_path = filename  # hold copy in front end too
        else:
            print("!!!Error loading library from ", filename)

    def save_library(self):
        initial_path = self.library_path if (self.library_path is not None) else self.app_path
        content = SaveDialog(save=self._save_library, cancel=self.dismiss_popup, initial_path=initial_path)
        self._popup = Popup(title="Save library file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def resave_library(self):
        # only shows dialog when there's no existing analysis_path
        if self.library_path is not None:
            return self._save_library(None, None, None)
        else:
            return self.save_library()

    def _save_library(self, path, selected_filenames, given_filename):
        print("DB: Saving library: ", path, selected_filenames, given_filename)
        if path is None:
            # allow internal calls to _save_library(None, None, None) to re-save library file
            if self.library_path is not None:
                filename = self.library_path
            else:
                return  # nothing to do
        elif len(selected_filenames) == 0:
            if len(given_filename) == 0:
                self.dismiss_popup()
                print("No filename given -- save library operation aborted!")
                return
            filename = _os.path.join(path, given_filename)
        else:
            filename = selected_filenames[0]

        self.dismiss_popup()

        if self.kernel.save_library(filename):
            print("Saved library to ", filename)
            self.library_path = filename
        else:
            print("!!!Error saving library to ", filename)

    def open_analysis(self):
        initial_path = self.analysis_path if (self.analysis_path is not None) \
            else (_os.path.dirname(self.library_path) if (self.library_path is not None) else self.app_path)
        content = OpenDialog(load=self._open_analysis, cancel=self.dismiss_popup,
                             initial_path=initial_path, filters=['*.json'])
        self._popup = Popup(title="Open analysis file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def _open_analysis(self, path, filenames):
        from pygsti.io.metadir import _class_for_name
        filename = filenames[0]
        print("Opening analysis: ", filename)
        self.dismiss_popup()

        tab_figure_info = self.kernel.read_analysis(filename)

        #Close existing tabs
        existing_tabs = list(self.ids.figure_areas.children)
        for tab in existing_tabs:
            if isinstance(tab, CloseableHeader):
                self.ids.figure_areas.remove_widget(tab)
                # TODO: also refresh library somewhere?

        for tab_name, tab_figures in tab_figure_info['tabs'].items():
            self.create_new_tab(tab_name, None)  # create a blank tab and switch to it
            for fig_dict in tab_figures:
                fig_id = fig_dict['figure_id']
                fig_position = fig_dict.get('position', None)
                fig_size = fig_dict.get('size', None)

                figwidget = self.kernel.create_figure_widget(fig_id, fig_dict['selector_values'], scale=1.0)
                fig_property_names = fig_dict['property_names']  #self.kernel.figure_property_names(fig_id)

                #Place figure widget into a container
                fig_container = FigureContainer(figwidget, fig_id, fig_property_names, caption=fig_dict['caption'],
                                                comment=fig_dict.get('comment', ''),
                                                root_widget=self, size_hint=(None, None))
                set_info_containers(fig_container, self.ids.lower_info_area, self.ids.status_label)
                if fig_position is not None: fig_container.pos = fig_position
                if fig_size is not None: fig_container.size = fig_size
                self.active_data_area.add_widget(fig_container)

        self.analysis_path = filename
        if tab_figure_info.get('loaded_library_path', None) is not None:
            self.library_path = tab_figure_info['loaded_library_path']
            self._refresh_library_lists()
        self.ids.analysis_mode_button.trigger_action(duration=0)  # always switch to analysis mode

    def save_analysis(self):
        initial_path = self.analysis_path if (self.analysis_path is not None) \
            else (_os.path.dirname(self.library_path) if (self.library_path is not None) else self.app_path)
        content = SaveDialog(save=self._save_analysis, cancel=self.dismiss_popup, initial_path=initial_path)
        self._popup = Popup(title="Save analysis file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def resave_analysis(self):
        # only shows dialog when there's no existing analysis_path
        if self.analysis_path is not None:
            return self._save_analysis(None, None, None)
        else:
            return self.save_analysis()

    def _save_analysis(self, path, selected_filenames, given_filename):
        print("DB: Saving analysis: ", path, selected_filenames, given_filename)
        if path is None:
            # allow internal calls to _save_analysis(None, None, None) to re-save analysis file
            if self.analysis_path is not None:
                filename = self.analysis_path
            else:
                return  # nothing to do
        elif len(selected_filenames) == 0:
            if len(given_filename) == 0:
                self.dismiss_popup()
                print("No filename given -- save analysis operation aborted!")
                return
            filename = _os.path.join(path, given_filename)
        else:
            filename = selected_filenames[0]

        print("Saving analysis to ", filename)
        self.dismiss_popup()

        frontend_figure_info = []  # maps figure_id -> dict with 'size' and 'position' as list of (k,v) pairs
        for tab in self.ids.figure_areas.tab_list:
            if not isinstance(tab, CloseableHeader):
                continue  # skip the "create a new tab" tab
            for fig_container in tab.content.ids.data_area.children:
                frontend_figure_info.append((fig_container.figure_id,
                                             {'size': fig_container.size,
                                              'position': fig_container.pos,
                                              'comment': fig_container.comment}))
        self.kernel.save_analysis(filename, frontend_figure_info)
        self.analysis_path = filename

    def import_edesign(self, include_children=False):
        """ Note: this should really be "import_protocol_data" since it also imports a dataset if one is present """
        root_name = self.ids.results_dir_selector.root_name
        root_file_path = self.ids.results_dir_selector.root_file_path
        results_dir_node = self.ids.results_dir_selector.selected_results_dir_node
        path_from_root = results_dir_node.path
        #results_dir = results_dir_node.data
        self._import_edesign(root_name, root_file_path, path_from_root, # REMOVE results_dir,
                             import_children=include_children)

    def _import_edesign(self, root_name, root_file_path, path_from_root, # REMOVE results_dir,
                        import_children=False, import_all_models=False):
        self.kernel.import_edesign(root_name, root_file_path, path_from_root, import_children, import_all_models)
        self._refresh_library_lists()

    def _refresh_library_lists(self):
        self.ids.edesign_library_list.clear_widgets()
        self.ids.library_info_area.clear_widgets()

        for name in self.kernel.library_item_names(category='edesign'):
            btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
            btn.bind(state=self.update_library_item_info)
            self.ids.edesign_library_list.add_widget(btn)

        self.ids.dataset_library_list.clear_widgets()
        for name in self.kernel.library_item_names(category='dataset'):
            btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
            btn.bind(state=self.update_library_item_info)
            self.ids.dataset_library_list.add_widget(btn)

        self.ids.model_library_list.clear_widgets()
        for name in self.kernel.library_item_names(category='model'):
            btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
            btn.bind(state=self.update_library_item_info)
            self.ids.model_library_list.add_widget(btn)

        #pth_for_name = [root_name] + path_from_root if root_name else path_from_root
        #name = '.'.join(pth_for_name)  # edesign name
        #
        #btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        #btn.bind(state=self.update_library_item_info)
        #if name not in self.kernel.library_item_names(category='edesign'):
        #
        #    data = results_dir.data  # a ProtocolData object
        #    #self.edesign_library[name] = EdesignLibElement(name, data.edesign, root_file_path, path_from_root, btn)
        #    self.ids.edesign_library_list.add_widget(btn)
        #    #print("Imported edesign: ", name)
        #
        #    if data.dataset is not None:
        #        btn2 = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        #        btn2.bind(state=self.update_library_item_info)
        #        #self.dataset_library[name] = DatasetLibElement(name, data.dataset, root_file_path, path_from_root, btn2)
        #        self.ids.dataset_library_list.add_widget(btn2)
        #        #print("Imported dataset: ", name)
        #
        #    if import_all_models:
        #        self._import_models(root_name, root_file_path, path_from_root, results_dir, 'all', 'all', 'all')
        #else:
        #    print("An e-design with name '%s' is already imported, and will not be clobbered." % name); return
        #
        #if import_children:
        #    for ky in results_dir.keys():
        #        child_path_from_root = path_from_root + [ky]
        #        child_results_dir = results_dir[ky]
        #        self._import_edesign(root_name, root_file_path, child_path_from_root,
        #                             child_results_dir, import_children, import_all_models)

    def import_models(self, include_all=False):
        root_name = self.ids.results_dir_selector.root_name
        results_dir_node = self.ids.results_dir_selector.selected_results_dir_node
        root_file_path = self.ids.results_dir_selector.root_file_path
        path_from_root = results_dir_node.path
        protocol_name = self.ids.results_dir_detail_selector.protocol_name
        additional_detail = self.ids.results_dir_detail_selector.additional_detail
        selected_model_names = self.ids.results_dir_detail_selector.selected_model_names

        #results_dir = results_dir_node.data  #REMOVE

        if include_all:
            self._import_models(root_name, root_file_path, path_from_root, # REMOVE results_dir,
                                'all', 'all', 'all')
        else:
            self._import_models(root_name, root_file_path, path_from_root, # REMOVE results_dir,
                                [protocol_name], [additional_detail], selected_model_names)

    def _import_models(self, root_name, root_file_path, path_from_root, protocol_names,
                       additional_details, model_names):
        self.kernel.import_models(root_name, root_file_path, path_from_root, protocol_names,
                                  additional_details, model_names)
        self._refresh_library_lists()

    def import_all(self):
        root_name = self.ids.results_dir_selector.root_name
        root_file_path = self.ids.results_dir_selector.root_file_path
        #root_results_dir = self.ids.results_dir_selector.root_results_dir
        self._import_edesign(root_name, root_file_path, path_from_root=[], # REMOVE results_dir=root_results_dir,
                             import_children=True, import_all_models=True)

    def import_model_from_file(self):
        initial_path = self.library_path if (self.library_path is not None) else self.app_path
        content = OpenDialog(load=self._import_model_from_file, cancel=self.dismiss_popup,
                             initial_path=initial_path, filters=['*.json'])
        self._popup = Popup(title="Open model file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def _import_model_from_file(self, path, filenames):
        print("Importing model file: ", filenames[0])
        self.dismiss_popup()
        self.kernel.import_models_from_files(filenames)
        self._refresh_library_lists()

    def import_dataset_from_file(self):
        initial_path = self.library_path if (self.library_path is not None) else self.app_path
        content = OpenDialog(load=self._import_dataset_from_file, cancel=self.dismiss_popup,
                             initial_path=initial_path, filters=['*.txt'])
        self._popup = Popup(title="Open data set file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def _import_dataset_from_file(self, path, filenames):
        print("Importing dataset file: ", filenames[0])
        self.dismiss_popup()
        self.kernel.import_datasets_from_files(filenames)
        self._refresh_library_lists()

    def update_library_item_info(self, togglebtn, val):
        info_area = self.ids.library_info_area
        info_area.clear_widgets()

        def add_info_row(widget, k, v, clickable=False):
            row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            key_lbl = Label(text=str(k), font_size=24, size_hint_y=None, height=40)
            key_anchor = AnchorLayout(anchor_x='center', anchor_y='center'); key_anchor.add_widget(key_lbl)
            row.add_widget(key_anchor)
            if clickable:
                value_lbl = WrappedClickableLabel(text=str(v), font_size=24)
            else:
                value_lbl = WrappedLabel(text=str(v), font_size=24)
            value_lbl.bind(height=lambda _, h: setattr(row, 'height', h))  # so row height adjusts with label
            row.add_widget(value_lbl)
            widget.add_widget(row)
            return value_lbl

        def show_name_popup(title, category, existing_name, item_widget):
            def update_name(popup, new_name):
                self.kernel.update_library_item_name(category, existing_name, new_name)
                item_widget.text = new_name
                self._popup.dismiss()
                self.update_library_item_info(item_widget, 'down')

            popup_content = NameDialog(text=existing_name,
                                       update=update_name,
                                       cancel=lambda: self._popup.dismiss())
            self._popup = Popup(title=title, content=popup_content,
                                size_hint=(0.6, None), height=300)
            self._popup.open()

        if val == 'down':
            key = togglebtn.text
            delete_btn = Button(text='Delete Item', size_hint_y=None, height=40, font_size=22, color=(1, 0, 0, 1))

            if togglebtn in self.ids.edesign_library_list.children:
                item_info = self.kernel.library_item_info('edesign', key)
                add_info_row(info_area, 'item type:', 'experiment design')
                name_val = add_info_row(info_area, 'name:', item_info['name'], clickable=True)
                add_info_row(info_area, 'root path:', item_info['root_path'])
                add_info_row(info_area, 'path from root:', str(item_info['path_from_root']))
                name_val.bind(on_release=lambda *x: show_name_popup('Update name', 'edesign', key, togglebtn))
                delete_btn.bind(on_release=lambda *x: self.remove_library_item('edesign', key, togglebtn))

            elif togglebtn in self.ids.dataset_library_list.children:
                item_info = self.kernel.library_item_info('dataset', key)
                add_info_row(info_area, 'item type:', 'data set')
                name_val = add_info_row(info_area, 'name:', item_info['name'], clickable=True)
                add_info_row(info_area, 'root path:', item_info['root_path'])
                add_info_row(info_area, 'path from root:', str(item_info['path_from_root']))
                name_val.bind(on_release=lambda *x: show_name_popup('Update name', 'dataset', key, togglebtn))
                delete_btn.bind(on_release=lambda *x: self.remove_library_item('dataset', key, togglebtn))
                
            elif togglebtn in self.ids.model_library_list.children:
                item_info = self.kernel.library_item_info('model', key)
                add_info_row(info_area, 'item type:', 'model')
                name_val = add_info_row(info_area, 'name:', item_info['name'], clickable=True)
                add_info_row(info_area, 'root path:', item_info['root_path'])
                add_info_row(info_area, 'path from root:', str(item_info['path_from_root']))
                add_info_row(info_area, 'protocol:', str(item_info['protocol_name']))
                for k, v in item_info['additional_details'].items():
                    add_info_row(info_area, k, str(v))
                name_val.bind(on_release=lambda *x: show_name_popup('Update name', 'model', key, togglebtn))
                delete_btn.bind(on_release=lambda *x: self.remove_library_item('model', key, togglebtn))

            info_area.add_widget(delete_btn)
            #print("Down from ", togglebtn.text)
        else:
            #print(val, " from ", togglebtn.text)
            pass

    def remove_library_item(self, category, name, item_widget):
        self.kernel.remove_library_item(category, name)
        #self._refresh_library_lists()

        #Could also just do a list refresh instead of this:
        self.ids.library_info_area.clear_widgets()
        if category == 'edesign':
            self.ids.edesign_library_list.remove_widget(item_widget)
        elif category == 'dataset':
            self.ids.dataset_library_list.remove_widget(item_widget)
        elif category == 'model':
            self.ids.model_library_list.remove_widget(item_widget)

    def add_item(self, item_text, positioning_option_overrides=None):
        print("Adding item ", item_text)

        if item_text in ['Model Violation Overview', 'Model Violation Detail', 'Gauge Inv. Metrics',
                         'Metrics', 'Raw Model Data', 'Reference']:  # NOTE: this list is duplicated in add_new_tab_preset_buttons!
            return self.add_preset(item_text)

        #cri = None
        active_tab_name = self.ids.figure_areas.current_tab.text

        options = self.positioning_options.copy()
        if positioning_option_overrides:
            options.update(positioning_option_overrides)
        scale = float(options.get('scale', '1.0'))

        if self.active_data_area is None:
            print("No active data area!  Must create or switch to a data tab before creating figures.")
            return

        existing_figs = list(self.active_data_area.children)
        fig_id = self.kernel.add_figure(active_tab_name, figure_type=item_text,
                                        name=item_text)  # TODO: better in future
        if fig_id is None:
            print("Cannot create " + item_text + " yet.")
            return

        figwidget = self.kernel.create_figure_widget(fig_id, self.default_figure_selector_vals, scale) #, options?)
        fig_property_names = self.kernel.figure_property_names(fig_id)

        #Place figure widget into a container
        fig_container = FigureContainer(figwidget, fig_id, fig_property_names, caption=item_text,
                                        comment="", root_widget=self, size_hint=(None, None))
        for k in fig_container.selector_vals.keys():
            fig_container.selector_vals[k] = self.default_figure_selector_vals[k]  # initialize default selector values

        set_info_containers(fig_container, self.ids.lower_info_area, self.ids.status_label)
        self.active_data_area.add_widget(fig_container)

        fig_size = figwidget.size
        arrange_mode = options.get('arrange', 'top down')

        if arrange_mode == 'top down':
            x = 0  # in this mode, x always == 0
            min_y = min([c.y for c in existing_figs]) if len(existing_figs) > 0 \
                else self.active_data_area.height
            #y = max(min_y - fig_size[1], 0)  # don't let y be < 0
            y = min_y - fig_size[1] - 100
            fig_container.pos = (x, y)

        elif arrange_mode == 'left right':
            min_y = min([c.y for c in existing_figs]) if len(existing_figs) > 0 \
                else self.active_data_area.height
            y = min_y  # place new figure level with lowest current figure
            max_x = max([(c.x + c.width) for c in existing_figs]) if len(existing_figs) > 0 else 0
            x = max_x + 100
            fig_container.pos = (x, y)

        elif arrange_mode == 'all in center':
            x = (self.active_data_area.size[0] - fig_size[0]) / 2
            y = (self.active_data_area.size[1] - fig_size[1]) / 2
            fig_container.pos = (x, y)
        else:
            raise ValueError("Invalid arrange mode: %s" % str(arrange_mode))



#REMOVE
#        
#        extra_capsule_args = dict(caption=item_text,
#                                  info_sidebar=self.ids.info_layout,
#                                  status_label=self.ids.status_label,
#                                  root_widget=self)
#
#        #default_figure_creation_args = self.selector_values_to_creation_args(self.default_figure_selector_vals)
#        if item_text == 'SpamTable':
#            #wstable = ws.SpamTable(models, titles, 'boxes', cri, False)  # titles?
#            figure_capsule = FigureCapsule(self.ws.SpamTable, ['*models', '*model_titles', 'boxes', cri, False],
#                                           **extra_capsule_args)
#        elif item_text == 'SpamParametersTable':
#            #wstable = ws.SpamParametersTable(models, titles, cri)
#            figure_capsule = FigureCapsule(self.ws.SpamParametersTable, ['*models', '*model_titles', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'GatesTable':
#            #wstable = ws.GatesTable(models, titles, 'boxes', cri)
#            figure_capsule = FigureCapsule(self.ws.GatesTable, ['*models', '*model_titles', 'boxes', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'ChoiTable':
#            #wstable = ws.ChoiTable(models, titles, cri)
#            figure_capsule = FigureCapsule(self.ws.ChoiTable, ['*models', '*model_titles', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'ModelVsTargetTable':
#            clifford_compilation = None
#            #wstable = ws.ModelVsTargetTable(model, target_model, clifford_compilation, cri)
#            figure_capsule = FigureCapsule(self.ws.ModelVsTargetTable, ['*model', '*target_model',
#                                                                        clifford_compilation, cri],
#                                           **extra_capsule_args)
#        elif item_text == 'GatesVsTargetTable':
#            #wstable = ws.GatesVsTargetTable(model, target_model, cri)  # wildcard?
#            figure_capsule = FigureCapsule(self.ws.GatesVsTargetTable, ['*model', '*target_model', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'SpamVsTargetTable':
#            #wstable = ws.SpamVsTargetTable(model, target_model, cri)
#            figure_capsule = FigureCapsule(self.ws.SpamVsTargetTable, ['*model', '*target_model', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'ErrgenTable':
#            #wstable = ws.ErrgenTable(model, target_model, cri)  # (more options)
#            figure_capsule = FigureCapsule(self.ws.ErrgenTable, ['*model', '*target_model', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'NQubitErrgenTable':
#            #wstable = ws.NQubitErrgenTable(model, cri)
#            figure_capsule = FigureCapsule(self.ws.NQubitErrgenTable, ['*model', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'GateDecompTable':
#            #wstable = ws.GateDecompTable(model, target_model, cri)
#            figure_capsule = FigureCapsule(self.ws.GateDecompTable, ['*model', '*target_model', cri],
#                                           **extra_capsule_args)
#        elif item_text == 'GateEigenvalueTable':
#            #wstable = ws.GateEigenvalueTable(model, target_model, cri,
#            #                                 display=('evals', 'rel', 'log-evals', 'log-rel'))
#            figure_capsule = FigureCapsule(self.ws.GateEigenvalueTable, ['*model', '*target_model', cri,
#                                                                         ('evals', 'rel', 'log-evals', 'log-rel')],
#                                           **extra_capsule_args)
#        elif item_text == 'DataSetOverviewTable':
#            #wstable = ws.DataSetOverviewTable(dataset, max_length_list)
#            figure_capsule = FigureCapsule(self.ws.DataSetOverviewTable, ['*dataset', '*maxlengths'],
#                                           **extra_capsule_args)
#        elif item_text == 'SoftwareEnvTable':
#            #wstable = ws.SoftwareEnvTable()
#            figure_capsule = FigureCapsule(self.ws.SoftwareEnvTable, [], **extra_capsule_args)
#        elif item_text == 'CircuitTable':
#            # wstable = ws.CircuitTable(...)  # wait until we can select circuit list; e.g. germs, fiducials
#            print("Wait until better selection methods to create circuit tables...")
#            figure_capsule = None
#        elif item_text == 'GatesSingleMetricTable':
#            #metric = 'inf'  # entanglement infidelity
#            #wstable = GatesSingleMetricTable(metric, ...)
#            print("Wait until better selection methods to create single-item gate metric tables...")
#            figure_capsule = None
#        elif item_text == 'StandardErrgenTable':
#            #wstable = ws.StandardErrgenTable(model.dim, 'hamiltonian', 'pp')  # not super useful; what about 'stochastic'?
#            figure_capsule = FigureCapsule(self.ws.StandardErrgenTable, ['*model_dim', 'H', 'pp'],
#                                           **extra_capsule_args)
#        elif item_text == 'GaugeOptParamsTable':
#            #wstable = ws.GaugeOptParamsTable(gaugeopt_args)
#            figure_capsule = FigureCapsule(self.ws.GaugeOptParamsTable, ['*gaugeopt_args'],
#                                           **extra_capsule_args)
#        elif item_text == 'MetadataTable':
#            #wstable = ws.MetadataTable(model, estimate_params)
#            figure_capsule = FigureCapsule(self.ws.MetadataTable, ['*model', '*estimate_params'],
#                                           **extra_capsule_args)
#        elif item_text == 'WildcardBudgetTable':
#            #wstable = ws.WildcardBudgetTable(estimate_params.get("unmodeled_error", None))
#            figure_capsule = FigureCapsule(self.ws.WildcardBudgetTable, ['*unmodeled_error'],
#                                           **extra_capsule_args)
#        elif item_text == 'FitComparisonTable':
#            #wstable = ws.FitComparisonTable(max_length_list, circuits_by_L, models_by_L, dataset)
#            figure_capsule = FigureCapsule(self.ws.FitComparisonTable, ['*maxlengths', '*circuits_by_maxl',
#                                                                        '*models_by_maxl', '*dataset',
#                                                                        '*objfn_builder'],
#                                           **extra_capsule_args)
#        elif item_text == 'FitComparisonBarPlot':
#            #wsplot = ws.FitComparisonBarPlot(max_length_list, circuits_by_L, models_by_L, dataset)
#            figure_capsule = FigureCapsule(self.ws.FitComparisonBarPlot, ['*maxlengths', '*circuits_by_maxl',
#                                                                          '*models_by_maxl', '*dataset'],
#                                           **extra_capsule_args)
#        elif item_text == 'FitComparisonBarPlotB':
#            #wsplot = ws.FitComparisonBarPlot(est_lbls_mt, [circuit_list] * len(est_mdls_mt),
#            #                                 est_mdls_mt, [dataset] * len(est_mdls_mt), objfn_builder)
#            def multiplx(titles, circuit_list, models, dataset, objfn_builder):
#                return self.ws.FitComparisonBarPlot(titles, [circuit_list] * len(titles),
#                                                    models, [dataset] * len(titles), objfn_builder)
#            figure_capsule = FigureCapsule(multiplx, ['*model_titles', '*circuit_list', '*models', '*dataset',
#                                                      '*objfn_builder'], **extra_capsule_args)
#        elif item_text == 'FitComparisonBoxPlot':
#            # used for multiple data sets -- enable this once we get better selection methods
#            print("Wait until better selection methods to create fit comparison box plot...")
#            figure_capsule = None
#        elif item_text in ('ColorBoxPlot', 'ColorScatterPlot', 'ColorHistogramPlot'):
#
#            if item_text == 'ColorBoxPlot': plot_type = "boxes"
#            elif item_text == "ColorScatterPlot": plot_type = "scatter"
#            else: plot_type = "histogram"
#
#            linlog_percentile = 5
#            #bgcolor = 'white'
#            #wsplot = ws.ColorBoxPlot(objfn_builder, circuit_list, dataset, model,
#            #    linlg_pcntle=linlog_percentile / 100, comm=None, bgcolor=bgcolor, typ=plot_type)
#            figure_capsule = FigureCapsule(self.ws.ColorBoxPlot,
#                                           ['*objfn_builder', '*circuit_list', '*dataset', '*model',
#                                            False, False, True, False, 'compact', linlog_percentile / 100,
#                                            None, None, None, None, None, plot_type],
#                                           **extra_capsule_args)
#
#        elif item_text in ['Model Violation Overview', 'Model Violation Detail', 'Gauge Inv. Metrics',
#                           'Metrics', 'Raw Model Data', 'Reference']:  # NOTE: this list is duplicated in add_new_tab_preset_buttons!
#            return self.add_preset(item_text)
#        else:
#            figure_capsule = None
#
#        if figure_capsule is not None:
#            options = self.positioning_options.copy()
#            if positioning_option_overrides:
#                options.update(positioning_option_overrides)
#            
#            scale = float(options.get('scale', '1.0'))
#            existing_figs = list(self.active_data_area.children)
#            figure_capsule.fill_args_from_creation_arg_dict(default_figure_creation_args)
#            fig_size = figure_capsule.update_figure_widget(self.active_data_area, scale)
#
#            arrange_mode = options.get('arrange', 'top down')
#            if arrange_mode == 'top down':
#                x = 0  # in this mode, x always == 0
#                min_y = min([c.y for c in existing_figs]) if len(existing_figs) > 0 \
#                    else self.active_data_area.height
#                #y = max(min_y - fig_size[1], 0)  # don't let y be < 0
#                y = min_y - fig_size[1] - 100
#                figure_capsule.fig_container.pos = (x, y)
#
#            elif arrange_mode == 'left right':
#                min_y = min([c.y for c in existing_figs]) if len(existing_figs) > 0 \
#                    else self.active_data_area.height
#                y = min_y  # place new figure level with lowest current figure
#                max_x = max([(c.x + c.width) for c in existing_figs]) if len(existing_figs) > 0 else 0
#                x = max_x + 100
#                figure_capsule.fig_container.pos = (x, y)
#
#            elif arrange_mode == 'all in center':
#                x = (self.active_data_area.size[0] - fig_size[0]) / 2
#                y = (self.active_data_area.size[1] - fig_size[1]) / 2
#                figure_capsule.fig_container.pos = (x, y)
#            else:
#                raise ValueError("Invalid arrange mode: %s" % str(arrange_mode))
#        else:
#            print("Cannot create " + item_text + " yet.")

    #def old_add_item(self, item_text):  # REMOVE
    #    print("OLD Adding item ", item_text)
    #
    #    resultsdir = self.resultsdir_selector_widget.selected_results_dir
    #    data = resultsdir.data
    #    edesign = data.edesign
    #
    #    results = self.results_selector_widget.selected_results
    #
    #    circuit_list = edesign.all_circuits_needing_data
    #    dataset = data.dataset
    #
    #    if isinstance(edesign, _StandardGSTDesign):
    #        max_length_list = edesign.maxlengths
    #        circuits_by_L = edesign.circuit_lists
    #    else:
    #        max_length_list = None
    #        circuits_by_L = None
    #
    #    if isinstance(results, _ModelEstimateResults):
    #        estimate = results.estimates[self.resultdetail_selector_widget.estimate_name]
    #        model = estimate.models[self.resultdetail_selector_widget.model_name]
    #        target_model = estimate.models['target'] if 'target' in estimate.models else None
    #        models = [model]
    #        titles = ['Estimate']
    #        objfn_builder = estimate.parameters.get(
    #            'final_objfn_builder', _objfns.ObjectiveFunctionBuilder.create_from('logl'))
    #        models_by_L = [estimate.models['iteration %d estimate' % i] for i in range(len(max_length_list))] \
    #            if (max_length_list is not None) else None
    #        est_lbls_mt = [est_name for est_name in results.estimates if est_name != "Target"]
    #        est_mdls_mt = [results.estimates[est_name].models.get('final iteration estimate', None)
    #                       for est_name in est_lbls_mt]
    #        gaugeopt_args = estimate.goparameters.get(self.resultdetail_selector_widget.model_name, {})
    #        estimate_params = estimate.parameters
    #    else:
    #        estimate = model = target_model = None
    #        models = titles = []
    #        objfn_builder = None
    #        models_by_L = None
    #        est_lbls_mt = None
    #        est_mdls_mt = None
    #        gaugeopt_args = {}
    #        estimate_params = {}
    #    cri = None
    #
    #    ws = Workspace(gui_mode='kivy')
    #    wstable = None
    #    wsplot = None
    #    if item_text == 'SpamTable':
    #        wstable = ws.SpamTable(models, titles, 'boxes', cri, False)  # titles?
    #    elif item_text == 'SpamParametersTable':
    #        wstable = ws.SpamParametersTable(models, titles, cri)
    #    elif item_text == 'GatesTable':
    #        wstable = ws.GatesTable(models, titles, 'boxes', cri)
    #    elif item_text == 'ChoiTable':
    #        wstable = ws.ChoiTable(models, titles, cri)
    #    elif item_text == 'ModelVsTargetTable':
    #        clifford_compilation = None
    #        wstable = ws.ModelVsTargetTable(model, target_model, clifford_compilation, cri)
    #    elif item_text == 'GatesVsTargetTable':
    #        wstable = ws.GatesVsTargetTable(model, target_model, cri)  # wildcard?
    #    elif item_text == 'SpamVsTargetTable':
    #        wstable = ws.SpamVsTargetTable(model, target_model, cri)
    #    elif item_text == 'ErrgenTable':
    #        wstable = ws.ErrgenTable(model, target_model, cri)  # (more options)
    #    elif item_text == 'NQubitErrgenTable':
    #        wstable = ws.NQubitErrgenTable(model, cri)
    #    elif item_text == 'GateDecompTable':
    #        wstable = ws.GateDecompTable(model, target_model, cri)
    #    elif item_text == 'GateEigenvalueTable':
    #        wstable = ws.GateEigenvalueTable(model, target_model, cri,
    #                                         display=('evals', 'rel', 'log-evals', 'log-rel'))
    #    elif item_text == 'DataSetOverviewTable':
    #        wstable = ws.DataSetOverviewTable(dataset, max_length_list)
    #    elif item_text == 'SoftwareEnvTable':
    #        wstable = ws.SoftwareEnvTable()
    #    elif item_text == 'CircuitTable':
    #        # wstable = ws.CircuitTable(...)  # wait until we can select circuit list; e.g. germs, fiducials
    #        print("Wait until better selection methods to create circuit tables...")
    #    elif item_text == 'GatesSingleMetricTable':
    #        #metric = 'inf'  # entanglement infidelity
    #        #wstable = GatesSingleMetricTable(metric, ...)
    #        print("Wait until better selection methods to create single-item gate metric tables...")
    #    elif item_text == 'StandardErrgenTable':
    #        wstable = ws.StandardErrgenTable(model.dim, 'hamiltonian', 'pp')  # not super useful; what about 'stochastic'?
    #    elif item_text == 'GaugeOptParamsTable':
    #        wstable = ws.GaugeOptParamsTable(gaugeopt_args)
    #    elif item_text == 'MetadataTable':
    #        wstable = ws.MetadataTable(model, estimate_params)
    #    elif item_text == 'WildcardBudgetTable':
    #        wstable = ws.WildcardBudgetTable(estimate_params.get("unmodeled_error", None))
    #    elif item_text == 'FitComparisonTable':
    #        wstable = ws.FitComparisonTable(max_length_list, circuits_by_L, models_by_L, dataset)
    #    elif item_text == 'FitComparisonBarPlot':
    #        wsplot = ws.FitComparisonBarPlot(max_length_list, circuits_by_L, models_by_L, dataset)
    #    elif item_text == 'FitComparisonBarPlotB':
    #        wsplot = ws.FitComparisonBarPlot(est_lbls_mt, [circuit_list] * len(est_mdls_mt),
    #                                         est_mdls_mt, [dataset] * len(est_mdls_mt), objfn_builder)
    #
    #    elif item_text == 'FitComparisonBoxPlot':
    #        # used for multiple data sets -- enable this once we get better selection methods
    #        print("Wait until better selection methods to create fit comparison box plot...")
    #    elif item_text in ('ColorBoxPlot', 'ColorScatterPlot', 'ColorHistogramPlot'):
    #
    #        if item_text == 'ColorBoxPlot': plot_type = "boxes"
    #        elif item_text == "ColorScatterPlot": plot_type = "scatter"
    #        else: plot_type = "histogram"
    #
    #        linlog_percentile = 5
    #        bgcolor = 'white'
    #        wsplot = ws.ColorBoxPlot(
    #            objfn_builder, circuit_list,
    #            dataset, model,  # could use non-gauge-opt model here?
    #            linlg_pcntle=linlog_percentile / 100, comm=None, bgcolor=bgcolor,
    #            typ=plot_type)
    #
    #    else:
    #        wstable = wsplot = None
    #
    #    if wstable is not None:
    #        tbl = wstable.tables[0]
    #        out = tbl.render('kivywidget', kivywidget_kwargs={'size_hint': (None, None)})
    #        tblwidget = out['kivywidget']
    #        #self.active_data_area.clear_widgets()
    #        fig = FigureContainer(tblwidget, item_text, size_hint=(None, None))
    #        set_info_containers(fig, self.ids.info_layout, self.ids.status_label)
    #        self.active_data_area.add_widget(fig)
    #    elif wsplot is not None:
    #        plt = wsplot.figs[0]
    #        constructor_fn, kwargs = plt.kivywidget
    #        natural_size = plt.metadata.get('natural_size', (300, 300))
    #        kwargs.update({'size_hint': (None, None)})
    #        pltwidget = constructor_fn(**kwargs)
    #        pltwidget.size = natural_size
    #        print("DB: PLOT Initial size = ", natural_size)
    #        #self.active_data_area.clear_widgets()
    #        fig = FigureContainer(pltwidget, item_text, size_hint=(None, None))
    #        set_info_containers(fig, self.ids.info_layout, self.ids.status_label)
    #        self.active_data_area.add_widget(fig)
    #    else:
    #        print("Cannot create " + item_text + " yet.")

    def add_preset(self, preset_name):
        if preset_name == 'Model Violation Overview':
            self.add_item('FitComparisonBarPlot', {'arrange': 'top down'})
            self.add_item('ColorHistogramPlot', {'arrange': 'left right'})

        elif preset_name == 'Model Violation Detail':
            self.add_item('FitComparisonTable', {'arrange': 'top down'})
            self.add_item('ColorBoxPlot', {'arrange': 'top down'})

        elif preset_name == 'Gauge Inv. Metrics':
            self.add_item('GateEigenvalueTable', {'arrange': 'top down'})
            self.add_item('ModelVsTargetTable', {'arrange': 'top down'})
            self.add_item('SpamParametersTable', {'arrange': 'left right'})

        elif preset_name == 'Metrics':
            self.add_item('GatesVsTargetTable', {'arrange': 'top down'})
            self.add_item('SpamVsTargetTable', {'arrange': 'top down'})
            self.add_item('ErrgenTable', {'arrange': 'top down'})

        elif preset_name == 'Raw Model Data':
            self.add_item('SpamTable', {'arrange': 'top down'})
            self.add_item('GatesTable', {'arrange': 'left right'})
            self.add_item('ChoiTable', {'arrange': 'top down'})

        elif preset_name == 'Reference':
            self.add_item('DataSetOverviewTable', {'arrange': 'top down'})
            self.add_item('MetadataTable', {'arrange': 'left right'})
            self.add_item('SoftwareEnvTable', {'arrange': 'top down'})
        else:
            raise ValueError("Invalid preset name: %s" % str(preset_name))

    def add_new_tab_preset_buttons(self, tab_content):

        def new_tab_from_preset(btn_obj):
            preset_nm = btn_obj.text
            self.create_new_tab(self.ids.new_tab_name.text, preset_nm)
            self.ids.new_tab_name.text = ''  # reset custom tab name

        for preset_name in ['Model Violation Overview', 'Model Violation Detail', 'Gauge Inv. Metrics',
                            'Metrics', 'Raw Model Data', 'Reference']:
            btn = Button(text=preset_name, size_hint_y=None, height=40)
            btn.bind(on_press=new_tab_from_preset)
            tab_content.add_widget(btn)

    def create_new_tab(self, tab_text, preset_name):
        if len(tab_text) == 0:
            if preset_name is None:
                existing_tab_names = set([t.text for t in self.ids.figure_areas.tab_list])
                tab_text = 'Custom Analysis'; i = 2
                while tab_text in existing_tab_names:
                    tab_text = 'Custom Analysis %d' % i; i += 1
            else:
                tab_text = preset_name
        tab_header = CloseableHeader(text=tab_text)
        tab_header.content = FigureAreaWidget(root_widget=self)

        self.ids.figure_areas.add_widget(tab_header, index=1)  # add to right of '+' tab
        self.ids.figure_areas.switch_to(tab_header)
        if preset_name is not None:
            self.add_preset(preset_name)

    def close_tab(self, tab):
        self.ids.figure_areas.remove_widget(tab)
        assert(len(self.ids.figure_areas.tab_list) > 0)  # should never be able to remove creation tab
        self.ids.figure_areas.switch_to(self.ids.figure_areas.tab_list[-1])

    def move_tab(self, tab, direction):
        #if direction == 'left':
        current_tab = self.ids.figure_areas.current_tab
        tabs = list(self.ids.figure_areas.tab_list)
        index = tabs.index(tab)
        if direction == 'left' and index < len(tabs) - 1:
            tabs.pop(index)
            tabs.insert(index + 1, tab)
        elif direction == 'right' and index > 1:
            tabs.pop(index)
            tabs.insert(index - 1, tab)
        self.ids.figure_areas.clear_widgets()
        for i, tab in enumerate(tabs):
            self.ids.figure_areas.add_widget(tab, index=i)
        self.ids.figure_areas.switch_to(current_tab)


class TreeViewLabelWithData(TreeViewLabel):
    def __init__(self, path, data, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.data = data


class BorderedBoxLayout(BoxLayout):
    thickness = BoundedNumericProperty(3, min=0)
    color = ListProperty([0.3, 0.3, 0.3, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.after:
            self._color = Color(*self.color)
            self.border_rect = Line(points=[], width=self.thickness)
        self._update_border()
        self.bind(color=lambda instr, value: setattr(self._color, "rgba", value))
        self.bind(size=self._update_border, pos=self._update_border)

    def _update_border(self, *args):
        t = self.thickness
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
        tv = TreeView(root_options=dict(text=('From ' + self.root_name)),
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
        results = results_dir.for_protocol[new_protocol_key] if new_protocol_key != '(none)' else None
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
            self.model_list_widget.clear_widgets()  #HACK

    def on_change_selected_estimate(self, spinner, new_estimate_key):
        #Note: this is only called when self.results is a ModelEstimateResults object
        #if len(self.rows) > 1:
        #    self.remove_widget(self.rows[1])  # remove second row == "Model: ..." row

        self.additional_detail['estimate_name'] = new_estimate_key
        self.rebuild()

        if new_estimate_key is not None and new_estimate_key != '(none)':
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


class FigureAreaWidget(ScrollView):  #BoxLayout, StencilView):
    # needs menus of all available tables/plots to add (for currently selected results/model/data/gaugeopt, etc)
    #resultsdir_selector_widget = ObjectProperty(None, allownone=True)
    #results_selector_widget = ObjectProperty(None, allownone=True)
    #resultdetail_selector_widget = ObjectProperty(None, allownone=True)
    root_widget = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        #kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

    def on_size(self, *args):
        if len(self.children) > 0:
            data_area = self.children[0]
            if self.width > data_area.minimal_size[0]:
                data_area.width = self.width
            if self.height > data_area.minimal_size[1]:
                data_area.height = self.height
            print("Figure area resize => data area size = ", data_area.size)
        
    #def on_results_selector_widget(self, inst, val):
    #    self.results_selector_widget.bind(selected_results=self.selection_change)
    #
    #def on_resultdetail_selector_widget(self, inst, val):
    #    self.resultdetail_selector_widget.bind(estimate_name=self.selection_change, model_name=self.selection_change)
    #
    #def selection_change(self, instance, value):
    #    print("Data area noticed a selected results or model change... do something in the future?")


class DataAreaWidget(RelativeLayout):  #ScatterLayout):
    root_widget = ObjectProperty(None, allownone=True)
    minimal_size = ListProperty([0, 0])  # minimum size needed to contain children (figures)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            Color(0.6, 0.6, 0.6, 1)  # gray
            self._bgrect = Rectangle(pos=(0,0), size=self.size)  # note: *relative* coords
        self.bind(size=self._draw)
        self._recomputing_min_size = False

    def _draw(self, *args):
        self._bgrect.size = self.size

    def add_widget(self, widget):
        widget.bind(pos=self.recompute_minimal_size, size=self.recompute_minimal_size)
        super().add_widget(widget)

    def recompute_minimal_size(self, *args):
        if len(self.children) == 0:
            self.minimal_size[:] = (0, 0)
        else:
            if self._recomputing_min_size:
                return  # don't do anything if we're already computing the minimal size
            # (recursion triggered by setting c.pos below)

            self._recomputing_min_size = True
            print("First child pos = ", self.children[0].pos, 'size=',self.children[0].size)
            max_x = max([(c.x + c.width) for c in self.children])
            max_y = max([(c.y + c.height) for c in self.children])
            min_x = min([c.x for c in self.children])
            min_y = min([c.y for c in self.children])
            print("min x,y = ", min_x, min_y, " max x,y = ", max_x, max_y)

            shift_x = -min_x if min_x < 0 else 0
            shift_y = -min_y if min_y < 0 else 0
            if shift_x > 0 or shift_y > 0:
                for c in self.children:
                    c.pos = (c.x + shift_x, c.y + shift_y)
                    Clock.schedule_once(lambda dt: c.content._redraw() if isinstance(c.content, TableWidget) else None)
                    #Above call fixes issue whereby table _redraw is called before cell positions are updated,
                    # causing the table lines to be drawn incorrectly.  Another pass (on the next frame) fixes this.

            self.minimal_size[:] = (max_x + shift_x, max_y + shift_y)
            self._recomputing_min_size = False

        if self.size[0] < self.minimal_size[0]:
            self.width = self.minimal_size[0]

        if self.size[1] < self.minimal_size[1]:
            self.height = self.minimal_size[1]
        print("Data area size = ", self.size)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print("data area touch down")
            if self.root_widget is None:
                return super().on_touch_down(touch)

            #print("TOUCH at ", touch.pos)
            #print("TOUCH -> local ", self.to_local(*touch.pos))
            #See if touch should active a figure container
            touch.push()
            touch.apply_transform_2d(self.to_local)  # because DataAreaWidget is a RelativeLayout
            for figc in self.children:  # loop over figure containers
                if figc.collide_point(*touch.pos):
                    print("Figure %s received touch-down event" % figc.caption)
                    self.root_widget.set_active_figure_container(figc)
                    break
            else:
                print("no collision with any figure container")
                self.root_widget.set_active_figure_container(None)
            touch.pop()

            # don't count activation as actual 'processing', so continue on and
            # let super decide whether this event is processed
            return super().on_touch_down(touch)
        else:
            return False


class FigureContainer(DragBehavior, ResizableBehavior, BoxLayout):
    def __init__(self, fig_widget, fig_id, fig_property_names, caption, comment, root_widget, **kwargs):
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

        self.figure_id = fig_id
        self.size = initial_size
        self.caption = caption
        self.comment = comment
        self.add_widget(Label(text=caption, bold=True, size_hint_y=None, height=50, color=(0,0,0,1), font_size=18))
        self.content = None
        self.set_content(fig_widget)  # sets self.content
        self.root_widget = root_widget

        self.property_names = fig_property_names

        #Initialize self.selector_vals from arguments and root_widget
        self.selector_vals = {}  # replicated on front and back ends
        selector_types = self.root_widget.selector_types_for_properties(self.property_names)
        for typ in selector_types:
            self.root_widget.add_figure_property_selector(typ, None, storage_dict=self.selector_vals)

        #self.set_cursor_mode(0)
        #Merge into figure container: root_widget, info_sidebar, status_label

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

    def populate_figure_property_panel(self, panel_widget):
        panel_widget.clear_widgets()
        panel_widget.add_widget(Label(text=self.caption, bold=True, size_hint_y=None, height=40))
        selector_types = self.root_widget.selector_types_for_properties(self.property_names)
        for typ in selector_types:
            self.root_widget.add_figure_property_selector(typ, panel_widget, storage_dict=self.selector_vals)
        btn = Button(text='Update', size_hint_y=None, height=40)
        btn.bind(on_release=self.update_figure)
        panel_widget.add_widget(btn)

    def update_figure(self, *args):
        scale = float(self.root_widget.positioning_options.get('scale', "1.0"))
        fig_widget = self.root_widget.kernel.create_figure_widget(self.figure_id, self.selector_vals, scale=scale)
        fig_widget.size = self.size
        self.set_content(fig_widget)

    def populate_info_panel(self, panel_widget):
        panel_widget.clear_widgets()
        info = [('Figure ID', str(self.figure_id)),
                ('Title', self.caption),
                ('Description', "<TODO: add figure description>")]
        for k, v in info:
            row = BoxLayout(orientation='horizontal', size_hint_y=None)
            row.add_widget(FixedHeightLabel(text=k, size_hint_x=0.4))
            row.add_widget(WrappedLabel(text=v, size_hint_x=0.6))
            panel_widget.add_widget(row)
        #panel_widget.add_widget(Label(text=self.caption, bold=True, size_hint_y=None, height=40))


class FigureArgumentSelector(BoxLayout):
    def create_default_args_selectors():
        pass


#Merge into figure container: root_widget, info_sidebar, status_label
#class FigureCapsule(object):
#
#    def __init__(self, creation_fn, args_template, root_widget, caption='', info_sidebar=None, status_label=None):
#        self.creation_fn = creation_fn
#        self.args_template = args_template
#        self.selector_vals = {}
#        self.args = []
#        self.caption = caption
#        self.fig_container = None
#        self._info_sidebar = info_sidebar
#        self._status_label = status_label
#        self.root_widget = root_widget
#
#        #Initialize self.selector_vals from arguments and root_widget
#        my_properties = [name for name in self.args_template if (isinstance(name, str) and name.startswith('*'))]
#        selector_types = self.root_widget.selector_types_for_properties(my_properties)
#        for typ in selector_types:
#            self.root_widget.add_figure_property_selector(typ, None, storage_dict=self.selector_vals)
#
#    def fill_args_from_creation_arg_dict(self, arg_dict):
#        self.args = []
#        for t in self.args_template:
#            if isinstance(t, str) and t.startswith('*'):
#                self.args.append(arg_dict[t])
#            else:
#                self.args.append(t)
#
#    def update_figure_widget(self, data_area, scale=1.0):
#        from pygsti.report.workspace import WorkspaceTable, WorkspacePlot
#
#        workspace_obj = self.creation_fn(*self.args)
#        if isinstance(workspace_obj, WorkspaceTable):
#            tbl = workspace_obj.tables[0]
#            out = tbl.render('kivywidget', kivywidget_kwargs={'size_hint': (None, None)})
#            figwidget = out['kivywidget']
#            print("DB: TABLE Initial size = ", figwidget.size, ' pos=', figwidget.pos)
#
#        elif isinstance(workspace_obj, WorkspacePlot):
#            plt = workspace_obj.figs[0]
#            constructor_fn, kwargs = plt.kivywidget
#            natural_size = plt.metadata.get('natural_size', (300, 300))
#            kwargs.update({'size_hint': (None, None)})
#            figwidget = constructor_fn(**kwargs)
#            figwidget.size = natural_size
#            print("DB: PLOT Initial size = ", natural_size, ' pos=', figwidget.pos)
#        else:
#            raise ValueError("Invalid figure type created: " + str(type(workspace_obj)))
#
#        if self.fig_container is None:
#            # Automatically scale size to fit visible window (OPTIONAL SETTING LATER?)
#            #max_scale = min(data_area.width / figwidget.width, data_area.height / figwidget.height)
#            #scale = min(scale, max_scale)  # scale figure down so it fits within data_area
#            #if scale != 1.0:
#            #    figwidget.size = (int(figwidget.size[0] * scale), int(figwidget.size[1] * scale))
#
#            fig = FigureContainer(figwidget, self.caption, self, size_hint=(None, None))  # Note: capsule and container should probably be one and the same...
#            set_info_containers(fig, self._info_sidebar, self._status_label)
#            data_area.add_widget(fig)
#            self.fig_container = fig
#        else:
#            figwidget.size = self.fig_container.size
#            self.fig_container.set_content(figwidget)
#
#        return figwidget.size  # for use in later processing to set figure positiong
#
#    def populate_figure_property_panel(self, panel_widget):
#        panel_widget.clear_widgets()
#
#        panel_widget.add_widget(Label(text=self.caption, bold=True, size_hint_y=None, height=40))
#        my_properties = [name for name in self.args_template if (isinstance(name, str) and name.startswith('*'))]
#        selector_types = self.root_widget.selector_types_for_properties(my_properties)
#        for typ in selector_types:
#            self.root_widget.add_figure_property_selector(typ, panel_widget, storage_dict=self.selector_vals)
#        btn = Button(text='Update', size_hint_y=None, height=40)
#        btn.bind(on_release=self.update_figure)
#        panel_widget.add_widget(btn)
#
#    def update_figure(self, *args):
#        fig_creation_args = self.root_widget.selector_values_to_creation_args(self.selector_vals)
#        self.fill_args_from_creation_arg_dict(fig_creation_args)
#        self.update_figure_widget(None)
#
#    def to_json_dict(self):
#        cls_to_build = self.creation_fn.__globals__['cls']  # some magic to get the underlying class being constructed
#        to_save = {'creation_cls': cls_to_build.__module__ + '.' + cls_to_build.__name__,
#                   'caption': self.caption,
#                   'args_template': self.args_template,
#                   'arg_selector_values': self.selector_vals,
#                   'size': self.fig_container.size,
#                   'position': self.fig_container.pos}
#        return to_save


class CustomAccordionTitle(Label):
    item = ObjectProperty(None, allownone=True)


class CustomAccordionItem(AccordionItem):
    #Overrides _update_title so we don't have to use (deprecated) templates to customize them
    # Basically copied from accordion.py
    def _update_title(self, dt):
        if not self.container_title:
            self._trigger_title()
            return
        c = self.container_title
        c.clear_widgets()
        instance = CustomAccordionTitle(item=self, text=self.title, bold=True, font_size=24)
        c.add_widget(instance)


#class CustomAccordionTitle(Label):
#    """ Mimics the (deprecated) default Kivy template for an accordion title"""
#    def __init__(self, text, item, **kwargs):
#        from kivy.graphics import PushMatrix, PopMatrix, Translate, Rotate, BorderImage
#        super().__init__(text=text, **kwargs)
#
#        with self.canvas.before:
#            Color(1, 1, 1, 1)
#            self.bi = BorderImage(source=item.background_normal if item.collapse else item.background_selected,
#                                  pos=self.pos, size=self.size)
#            PushMatrix()
#            self.t1 = Translate(xy=(self.center_x, self.center_y))
#            Rotate(angle= 90 if item.orientation == 'horizontal' else 0, axis=(0, 0, 1))
#            self.t2 = Translate(xy=(-self.center_x, -self.center_y))
#
#        with self.canvas.after:
#            PopMatrix
#
#        self.bind(pos=self.update, size=self.update)
#        item.bind(collapse=lambda inst, v: setattr(self.bi, 'source', inst.background_normal
#                                                   if v else inst.background_selected))
#
#    def update(self, *args):
#        self.bi.pos = self.pos
#        self.bi.size = self.size
#        self.t1.xy = (self.center_x, self.center_y)
#        self.t2.xy = (-self.center_x, -self.center_y)


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


class LoadRootDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    initial_path = StringProperty('')


class OpenDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    initial_path = StringProperty('')
    filters = ListProperty([])


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    filename = ObjectProperty(None)
    cancel = ObjectProperty(None)
    initial_path = StringProperty('')


class EditTabDialog(FloatLayout):
    ok = ObjectProperty(None)
    cancel = ObjectProperty(None)
    reorder = ObjectProperty(None)
    tab_name = StringProperty('')


class DataExplorerApp(App):
    def __init__(self, app_path, initial_kernel_json_name):  #, test_widget):
        self.app_path = app_path
        self.initial_kernel_json_name = initial_kernel_json_name
        #self.test_widget = test_widget
        super().__init__()

        cache_file = _os.path.join(app_path, '.pygsti_latex_widget_cache')
        if _os.path.exists(cache_file):
            print("Reading latex widget cache from ", cache_file)
            LatexWidget.read_cache(cache_file)
        else:
            LatexWidget.svg_cache = {}  # create an empty cache

    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        return RootExplorerWidget(self.app_path, self.initial_kernel_json_name)

    def on_request_close(self, *args):
        print("Writing latex widget cache to speedup latex -> svg conversion in the future.")
        LatexWidget.write_cache(_os.path.join(self.app_path, '.pygsti_latex_widget_cache'))


#if __name__ == '__main__':
#    DataExplorerApp(tblwidget).run()
