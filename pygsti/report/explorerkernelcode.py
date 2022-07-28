import numpy as np
import json
import os
import inspect as _inspect

import pygsti
from pygsti.report.workspace import WorkspaceTable, WorkspacePlot
from pygsti.protocols.protocol import CircuitListsDesign as _CircuitListsDesign
from pygsti.protocols.gst import StandardGSTDesign as _StandardGSTDesign
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.protocols.estimate import Estimate as _Estimate


class EdesignLibElement(object):

    def __init__(self, name, edesign, root_file_path, path_from_root):
        self.name = name
        self.edesign = edesign
        self.root_file_path = root_file_path
        self.path_from_root = path_from_root  # a list, e.g. ['CombinedDesign1', 'Q0']

    @classmethod
    def from_json_dict(cls, json_dict, result_dirs_cache=None):
        if result_dirs_cache is None: result_dirs_cache = {}
        if json_dict['root_path'] not in result_dirs_cache:
            result_dirs_cache[json_dict['root_path']] = pygsti.io.read_results_from_dir(json_dict['root_path'])
        results_dir = result_dirs_cache[json_dict['root_path']]
        for key in json_dict['path_from_root']:
            results_dir = results_dir[key]
        protocol_data = results_dir.data
        return cls(json_dict['name'], protocol_data.edesign, json_dict['root_path'], json_dict['path_from_root'])

    def to_json_dict(self):
        return {'name': self.name,
                'root_path': self.root_file_path,
                'path_from_root': self.path_from_root}

    @property
    def info(self):
        return self.to_json_dict()


class DatasetLibElement(object):

    def __init__(self, name, dataset, root_file_path, path_from_root):
        self.name = name
        self.dataset = dataset
        self.root_file_path = root_file_path
        self.path_from_root = path_from_root  # a list, e.g. ['CombinedDesign1', 'Q0']

    @classmethod
    def from_json_dict(cls, json_dict, result_dirs_cache=None):
        if result_dirs_cache is None: result_dirs_cache = {}
        if json_dict['root_path'] not in result_dirs_cache:
            result_dirs_cache[json_dict['root_path']] = pygsti.io.read_results_from_dir(json_dict['root_path'])
        results_dir = result_dirs_cache[json_dict['root_path']]
        for key in json_dict['path_from_root']:
            results_dir = results_dir[key]
        protocol_data = results_dir.data
        return cls(json_dict['name'], protocol_data.dataset, json_dict['root_path'], json_dict['path_from_root'])

    def to_json_dict(self):
        return {'name': self.name,
                'root_path': self.root_file_path,
                'path_from_root': self.path_from_root}

    @property
    def info(self):
        return self.to_json_dict()


class ModelLibElement(object):
    def __init__(self, name, model_name, model, model_container, root_file_path, path_from_root,
                 protocol_name, additional_details=None):
        self.name = name
        self.model = model
        self.model_name = model_name
        self.model_container = model_container
        self.root_file_path = root_file_path
        self.path_from_root = path_from_root  # a list, e.g. ['CombinedDesign1', 'Q0']
        self.protocol_name = protocol_name
        self.additional_details = additional_details if (additional_details is not None) else {}

    @classmethod
    def from_json_dict(cls, json_dict, result_dirs_cache=None):
        if result_dirs_cache is None: result_dirs_cache = {}
        if json_dict['root_path'] not in result_dirs_cache:
            result_dirs_cache[json_dict['root_path']] = pygsti.io.read_results_from_dir(json_dict['root_path'])
        results_dir = result_dirs_cache[json_dict['root_path']]
        for key in json_dict['path_from_root']:
            results_dir = results_dir[key]
        results_obj = results_dir.for_protocol[json_dict['protocol_name']]
        if 'estimate_name' in json_dict['additional_details']:
            model_container = results_obj.estimates[json_dict['additional_details']['estimate_name']]
        else:
            model_container = results_obj
        model = model_container.models[json_dict['model_name']]

        return cls(json_dict['name'], json_dict['model_name'], model, model_container,
                   json_dict['root_path'], json_dict['path_from_root'], json_dict['protocol_name'],
                   json_dict['additional_details'])

    def to_json_dict(self):
        return {'name': self.name,
                'model_name': self.model_name,
                'root_path': self.root_file_path,
                'path_from_root': self.path_from_root,
                'protocol_name': self.protocol_name,
                'additional_details': self.additional_details}

    @property
    def info(self):
        return self.to_json_dict()



class ExplorerLibrary(object):
    def __init__(self):
        self.edesign_library = {}
        self.dataset_library = {}
        self.model_library = {}
        self.path = None

    def import_edesign(self, root_name, root_file_path, jsond_path_from_root,
                       boolstr_import_children='False', boolstr_import_all_models='False'):
        path_from_root = json.loads(jsond_path_from_root)
        import_children = bool(boolstr_import_children == 'True')
        import_all_models = bool(boolstr_import_all_models == 'True')

        pth_for_name = [root_name] + path_from_root if root_name else path_from_root
        name = '.'.join(pth_for_name)  # edesign name
        results_dir = pygsti.io.read_results_from_dir(root_file_path)
        for ky in path_from_root:
            results_dir = results_dir[ky]

        #btn = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
        #btn.bind(state=self.update_library_item_info)
        if name not in self.edesign_library:

            data = results_dir.data  # a ProtocolData object
            self.edesign_library[name] = EdesignLibElement(name, data.edesign, root_file_path, path_from_root) #, btn)
            #self.ids.edesign_library_list.add_widget(btn)
            print("Imported edesign: ", name)

            if data.dataset is not None:
                #btn2 = ToggleButton(text=name, size_hint_y=None, height=40, group='libraryitem')
                #btn2.bind(state=self.update_library_item_info)
                self.dataset_library[name] = DatasetLibElement(name, data.dataset, root_file_path, path_from_root) #, btn2)
                #self.ids.dataset_library_list.add_widget(btn2)
                print("Imported dataset: ", name)

            if import_all_models:
                self.import_models(root_name, root_file_path, jsond_path_from_root, 'all', 'all', 'all')
        else:
            print("An e-design with name '%s' is already imported, and will not be clobbered." % name); return False

        if import_children:
            for ky in results_dir.keys():
                child_path_from_root = path_from_root + [ky]
                child_results_dir = results_dir[ky]
                self.import_edesign(root_name, root_file_path, child_path_from_root,
                                    child_results_dir, import_children, import_all_models)
        return True

    def import_models(self, root_name, root_file_path, jsond_path_from_root, jsond_protocol_names,
                      jsond_additional_details, jsond_model_names):
        path_from_root = json.loads(jsond_path_from_root)
        protocol_names = json.loads(jsond_protocol_names)
        additional_details = json.loads(jsond_additional_details)
        model_names = json.loads(jsond_model_names)

        results_dir = pygsti.io.read_results_from_dir(root_file_path)
        for ky in path_from_root:
            results_dir = results_dir[ky]

        if protocol_names == 'all':
            protocol_names = list(results_dir.for_protocol.keys())

        from pygsti.protocols.gst import ModelEstimateResults as _ModelEstimateResults
        for protocol_name in protocol_names:
            results = results_dir.for_protocol[protocol_name]

            if additional_details == 'all':
                if isinstance(results, _ModelEstimateResults):
                    additional_details = [{'estimate_name': est_name} for est_name in results.estimates.keys()]
                else:
                    additional_details = [{}]

            for additional_detail in additional_details:
                if 'estimate_name' in additional_detail:
                    estimate = results.estimates[additional_detail['estimate_name']]
                    model_container = estimate
                else:
                    model_container = results

                pth_for_name = [root_name] + path_from_root if root_name else path_from_root

                if model_names == 'all':
                    model_names = list(model_container.models.keys())

                for model_name in model_names:
                    name = '.'.join(pth_for_name + [protocol_name] + list(additional_detail.values()) + [model_name])
                    model = model_container.models[model_name]
                    if name in self.model_library:
                        print("A model named '%s' is already imported and will not be clobbered." % name)
                        continue

                    self.model_library[name] = ModelLibElement(name, model_name, model, model_container,
                                                               root_file_path, path_from_root,
                                                               protocol_name, additional_detail) # btn
                    print("Importing model: ", model_name)

    def import_models_from_files(self, jsond_filenames):
        filenames = json.loads(jsond_filenames)

        for filename in filenames:
            print("Importing model file: ", filename)

            model = pygsti.models.Model.read(filename)
            name = os.path.basename(filename)
            model_name = protocol_name = 'N/A'
            model_container = None
            additional_detail = {}
            root_file_path = filename
            path_from_root = []

            if name in self.model_library:
                print("A model named '%s' is already imported and will not be clobbered." % name)
                return False
            self.model_library[name] = ModelLibElement(name, model_name, model, model_container,
                                                       root_file_path, path_from_root,
                                                       protocol_name, additional_detail)
        return True

    def import_datasets_from_files(self, jsond_filenames):
        filenames = json.loads(jsond_filenames)

        for filename in filenames:
            ds = pygsti.io.read_dataset(filename)
            name = os.path.basename(filename)
            root_file_path = filename
            path_from_root = []

            if name in self.edesign_library:
                print("An edesign named '%s' is already imported and will not be clobbered." % name)
                return False

            #edesign = _ExperimentDesign(list(ds.keys()))  # just create a simple experiment design around ds
            #data = _ProtocolData(edesign, ds)
            self.dataset_library[name] = DatasetLibElement(name, ds, root_file_path, path_from_root)
        return True

    def load_from_file(self, filename):
        with open(filename) as f:
            d = json.load(f)

        assert(d['file_type'].startswith('library')), "This doesn't look like a library file!"

        #Clear existing library)
        self.edesign_library.clear()
        self.dataset_library.clear()
        self.model_library.clear()

        result_dirs_cache = {}
        for key, item_dict in d['edesign_library'].items():
            item = EdesignLibElement.from_json_dict(item_dict, result_dirs_cache)
            self.edesign_library[item.name] = item

        for key, item_dict in d['dataset_library'].items():
            item = DatasetLibElement.from_json_dict(item_dict, result_dirs_cache)
            self.dataset_library[item.name] = item

        for key, item_dict in d['model_library'].items():
            item = ModelLibElement.from_json_dict(item_dict, result_dirs_cache)
            self.model_library[item.name] = item

        self.path = filename
        return True

    def save_to_file(self, filename):
        from pygsti import __version__ as _pygsti_version
        to_save = {'pygsti version': _pygsti_version,
                   'creator': 'pyGSTi data explorer',
                   'file_type': 'library.v1',
                   'edesign_library': {},
                   'dataset_library': {},
                   'model_library': {}}
        for key, item in self.edesign_library.items():
            to_save['edesign_library'][key] = item.to_json_dict()
        for key, item in self.dataset_library.items():
            to_save['dataset_library'][key] = item.to_json_dict()
        for key, item in self.model_library.items():
            to_save['model_library'][key] = item.to_json_dict()

        with open(filename, 'w') as f:
            json.dump(to_save, f, indent=4)
        return True

    def _get_library_dict(self, category):
        if category == 'edesign': return self.edesign_library
        elif category == 'dataset': return self.dataset_library
        elif category == 'model': return self.model_library
        else: raise ValueError("Invalid library category: %s" % category)

    def update_item_name(self, category, existing_name, new_name):
        libdict = self._get_library_dict(category)
        libdict[existing_name].name = new_name
        libdict[new_name] = libdict[existing_name]
        del libdict[existing_name]
        return True

    def remove_item(self, category, name):
        libdict = self._get_library_dict(category)
        if name in libdict:
            del libdict[name]
            return True
        else:
            return False


class FigureCapsule(object):

    def __init__(self, creation_fn, args_template, figure_id, caption='', tab_name=''):
        self.workspace_obj = None
        self.creation_fn = creation_fn
        self.args_template = args_template
        self.selector_values = {}
        self.args = []
        self.caption = caption
        self.tab_name = tab_name
        self.figure_id = figure_id
        self.property_names = [name for name in self.args_template if (isinstance(name, str) and name.startswith('*'))]
        #self.fig_container = None
        #self._info_sidebar = info_sidebar
        #self._status_label = status_label

        ##Initialize self.selector_vals from arguments and root_widget
        #my_properties = [name for name in self.args_template if (isinstance(name, str) and name.startswith('*'))]
        #selector_types = self.root_widget.selector_types_for_properties(my_properties)
        #for typ in selector_types:
        #    self.root_widget.add_figure_property_selector(typ, None, storage_dict=self.selector_vals)

    def fill_args_from_creation_arg_dict(self, arg_dict):
        self.args = []
        for t in self.args_template:
            if isinstance(t, str) and t.startswith('*'):
                self.args.append(arg_dict[t])
            else:
                self.args.append(t)

    def update_workspace_obj(self, selector_values, arg_dict):
        self.selector_values = selector_values  # just for storage (?)
        self.fill_args_from_creation_arg_dict(arg_dict)
        self.workspace_obj = self.creation_fn(*self.args)

    #def update_figure_widget(self, data_area, scale=1.0):
    def create_figure_widget_json(self):
        if self.workspace_obj is None:
            return ''

        elif isinstance(self.workspace_obj, WorkspaceTable):
            tbl = self.workspace_obj.tables[0]
            figwidget_factory = tbl.render('kivywidget')['kivywidget']

        elif isinstance(self.workspace_obj, WorkspacePlot):
            plt = self.workspace_obj.figs[0]
            figwidget_factory = plt.kivywidget_factory

            #TODO REMOVE
            ##HERE -- json and send constructor_fn and kwargs to front end, then run below code there?
            #natural_size = plt.metadata.get('natural_size', (300, 300))
            #kwargs.update({'size_hint': (None, None)})
            #figwidget = constructor_fn(**kwargs)
            #figwidget.size = natural_size
            #print("DB: PLOT Initial size = ", natural_size, ' pos=', figwidget.pos)
        else:
            raise ValueError("Invalid workspace object type: " + str(type(self.workspace_obj)))

        #TODO REMOVE but move find_int64_in_json_dict to a tools module? 
        #import pprint
        #import numpy
        #pp = pprint.PrettyPrinter(indent=2)
        #s = figwidget_factory.to_nice_serialization()
        #pp.pprint(s)
        #def find_in_json_dict(d, bc=None, typ=numpy.int64):
        #    if bc is None: bc = []
        #    if isinstance(d, numpy.int64): print("Found ", typ, " at ", bc)
        #    elif isinstance(d, dict):
        #        for k, v in d.items():
        #            find_in_json_dict(v, bc + [k], typ)
        #    elif isinstance(d, (list, tuple)):
        #        for i, v in enumerate(d):
        #            find_in_json_dict(v, bc + [i], typ)
        #find_in_json_dict(s, None, numpy.ndarray)
        #import bpdb; bpdb.set_trace()
        return json.dumps(figwidget_factory.to_nice_serialization())  # send widget factory json to front end


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

    def to_json_dict(self):
        cls_to_build = self.creation_fn.__globals__['cls']  # some magic to get the underlying class being constructed
        to_save = {'creation_cls': cls_to_build.__module__ + '.' + cls_to_build.__name__,
                   'figure_id': self.figure_id,
                   'caption': self.caption,
                   'tab_name': self.tab_name,
                   'args_template': self.args_template,
                   'arg_selector_values': self.selector_values}
        return to_save

    @classmethod
    def from_json_dict(cls, json_dict, workspace):
        creation_cls = pygsti.io.metadir._class_for_name(json_dict['creation_cls'])

        # Python magic that dynamically creates a factory function just like a Workspace object does.
        # The result is a function like self.ws.SomeTable(...) that implicitly gets it's 'ws' arg set.
        argspec = _inspect.getargspec(creation_cls.__init__)
        argnames = argspec[0]
        factoryfn_argnames = argnames[2:]  # strip off self & ws args
        factoryfn_argspec = (factoryfn_argnames,) + argspec[1:]
        #signature = _inspect.formatargspec(formatvalue=lambda val: "", *factoryfn_argspec)  # removes defaults
        signature = _inspect.formatargspec(*factoryfn_argspec)
        signature = signature[1:-1]  # strip off parenthesis from ends of "(signature)"
        factoryfn_def = (
            'def factoryfn(%(signature)s):\n'
            '    return cls(self, %(signature)s)' %
            {'signature': signature})
        exec_globals = {'cls': creation_cls, 'self': workspace}
        exec(factoryfn_def, exec_globals)
        factoryfn = exec_globals['factoryfn']

        return cls(factoryfn, json_dict['args_template'], json_dict['figure_id'],
                   json_dict['caption'], json_dict['tab_name'])


class PyGSTiAnalysis(object):
    def __init__(self, library):
        self.lib = library
        self.capsules = {}
        self.ws = pygsti.report.Workspace(gui_mode='kivy')
        self.path = None

    def selector_values_to_creation_args(self, figure_selector_vals):  ## REVAMP - creation args should just be *names* of kernel variables
        creation_args = {}
        #all_properties = ['*models', '*model_titles', '*model', '*model_title', '*target_model',
        #                  '*dataset', '*edesign', '*circuit_list', '*maxlengths', '*circuits_by_maxl',
        #                  '*objfn_builder', '*gaugeopt_args', '*estimate_params']

        for typ, val in figure_selector_vals.items():
            if val == '(none)':
                continue  # don't populate any creation args

            if typ == '**model_title':
                if val:
                    creation_args['*model_title'] = val
            elif typ == '**model':
                creation_args['*model'] = self.lib.model_library[val].model
                creation_args['*model_dim'] = self.lib.model_library[val].model.dim
                if '*model_title' not in creation_args:
                    creation_args['*model_title'] = self.lib.model_library[val].model_name
                if isinstance(self.lib.model_library[val].model_container, _Estimate):
                    estimate = self.lib.model_library[val].model_container
                    creation_args['*estimate_params'] = estimate.parameters
                    creation_args['*unmodeled_error'] = estimate.parameters.get("unmodeled_error", None)
                    creation_args['*gaugeopt_args'] = estimate.goparameters.get(self.lib.model_library[val].model_name, {})
                    if isinstance(self.lib.model_library[val].model_container.parent.data.edesign, _StandardGSTDesign):
                        max_length_list = self.lib.model_library[val].model_container.parent.data.edesign.maxlengths
                        creation_args['*models_by_maxl'] = [estimate.models['iteration %d estimate' % i]
                                                            for i in range(len(max_length_list))]
            elif typ == '**target_model':
                creation_args['*target_model'] = self.lib.model_library[val].model
            elif typ == '**edesign':
                edesign = self.lib.edesign_library[val].edesign
                creation_args['*edesign'] = edesign
                creation_args['*circuit_list'] = edesign.all_circuits_needing_data
                creation_args['*circuit_lists'] = edesign.circuit_lists \
                    if isinstance(edesign, _CircuitListsDesign) else None
                if isinstance(edesign, _StandardGSTDesign):
                    creation_args['*maxlengths'] = edesign.maxlengths
                    creation_args['*circuits_by_maxl'] = edesign.circuit_lists
            elif typ == '**dataset':
                dataset = self.lib.dataset_library[val].dataset
                creation_args['*dataset'] = dataset
            elif typ == '**objfn_builder':
                if val == 'from estimate':
                    if '**model' in figure_selector_vals:
                        k = figure_selector_vals['**model']
                        mdl_container = self.lib.model_library[k].model_container
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

    def add_figure(self, tab_name, figure_type, name):  # returns 'id' of item
        cri = None
        figure_id = np.random.randint(10000)  # TODO - make unique ID generation more robust!! -- could have collisions here!
        extra_capsule_args = dict(caption=name,
                                  figure_id=figure_id,
                                  tab_name=tab_name)

        if figure_type == 'SpamTable':
            #wstable = ws.SpamTable(models, titles, 'boxes', cri, False)  # titles?
            figure_capsule = FigureCapsule(self.ws.SpamTable, ['*models', '*model_titles', 'boxes', cri, False],
                                           **extra_capsule_args)
        elif figure_type == 'SpamParametersTable':
            #wstable = ws.SpamParametersTable(models, titles, cri)
            figure_capsule = FigureCapsule(self.ws.SpamParametersTable, ['*models', '*model_titles', cri],
                                           **extra_capsule_args)
        elif figure_type == 'GatesTable':
            #wstable = ws.GatesTable(models, titles, 'boxes', cri)
            figure_capsule = FigureCapsule(self.ws.GatesTable, ['*models', '*model_titles', 'boxes', cri],
                                           **extra_capsule_args)
        elif figure_type == 'ChoiTable':
            #wstable = ws.ChoiTable(models, titles, cri)
            figure_capsule = FigureCapsule(self.ws.ChoiTable, ['*models', '*model_titles', cri],
                                           **extra_capsule_args)
        elif figure_type == 'ModelVsTargetTable':
            clifford_compilation = None
            #wstable = ws.ModelVsTargetTable(model, target_model, clifford_compilation, cri)
            figure_capsule = FigureCapsule(self.ws.ModelVsTargetTable, ['*model', '*target_model',
                                                                        clifford_compilation, cri],
                                           **extra_capsule_args)
        elif figure_type == 'GatesVsTargetTable':
            #wstable = ws.GatesVsTargetTable(model, target_model, cri)  # wildcard?
            figure_capsule = FigureCapsule(self.ws.GatesVsTargetTable, ['*model', '*target_model', cri],
                                           **extra_capsule_args)
        elif figure_type == 'SpamVsTargetTable':
            #wstable = ws.SpamVsTargetTable(model, target_model, cri)
            figure_capsule = FigureCapsule(self.ws.SpamVsTargetTable, ['*model', '*target_model', cri],
                                           **extra_capsule_args)
        elif figure_type == 'ErrgenTable':
            #wstable = ws.ErrgenTable(model, target_model, cri)  # (more options)
            figure_capsule = FigureCapsule(self.ws.ErrgenTable, ['*model', '*target_model', cri],
                                           **extra_capsule_args)
        elif figure_type == 'NQubitErrgenTable':
            #wstable = ws.NQubitErrgenTable(model, cri)
            figure_capsule = None  # FigureCapsule(self.ws.NQubitErrgenTable, ['*model', cri],
                                   #          **extra_capsule_args)
        elif figure_type == 'GateDecompTable':
            #wstable = ws.GateDecompTable(model, target_model, cri)
            figure_capsule = FigureCapsule(self.ws.GateDecompTable, ['*model', '*target_model', cri],
                                           **extra_capsule_args)
        elif figure_type == 'GateEigenvalueTable':
            #wstable = ws.GateEigenvalueTable(model, target_model, cri,
            #                                 display=('evals', 'rel', 'log-evals', 'log-rel'))
            figure_capsule = FigureCapsule(self.ws.GateEigenvalueTable, ['*model', '*target_model', cri,
                                                                         ('evals', 'rel', 'log-evals', 'log-rel')],
                                           **extra_capsule_args)
        elif figure_type == 'DataSetOverviewTable':
            #wstable = ws.DataSetOverviewTable(dataset, max_length_list)
            figure_capsule = FigureCapsule(self.ws.DataSetOverviewTable, ['*dataset', '*maxlengths'],
                                           **extra_capsule_args)
        elif figure_type == 'SoftwareEnvTable':
            #wstable = ws.SoftwareEnvTable()
            figure_capsule = FigureCapsule(self.ws.SoftwareEnvTable, [], **extra_capsule_args)
        elif figure_type == 'CircuitTable':
            # wstable = ws.CircuitTable(...)  # wait until we can select circuit list; e.g. germs, fiducials
            print("Wait until better selection methods to create circuit tables...")
            figure_capsule = None
        elif figure_type == 'GatesSingleMetricTable':
            #metric = 'inf'  # entanglement infidelity
            #wstable = GatesSingleMetricTable(metric, ...)
            print("Wait until better selection methods to create single-item gate metric tables...")
            figure_capsule = None
        elif figure_type == 'StandardErrgenTable':
            #wstable = ws.StandardErrgenTable(model.dim, 'hamiltonian', 'pp')  # not super useful; what about 'stochastic'?
            figure_capsule = FigureCapsule(self.ws.StandardErrgenTable, ['*model_dim', 'H', 'pp'],
                                           **extra_capsule_args)
        elif figure_type == 'GaugeOptParamsTable':
            #wstable = ws.GaugeOptParamsTable(gaugeopt_args)
            figure_capsule = FigureCapsule(self.ws.GaugeOptParamsTable, ['*gaugeopt_args'],
                                           **extra_capsule_args)
        elif figure_type == 'MetadataTable':
            #wstable = ws.MetadataTable(model, estimate_params)
            figure_capsule = FigureCapsule(self.ws.MetadataTable, ['*model', '*estimate_params'],
                                           **extra_capsule_args)
        elif figure_type == 'WildcardBudgetTable':
            #wstable = ws.WildcardBudgetTable(estimate_params.get("unmodeled_error", None))
            figure_capsule = FigureCapsule(self.ws.WildcardBudgetTable, ['*unmodeled_error'],
                                           **extra_capsule_args)
        elif figure_type == 'FitComparisonTable':
            #wstable = ws.FitComparisonTable(max_length_list, circuits_by_L, models_by_L, dataset)
            figure_capsule = FigureCapsule(self.ws.FitComparisonTable, ['*maxlengths', '*circuits_by_maxl',
                                                                        '*models_by_maxl', '*dataset',
                                                                        '*objfn_builder'],
                                           **extra_capsule_args)
        elif figure_type == 'FitComparisonBarPlot':
            #wsplot = ws.FitComparisonBarPlot(max_length_list, circuits_by_L, models_by_L, dataset)
            figure_capsule = FigureCapsule(self.ws.FitComparisonBarPlot, ['*maxlengths', '*circuits_by_maxl',
                                                                          '*models_by_maxl', '*dataset'],
                                           **extra_capsule_args)
        elif figure_type == 'FitComparisonBarPlotB':
            #wsplot = ws.FitComparisonBarPlot(est_lbls_mt, [circuit_list] * len(est_mdls_mt),
            #                                 est_mdls_mt, [dataset] * len(est_mdls_mt), objfn_builder)
            def multiplx(titles, circuit_list, models, dataset, objfn_builder):
                return self.ws.FitComparisonBarPlot(titles, [circuit_list] * len(titles),
                                                    models, [dataset] * len(titles), objfn_builder)
            figure_capsule = FigureCapsule(multiplx, ['*model_titles', '*circuit_list', '*models', '*dataset',
                                                      '*objfn_builder'], **extra_capsule_args)
        elif figure_type == 'FitComparisonBoxPlot':
            # used for multiple data sets -- enable this once we get better selection methods
            print("Wait until better selection methods to create fit comparison box plot...")
            figure_capsule = None
        elif figure_type in ('ColorBoxPlot', 'ColorScatterPlot', 'ColorHistogramPlot'):

            if figure_type == 'ColorBoxPlot': plot_type = "boxes"
            elif figure_type == "ColorScatterPlot": plot_type = "scatter"
            else: plot_type = "histogram"

            linlog_percentile = 5
            #bgcolor = 'white'
            #wsplot = ws.ColorBoxPlot(objfn_builder, circuit_list, dataset, model,
            #    linlg_pcntle=linlog_percentile / 100, comm=None, bgcolor=bgcolor, typ=plot_type)
            figure_capsule = FigureCapsule(self.ws.ColorBoxPlot,
                                           ['*objfn_builder', '*circuit_list', '*dataset', '*model',
                                            False, False, True, False, 'compact', linlog_percentile / 100,
                                            None, None, None, None, None, plot_type],
                                           **extra_capsule_args)
        else:
            figure_capsule = None  # indicates figure could not be created

        #if tab_name not in self.items_by_tab:
        #    self.ids_by_tab[tab_name] = set()
        #self.ids_by_tab[tab_name].add(figure_id)
        if figure_capsule is not None:
            self.capsules[figure_id] = figure_capsule
            return figure_id
        else:
            return None

    def update_figure(self, figure_id, jsond_selector_values):
        selector_values = json.loads(jsond_selector_values)
        figure_creation_args = self.selector_values_to_creation_args(selector_values)
        self.capsules[figure_id].update_workspace_obj(selector_values, figure_creation_args)

    def remove_figure(self, figure_id):
        if figure_id in self.capsules:
            del self.capsules[figure_id]
            return True
        else:
            return False

    def load_from_file(self, filename):
        with open(filename) as f:
            d = json.load(f)

        if d['file_type'].startswith('library'):
            print("Actually, this is a library file -- opening it as such:")
            self.lib.load_from_file(filename)
            return {}  # no front end info from loading a library

        assert(d['file_type'].startswith('analysis')), "This doesn't look like a library file!"
        front_end_info = {'tabs': {tab_name: [] for tab_name in d['tab_names']},
                          'loaded_library_path': None}

        if (d['library_relative_path'] and len(self.lib.edesign_library) == 0
           and len(self.lib.dataset_library) == 0 and len(self.lib.model_library) == 0):
            library_path = os.path.abspath(os.path.join(os.path.dirname(filename), d['library_relative_path']))
            #print("Opening library from relative path: ", library_path)
            self.lib.load_from_file(library_path)
            front_end_info['loaded_library_path'] = library_path

        for figure_dict in d['figures']:
            #print("Building ", figure_dict['caption'])
            capsule = FigureCapsule.from_json_dict(figure_dict, self.ws)
            selector_vals = figure_dict['arg_selector_values']
            self.capsules[capsule.figure_id] = capsule

            #These are done by front end, when it creates its widgets based on
            # the selector_vals we send, but could do it redundantly here:
            #figure_creation_args = self.selector_values_to_creation_args(selector_vals)
            #capsule.update_workspace_obj(selector_vals, figure_creation_args)

            fig_frontend_info = {'figure_id': capsule.figure_id, 'caption': capsule.caption,
                                 'selector_values': selector_vals,
                                 'property_names': capsule.property_names}
            for k in ('size', 'position', 'comment'):
                if k in figure_dict: fig_frontend_info[k] = figure_dict[k]
            front_end_info['tabs'][capsule.tab_name].append(fig_frontend_info)

        self.path = filename
        return front_end_info

    def save_to_file(self, filename, jsond_frontend_figure_info=None):
        frontend_figure_info = {k: v for k, v in json.loads(jsond_frontend_figure_info)}

        from pygsti import __version__ as _pygsti_version
        to_save = {'pygsti version': _pygsti_version, 'creator': 'pyGSTi data explorer',
                   'file_type': 'analysis.v1',
                   'library_relative_path': (os.path.relpath(self.lib.path, os.path.dirname(filename))
                                             if (self.lib.path is not None) else None),
                   'tab_names': [],
                   'figures': []
                   }

        for fig_id, capsule in self.capsules.items():
            assert(fig_id == capsule.figure_id)
            capsule_info = capsule.to_json_dict()
            if frontend_figure_info and fig_id in frontend_figure_info:
                capsule_info.update(frontend_figure_info[fig_id])
            to_save['figures'].append(capsule_info)
            if capsule.tab_name not in to_save['tab_names']:
                to_save['tab_names'].append(capsule.tab_name)

        with open(filename, 'w') as f:
            json.dump(to_save, f, indent=4)
        self.path = filename

    
#Global variables
#ws = Workspace(gui_mode='kivy')
lib = ExplorerLibrary()
analysis = PyGSTiAnalysis(lib)
