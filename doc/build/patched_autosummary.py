"""
 This file contains a *slightly* (see single "OLD" changed line) version of
 the `generate_autosummary_docs` function found in 
 .../site_packages/sphinx/ext/autosummary/generate.py, which determines how 
 to process the .. autosummary:: directives in pyGSTi's documentation-
 generating .rst files.  

 In pyGSTi, sub-package structure is used to further
 organize the code, so that while routines are actually contained in modules
 beneath sub-packages beneath the root "pygsti" package, we want to expose the
 routines as being members of just one of the sub-packages.  

 For example, the "do_long_sequence_gst" routine is contained in the
 longsequence.py module beneath the "drivers" sub-package.  Instead of requiring
 users to call this routine as pygsti.drivers.longsequence.do_long_sequence_gst
 we import it into pygsti.drivers (and pygsti for that matter) so that it can be
 called as pygsti.drivers.do_long_sequence_gst or just pygsti.do_long_sequence_gst.

 Not comes the issue with Sphinx's autosummary extension.  It performs a check
 (meant to refrain from documenting externally-imported stuff) that tests whether
 a function (or class, or whatever) object's ".__module__" strictly equals the
 ".__name__" of the module/package being documented.  This means that the
 pygsti.drivers.longsequence.do_long_sequence_gst function (whose __module__
 is pygsti.drivers.longsequence) would NOT be documented within the pygsti.drivers
 sub-package - but we want it to be!

 The hack solution implemented here is to monkey patch the 
 sphinx.ext.autosummary.generate.generate_autosummary_docs function to test
 whether the object's __module__ simple *starts-with* the name of the module
 or package being documented.
"""

import sys, os
from sphinx.ext.autosummary.generate import *
from sphinx.ext.autosummary.generate import _simple_warn, _simple_info, _underline
from sphinx.util import logger, display_chunk
from sphinx.util.console import bold

#Lazy "import *" above: could list all the things we need, like:
#package_dir
#BuiltinTemplateLoader()
#FileSystemLoader(template_dirs)  # type: ignore
#SandboxedEnvironment(loader=template_loader)
#_underline
#rst_escape
#find_autosummary_in_files(sources)


def generate_autosummary_docs_patch(sources, output_dir=None, suffix='.rst',
                                    warn=_simple_warn, info=_simple_info,
                                    base_path=None, builder=None, template_dir=None,
                                    imported_members=False, app=None):
    # type: (List[unicode], unicode, unicode, Callable, Callable, unicode, Builder, unicode, bool, Any) -> None  # NOQA
    showed_sources = list(sorted(sources))
    if len(showed_sources) > 20:
        showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
    info('[autosummary] generating autosummary for: %s' %
         ', '.join(showed_sources))

    if output_dir:
        info('[autosummary] writing to %s' % output_dir)

    if base_path is not None:
        sources = [os.path.join(base_path, filename) for filename in sources]

    # create our own templating environment
    template_dirs = None  # type: List[unicode]
    template_dirs = [os.path.join(package_dir, 'ext',
                                  'autosummary', 'templates')]

    template_loader = None  # type: BaseLoader
    if builder is not None:
        # allow the user to override the templates
        template_loader = BuiltinTemplateLoader()
        template_loader.init(builder, dirs=template_dirs)
    else:
        if template_dir:
            template_dirs.insert(0, template_dir)
        template_loader = FileSystemLoader(template_dirs)  # type: ignore
    template_env = SandboxedEnvironment(loader=template_loader)
    template_env.filters['underline'] = _underline

    # replace the builtin html filters
    template_env.filters['escape'] = rst_escape
    template_env.filters['e'] = rst_escape

    # read
    items = find_autosummary_in_files(sources)

    # keep track of new files
    new_files = []

    # write
    for name, path, template_name in sorted(set(items), key=str):
        if path is None:
            # The corresponding autosummary:: directive did not have
            # a :toctree: option
            continue

        path = output_dir or os.path.abspath(path)
        ensuredir(path)

        try:
            name, obj, parent, mod_name = import_by_name(name)
        except ImportError as e:
            warn('[autosummary] failed to import %r: %s' % (name, e))
            continue

        fn = os.path.join(path, name + suffix)

        # skip it if it exists
        if os.path.isfile(fn):
            continue

        new_files.append(fn)

        with open(fn, 'w') as f:
            try:
                doc = get_documenter(app, obj, parent) # newer Sphinx versions
            except(TypeError): # when fn takes only 2 args
                doc = get_documenter(obj, parent) # older Sphinx versions

            if template_name is not None:
                template = template_env.get_template(template_name)
            else:
                try:
                    template = template_env.get_template('autosummary/%s.rst'
                                                         % doc.objtype)
                except TemplateNotFound:
                    template = template_env.get_template('autosummary/base.rst')

            def get_members(obj, typ, include_public=[], imported=True):
                # type: (Any, unicode, List[unicode], bool) -> Tuple[List[unicode], List[unicode]]  # NOQA
                items = []  # type: List[unicode]
                dbcount = 0
                for name in dir(obj):
                    try:
                        value = safe_getattr(obj, name)
                    except AttributeError as e:
                        #print("EGN AttrErr: %s.%s: " % (obj.__name__,name),str(e))
                        continue
                    try:
                        documenter = get_documenter(app, value, obj) #newer Sphinx versions
                    except(TypeError): # when fn takes only 2 args
                        documenter = get_documenter(value, obj) # older Sphinx versions
                        
                    #print("EGN %s.%s typ = " % (obj.__name__,name),documenter.objtype, " tgt=",typ)
                    if documenter.objtype == typ:
                        valmod = getattr(value, '__module__', None)
                        valmod_parts = valmod.split(".") if (valmod is not None) else []
                        objname_parts = obj.__name__.split(".")
                        #OLD if imported or getattr(value, '__module__', None) == obj.__name__:
                        #DEBUG if imported or getattr(value, '__module__', None) == obj.__name__ or obj.__name__ == "pygsti":
                        #DEBUG if imported or (getattr(value, '__module__', None).startswith( obj.__name__ ) and dbcount < 100):
                        #if imported or getattr(value, '__module__', None).startswith( obj.__name__ ):
                        if imported or valmod == obj.__name__ or \
                           (len(valmod_parts) > 2 and len(objname_parts) == 2 and valmod_parts[0:2] == objname_parts):
                            # skip imported members if expected
                            items.append(name)
                            #else:
                            #    print("EXTRA: ", name, " valmod=",getattr(value, '__module__', None), " parent=",obj.__name__, " typ=",typ,
                            #          file=sys.stdout)
                            dbcount += 1
                        #else: print("SKIPPED: ",imported, getattr(value, '__module__', None), obj.__name__)
                public = [x for x in items
                          if x in include_public or not x.startswith('_')]
                return public, items

            ns = {}  # type: Dict[unicode, Any]

            if doc.objtype == 'module':
                ns['members'] = dir(obj)
                ns['functions'], ns['all_functions'] = \
                    get_members(obj, 'function', imported=imported_members)
                ns['classes'], ns['all_classes'] = \
                    get_members(obj, 'class', imported=imported_members)
                ns['exceptions'], ns['all_exceptions'] = \
                    get_members(obj, 'exception', imported=imported_members)
                #print("EGN: module type: ", obj.__name__,"\nFNS: ",
                #      ns['functions'], "\nCLASSES:",ns['classes'])
            elif doc.objtype == 'class':
                ns['members'] = dir(obj)
                ns['methods'], ns['all_methods'] = \
                    get_members(obj, 'method', ['__init__'])
                ns['attributes'], ns['all_attributes'] = \
                    get_members(obj, 'attribute')

            parts = name.split('.')
            if doc.objtype in ('method', 'attribute'):
                mod_name = '.'.join(parts[:-2])
                cls_name = parts[-2]
                obj_name = '.'.join(parts[-2:])
                ns['class'] = cls_name
            else:
                mod_name, obj_name = '.'.join(parts[:-1]), parts[-1]

            ns['fullname'] = name
            ns['module'] = mod_name
            ns['objname'] = obj_name
            ns['name'] = parts[-1]

            ns['objtype'] = doc.objtype
            ns['underline'] = len(name) * '='

            rendered = template.render(**ns)
            f.write(rendered)  # type: ignore

    # descend recursively to new files
    if new_files:
        generate_autosummary_docs_patch(new_files, output_dir=output_dir,
                                        suffix=suffix, warn=warn, info=info,
                                        base_path=base_path, builder=builder,
                                        template_dir=template_dir, app=app)

        
def quiet_old_status_iterator(iterable, summary, color="darkgreen", stringify_func=display_chunk):
    # type: (Iterable, unicode, str, Callable[[Any], unicode]) -> Iterator
    l = 0
    for item in iterable:
        if l == 0:
            logger.info(bold(summary), nonl=True)
            l = 1
        #QUIET
        #logger.info(stringify_func(item), color=color, nonl=True)
        #logger.info(" ", nonl=True)
        yield item
    if l == 1:
        logger.info('')
        

def quiet_status_iterator(iterable, summary, color="darkgreen", length=0, verbosity=0,
                          stringify_func=display_chunk):
    # type: (Iterable, unicode, str, int, int, Callable[[Any], unicode]) -> Iterable  # NOQA
    if length == 0:
        for item in quiet_old_status_iterator(iterable, summary, color, stringify_func):
            yield item
        return
    l = 0
    summary = bold(summary)
    logger.info(summary + " QUIET")
    for item in iterable:
        l += 1
        #QUIET
        #s = '%s[%3d%%] %s' % (summary, 100 * l / length, colorize(color, stringify_func(item)))
	#if verbosity:
        #    s += '\n'
        #else:
        #    s = term_width_line(s)
        #logger.info(s, nonl=True)
        yield item
    if l > 0:
        logger.info('')

