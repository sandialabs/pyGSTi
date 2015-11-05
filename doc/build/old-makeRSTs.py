import inspect
import os, pkgutil



def makeRSTs(package, path=".", verbosity=0):
    objs = inspect.getmembers(package)
    pkgpath = os.path.dirname(package.__file__)
    true_modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    pkgname = package.__name__

    modules = []; functions = []; classes = []
    for name,obj in objs:
        if inspect.ismodule(obj):
            modules.append(name)
        if inspect.isclass(obj):
            classes.append(name)
        if inspect.isfunction(obj):
            functions.append(name)

    if verbosity > 1:
        print "Making .rst doc files for %s package" % pkgname
    if verbosity > 2:
        print "\n\nModules = \n", "\n".join(map(str,modules))
        print "\n\nTrue Modules = \n", "\n".join(map(str,true_modules))
        print "\n\nClasses = \n", "\n".join(map(str,classes))
        print "\n\nFunctions = \n", "\n".join(map(str,functions))

    ######################################
    #Create top level package rst
    ######################################

    filename = "%s/%s.rst" % (path,pkgname)
    f = open(filename, 'w')
    print >> f, pkgname
    print >> f, "="*len(pkgname),"\n"
    print >> f, ".. automodule:: %s" % pkgname, "\n"
    print >> f, "    .. rubric:: Custom Sub-modules\n"
    print >> f, "    .. autosummary::"
    print >> f, "       :template: my_autosummary_module.rst"
    print >> f, "       :toctree:\n"
    for moduleName in true_modules:
        print >> f, "       %s.%s" % (pkgname, moduleName)

    print >> f, "\n"
    print >> f, "    .. rubric:: Functions\n"
    print >> f, "    .. autosummary::"
    print >> f, "       :toctree:\n"
    for fnName in functions:
        print >> f, "       %s" % fnName


    print >> f, "\n"
    print >> f, "    .. rubric:: Classes\n"
    print >> f, "    .. autosummary::"
    print >> f, "       :toctree:\n"
    for className in classes:
        print >> f, "       %s" % className

    f.close()
    print "Wrote",filename

    ######################################
    #Create package classes rst files
    ######################################




if __name__ == "__main__":
    # Make RSTs for GST
    import GST
    makeRSTs(GST)
