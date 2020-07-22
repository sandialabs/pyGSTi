To rebuild GST docs, just do (from within docs/build directory):
```console
$ make html
$ ./install.sh
```

Here's a snipped from the sphinx install log that might be relevant:

  You should now populate your master file ./index.rst and create other documentation
  source files. Use the Makefile to build the docs, like so:
     make builder
  where "builder" is one of the supported builders, e.g. html, latex or linkcheck.
