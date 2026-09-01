"""
Javascript bootstrap used to make Workspace output work inside a Jupyter notebook.

Historically pyGSTi relied on the *classic* Jupyter Notebook, which exposed
`jquery`, `jquery-UI`, `plotly`, `katex` and `autorender` as RequireJS (AMD)
modules on every notebook page.  JupyterLab >= 3 and Notebook >= 7 do not ship
RequireJS at all, and the HTML files produced by ``nbconvert`` ship a RequireJS
that knows nothing about those module names.  Any output cell whose javascript
was wrapped in ``require([...], function(...){...})`` therefore either raised
``ReferenceError: require is not defined`` or hung forever waiting on a module
that could never load.

This module replaces that dependency with a tiny loader of pyGSTi's own:

* ``pygsti_require(deps, callback)`` behaves like the AMD ``require`` that the
  emitted javascript expects, but resolves its five module names from globals
  that pyGSTi itself has loaded.  Calls made before the libraries have finished
  loading are queued and replayed, in order, once they are ready.
* the libraries are loaded with plain ``<script>`` tags -- either sequentially
  from a CDN (``connected=True``) or inlined directly into the output cell
  (``connected=False``).  Neither path needs an ``offline/`` directory next to
  the notebook, so neither depends on a notebook-server URL scheme.

The AMD globals ``define``/``require`` are left alone, with one exception: while
the CDN scripts are being fetched, ``window.define`` is temporarily hidden so
that the UMD wrappers in jquery-ui and plotly.js take their browser-global
branch instead of registering an anonymous AMD module (which a page-level
RequireJS would reject with "Mismatched anonymous define()").

Nothing here is used by report generation -- reports carry their own page
scaffolding and load these libraries with ordinary ``<script>`` tags.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import base64 as _base64
import mimetypes as _mimetypes
import os as _os
import re as _re

#: CDN locations used when ``connected=True``.  The order matters: jquery-ui is
#: a jQuery plugin and plotly's MathJax hooks want jQuery present already.
CDN_URLS = [
    ("jquery", "https://code.jquery.com/jquery-3.6.4.min.js"),
    ("jquery-UI", "https://code.jquery.com/ui/1.12.1/jquery-ui.min.js"),
    ("plotly", "https://cdn.plot.ly/plotly-3.0.1.min.js"),
    ("katex", "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js"),
    ("autorender", "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/contrib/auto-render.min.js"),
]

#: The same libraries as files under ``templates/offline``, used when
#: ``connected=False``.  Same order, same reasons.
OFFLINE_FILENAMES = [
    ("jquery", "jquery-3.6.4.min.js"),
    ("jquery-UI", "jquery-ui.min.js"),
    ("plotly", "plotly-3.0.1.min.js"),
    ("katex", "katex.min.js"),
    ("autorender", "auto-render.min.js"),
]


#: Defines ``window.pygsti_require`` and friends.  Emitted exactly once, as an
#: inline script, so that it is in place synchronously before any figure cell
#: below it is rendered.
LOADER_JS = r"""
(function() {
  if (window.pygsti_require) { return; }  // already initialized

  var modules = {};      // module name -> value, populated by pygsti_modules_ready
  var queued = [];       // [deps, callback] pairs waiting for the libraries
  var isReady = false;
  var stashedDefine;
  var haveStash = false;

  function resolve(deps) {
    return deps.map(function(name) { return modules[name]; });
  }

  function invoke(deps, callback) {
    try {
      callback.apply(null, resolve(deps));
    } catch (err) {
      console.error('pyGSTi: error running notebook figure javascript:', err);
    }
  }

  // Drop-in replacement for the AMD `require` that pyGSTi's emitted javascript
  // used to rely on.  Runs `callback` immediately if the libraries are loaded,
  // otherwise queues it (FIFO, so cell order is preserved).
  window.pygsti_require = function(deps, callback) {
    if (isReady) { invoke(deps, callback); }
    else { queued.push([deps, callback]); }
  };

  // Hide window.define while third-party UMD bundles are being loaded with
  // ordinary script elements, so they export browser globals rather than register
  // anonymous AMD modules with whatever RequireJS the page happens to have.
  window.pygsti_amd_suspend = function() {
    if (haveStash) { return; }
    stashedDefine = window.define;
    haveStash = true;
    try { window.define = undefined; } catch (err) { /* non-writable: nothing to do */ }
  };

  window.pygsti_amd_restore = function() {
    if (!haveStash) { return; }
    try { window.define = stashedDefine; } catch (err) { /* see above */ }
    haveStash = false;
  };

  // Bind the loaded libraries to the module names the emitted javascript uses,
  // then flush everything that was queued while they were loading.
  window.pygsti_modules_ready = function() {
    if (isReady) { return; }
    modules['jquery'] = window.jQuery;
    modules['jquery-UI'] = window.jQuery ? window.jQuery.ui : undefined;
    modules['plotly'] = window.Plotly;
    modules['katex'] = window.katex;
    modules['autorender'] = window.renderMathInElement;
    window.jQueryUI = modules['jquery-UI'];

    if (!window.plotman && typeof PlotManager !== 'undefined') {
      window.plotman = new PlotManager();
    }

    var missing = ['jquery', 'jquery-UI', 'plotly'].filter(
      function(name) { return !modules[name]; });
    if (missing.length) {
      console.error('pyGSTi: these libraries failed to load: ' + missing.join(', '));
    }

    isReady = true;
    var pending = queued;
    queued = [];
    pending.forEach(function(item) { invoke(item[0], item[1]); });
  };

  // Append script elements one at a time, in order, then call `done`.
  window.pygsti_load_scripts = function(urls, done) {
    var i = 0;
    function next() {
      if (i >= urls.length) { done(); return; }
      var url = urls[i++];
      var el = document.createElement('script');
      el.src = url;
      el.async = false;
      el.onload = next;
      el.onerror = function() {
        console.error('pyGSTi: failed to load ' + url);
        next();
      };
      document.head.appendChild(el);
    }
    next();
  };
})();
"""


def _offline_path(filename):
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "templates", "offline", filename)


def _read(filename):
    with open(_offline_path(filename), 'r', encoding='utf-8') as f:
        return f.read()


#: Only rewrite url(...) references to small raster images.  Webfonts are
#: deliberately excluded: KaTeX 0.7.1 alone ships 4 MB of them, and KaTeX is only
#: reached as a fallback when the front-end provides no MathJax of its own.
_MAX_DATA_URI_BYTES = 64 * 1024
_DATA_URI_EXTENSIONS = ('.png', '.gif', '.jpg', '.jpeg')

_CSS_URL_RE = _re.compile(r"""url\(\s*(['"]?)([^'")]+)\1\s*\)""")


def image_data_uri(filename):
    """
    A ``data:`` URI for an image under ``templates/offline/images``.

    Parameters
    ----------
    filename : str
        Name of the image file, e.g. ``"ui-icons_222222_256x240.png"``.

    Returns
    -------
    str
    """
    path = _offline_path(_os.path.join("images", filename))
    mime = _mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, 'rb') as f:
        return "data:%s;base64,%s" % (mime, _base64.b64encode(f.read()).decode('ascii'))


def inline_css(filename):
    """
    A ``<style>`` block holding one of the ``templates/offline`` style sheets.

    Relative ``url(...)`` references are rewritten to ``data:`` URIs where the
    referenced file is small enough, since a relative URL in notebook cell
    output resolves against the *page*, not the notebook's directory.  Anything
    too large (the KaTeX webfonts) is left alone and simply will not load.

    Parameters
    ----------
    filename : str
        Name of the css file relative to ``templates/offline``.

    Returns
    -------
    str
    """
    css = _read(filename)

    def _sub(match):
        url = match.group(2)
        if url.startswith(('data:', 'http:', 'https:', '//', '/')):
            return match.group(0)
        relpath = url.split('?')[0].split('#')[0]
        # some of these style sheets are written relative to the *output*
        # directory (where an `offline/` folder gets copied), not to themselves
        if relpath.startswith('offline/'):
            relpath = relpath[len('offline/'):]
        path = _offline_path(relpath)
        if not path.lower().endswith(_DATA_URI_EXTENSIONS):
            return match.group(0)
        if not _os.path.isfile(path) or _os.path.getsize(path) > _MAX_DATA_URI_BYTES:
            return match.group(0)
        mime = _mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, 'rb') as f:
            data = _base64.b64encode(f.read()).decode('ascii')
        return "url(data:%s;base64,%s)" % (mime, data)

    return "<style>\n" + _CSS_URL_RE.sub(_sub, css) + "\n</style>\n"


def loader_script():
    """
    The HTML defining ``pygsti_require`` and its helpers.

    Must appear in the notebook before any Workspace figure output.

    Returns
    -------
    str
    """
    return "<script type='text/javascript'>" + LOADER_JS + "</script>\n"


def library_script(connected):
    """
    The HTML that loads jQuery, jQuery-UI, plotly and KaTeX and then releases
    everything queued on ``pygsti_require``.

    Parameters
    ----------
    connected : bool
        If True the libraries are pulled from a CDN, which keeps the notebook
        small but requires an internet connection when the notebook is *viewed*.
        If False their sources are inlined into the output cell, which makes the
        notebook self-contained (and large).

    Returns
    -------
    str
    """
    if connected:
        urls = ",\n".join("  '%s'" % url for _, url in CDN_URLS)
        return ("<script type='text/javascript'>\n"
                "window.pygsti_amd_suspend();\n"
                "window.pygsti_load_scripts([\n%s\n], function() {\n"
                "  window.pygsti_amd_restore();\n"
                "  window.pygsti_modules_ready();\n"
                "});\n"
                "</script>\n") % urls

    # Offline: inline the sources.  Separate <script> blocks, because Jupyter
    # replays inline scripts synchronously and in document order, which gives us
    # the load ordering for free.
    parts = ["<script type='text/javascript'>window.pygsti_amd_suspend();</script>\n"]
    for _, filename in OFFLINE_FILENAMES:
        parts.append("<script type='text/javascript'>\n" + _read(filename) + "\n</script>\n")
    parts.append("<script type='text/javascript'>"
                 "window.pygsti_amd_restore(); window.pygsti_modules_ready();"
                 "</script>\n")
    return "".join(parts)
