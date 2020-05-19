"""
Defines the ReportText class
"""

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict as _OrderedDict
from .row import Row


class ReportText(object):
    """
    Text representation, renderable in multiple formats

    Parameters
    ----------
    text : str, bytes, or list
        The text to display, possibly in need of processing as
        specified by `fmt`.

    form : { None, "VerbosityPrinter" }
        What format `text` is in.  If None, then plain text that doesn't
        require any processing is assumed.  If `"VerbosityPrinter`", then
        `text` is assumed to be a list of `(type, level, message)` tuples
        as recorded by a :class:`VerbosityPrinter` object.
    """

    def __init__(self, text, form=None):
        '''
        Create a text-block object

        Parameters
        ----------
        text : str, bytes, or list
            The text to display, possibly in need of processing as
            specified by `fmt`.

        form : { None, "VerbosityPrinter" }
            What format `text` is in.  If None, then plain text that doesn't
            require any processing is assumed.  If `"VerbosityPrinter`", then
            `text` is assumed to be a list of `(type, level, message)` tuples
            as recorded by a :class:`VerbosityPrinter` object.
        '''
        self._raw = text
        self._format = form

    def render(self, fmt, text_id=None):
        """
        Render this text-block object

        Parameters
        ----------
        fmt : string
            name of format to be used

        text_id : string
            an ID to use in formats where it is applicable

        Returns
        -------
        string
        """
        if self._format is None:
            return {fmt: str(self._raw)}

        elif self._format == "VerbosityPrinter":
            if fmt == "html":

                if len(self._raw) == 0:
                    return {"html": "(No recorded output)"}

                out = "<ul class='collapsibleList'>"
                currentLevel = min([level for (_, level, _) in self._raw])
                li_open = False  # or could set to True and add
                # "<li>--- Output root (click to expand) ---" to out variable

                for i, (typ, level, msg) in enumerate(self._raw):
                    if typ == "LOG":
                        assert(level > 0), "Levels < 0 are not allowed!"
                        if level == currentLevel:
                            if li_open: out += "</li>\n"  # end last (open) li
                            out += "<li>"  # li for current item
                        elif level < currentLevel:  # less-indented level -- end open li and end open list(s)
                            if li_open: out += "</li>\n"  # end last (open) li
                            while level < currentLevel:
                                out += "</ul>\n</li>\n"; currentLevel -= 1  # end open lists
                            out += "<li>"  # add li for curren item
                        elif level > currentLevel:  # more-indented level -- open new list(s) within already open li
                            if not li_open: out += "<li>(unnamed level)"  # open an li to put ul tag within
                            out += "<ul>\n<li>"; currentLevel += 1
                            while level > currentLevel:
                                out += "(unnamed level)\n<ul>\n<li>"; currentLevel += 1

                        assert(level == currentLevel)
                        formatted_msg = msg.strip().strip("-:")
                        if len(self._raw) > i + 1 and self._raw[i + 1][1] > currentLevel:
                            formatted_msg = "<b>" + formatted_msg + "</b>"
                        out += formatted_msg  # DEBUG + ":%d" % currentLevel #add message to open li

                    elif typ == "WARNING":
                        if li_open: out += "</li>\n"
                        out += "<li>WWW %s" % msg.strip()
                    elif typ == "ERROR":
                        if li_open: out += "</li>\n"
                        out += "<li>EEE %s" % msg.strip()

                    li_open = True  # this loop always leaves an li tag open

                while currentLevel > 0:
                    out += "</li>\n</ul>\n"; currentLevel -= 1
                return {'html': out}

            elif fmt == "latex":
                return {"latex": "Rendering text blocks as latex is not supported yet."}

            elif fmt == "python":
                return {"python": self._raw}
            else:
                raise NotImplementedError("Cannot render text blocks in %s format!" % fmt)

        else:
            raise ValueError("Invalid `form` argument (passed to __init__): %s"
                             % self._format)

    def __str__(self):
        s = "*** ReportText object ***\n"
        s += self._raw  # TODO: process _raw as per _format
        return s
