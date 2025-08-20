""" Help section """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.section import Section as _Section


class HelpSection(_Section):
    _HTML_TEMPLATE = 'tabs/Help.html'

    @_Section.figure_factory()
    def example_table(workspace, switchboard, **kwargs):
        example_table = workspace.ExampleTable()
        example_table.set_render_options(click_to_display=True)
        return example_table
