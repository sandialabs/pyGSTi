#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" This is a placeholder script to warn pyGSTi users of a change in project structure.

As of pyGSTi v0.9.9, the pyGSTi source directory has been moved from
`/packages/pygsti` to `/pygsti`. For most users, this change should be
completely imperceptible. However, if you have installed pyGSTi from
source in development mode, i.e. using `pip install -e .`, your pyGSTi
installation may now be broken.
"""

from pathlib import Path
import warnings

pygsti_root = Path(__file__).absolute().parent.parent.parent

instructions = """
\u001b[31m\u001b[1mIf you are seeing this message, you need to reinstall pyGSTi!\u001b[0m
Open a shell and run the following commands:

1. `cd {pygsti_root}`
2. `pip install -e .[complete]`
3. `python -c "import pygsti"`

After following these instructions, if you still see this message,
check to make sure that you don't have a GST.pth file located in
your local site-packages directory (try running `find ~ -name GST.pth`).

After removing any GST.pth files, if you're still seeing this
message, leave a bug report for the pyGSTi developers at
https://github.com/pyGSTio/pyGSTi/issues
""".format(pygsti_root=pygsti_root)

warnings.warn(__doc__ + instructions)
raise NotImplementedError()
