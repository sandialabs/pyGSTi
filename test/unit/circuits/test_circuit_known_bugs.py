"""Pins for verified, currently-unfixed Circuit bugs. None are open right now, so this
module holds no tests.

This is where bugs found by the characterization tests get recorded. When one turns up,
pin it here rather than leaving it undocumented:

    # KNOWN BUG, pyGSTi issue #NNN -- assertions pin the *buggy* behavior,

so the suite documents the bug and the eventual fix is forced to flip the pin in the
same PR that fixes it. If one of these tests goes red after your change, you have
probably fixed the referenced issue: flip or delete the pin in the same PR and note the
issue number.

A flipped pin does not stay here. It moves to whichever module owns the behavior it now
describes, since at that point it is an ordinary regression test.
"""
