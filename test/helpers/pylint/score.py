#!/usr/bin/env python
from .lintAll           import lint_all

def get_score(source=lint_all):
    output     = source()
    scoreLine  = output[-2]
    # Score is located between leftmost slash and then rightmost space:
    # -> Your code has been rated at 9.57/10 (previous run: 9.57/10, +0.00)
    #                                    ^
    # -> Your code has been rated at 9.57
    #                               ^
    # -> 9.57
    score      = float(scoreLine.split('/', 1)[0].rsplit(' ', 1)[1])
    return score
