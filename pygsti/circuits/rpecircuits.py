#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Functions for creating circuit lists
"""
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti import tools as _tools


def make_rpe_alpha_str_lists_gx_gz(k_list):
    """
    Make alpha cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate alpha (Z rotation angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    cosStrList : list of Circuits
        The list of "cosine strings" to be used for alpha estimation.
    sinStrList : list of Circuits
        The list of "sine strings" to be used for alpha estimation.
    """
    cosStrList = []
    sinStrList = []
    for k in k_list:
        cosStrList += [_Circuit(('Gi', 'Gx', 'Gx', 'Gz')
                                + ('Gz',) * k
                                + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                'GiGxGxGzGz^' + str(k) + 'GzGzGzGxGx')]

        sinStrList += [_Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                + ('Gz',) * k
                                + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                'GxGxGzGzGz^' + str(k) + 'GzGzGzGxGx')]

        #From RPEToolsNewNew.py
        ##cosStrList += [_Circuit(('Gi','Gx','Gx')+
        ##                                ('Gz',)*k +
        ##                                ('Gx','Gx'),
        ##                                'GiGxGxGz^'+str(k)+'GxGx')]
        #
        #
        #cosStrList += [_Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GxGx')]
        #
        #
        #sinStrList += [_Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gz','Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GzGxGx')]

    return cosStrList, sinStrList


def make_rpe_epsilon_str_lists_gx_gz(k_list):
    """
    Make epsilon cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate epsilon (X rotation angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    epsilonCosStrList : list of Circuits
        The list of "cosine strings" to be used for epsilon estimation.
    epsilonSinStrList : list of Circuits
        The list of "sine strings" to be used for epsilon estimation.
    """
    epsilonCosStrList = []
    epsilonSinStrList = []

    for k in k_list:
        epsilonCosStrList += [_Circuit(('Gx',) * k
                                       + ('Gx',) * 4,
                                       'Gx^' + str(k) + 'GxGxGxGx')]

        epsilonSinStrList += [_Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                       + ('Gx',) * k
                                       + ('Gx',) * 4,
                                       'GxGxGzGzGx^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #epsilonCosStrList += [_Circuit(('Gx',)*k,
        #                                       'Gx^'+str(k))]
        #
        #epsilonSinStrList += [_Circuit(('Gx','Gx')+('Gx',)*k,
        #                                       'GxGxGx^'+str(k))]

    return epsilonCosStrList, epsilonSinStrList


def make_rpe_theta_str_lists_gx_gz(k_list):
    """
    Make theta cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate theta (X-Z axes angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    thetaCosStrList : list of Circuits
        The list of "cosine strings" to be used for theta estimation.
    thetaSinStrList : list of Circuits
        The list of "sine strings" to be used for theta estimation.
    """
    thetaCosStrList = []
    thetaSinStrList = []

    for k in k_list:
        thetaCosStrList += [_Circuit(
            ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4, '(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        thetaSinStrList += [_Circuit(
            ('Gx', 'Gx', 'Gz', 'Gz')
            + ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4,
            '(GxGxGzGz)(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #thetaCosStrList += [_Circuit(
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]
        #
        #thetaSinStrList += [_Circuit(
        #       ('Gx','Gx')+
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       'GxGx(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]

    return thetaCosStrList, thetaSinStrList


def make_rpe_string_list_d(log2k_max):
    """
    Creates a dictionary containing all the circuits needed for RPE.

    This includes circuits for all RPE cosine and sine experiments for all three angles.

    Parameters
    ----------
    log2k_max : int
        Maximum number of times to repeat an RPE "germ"

    Returns
    -------
    totalStrListD : dict
        A dictionary containing all circuits for all sine and cosine
        experiments for alpha, epsilon, and theta.
        The keys of the returned dictionary are:

        - 'alpha','cos' : List of circuits for cosine experiments used
          to determine alpha.
        - 'alpha','sin' : List of circuits for sine experiments used to
          determine alpha.
        - 'epsilon','cos' : List of circuits for cosine experiments used to
           determine epsilon.
        - 'epsilon','sin' : List of circuits for sine experiments used to
          determine epsilon.
        - 'theta','cos' : List of circuits for cosine experiments used to
          determine theta.
        - 'theta','sin' : List of circuits for sine experiments used to
          determine theta.
        - 'totalStrList' : All above circuits combined into one list;
          duplicates removed.
    """
    kList = [2**k for k in range(log2k_max + 1)]
    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)
    totalStrList = alphaCosStrList + alphaSinStrList + \
        epsilonCosStrList + epsilonSinStrList + \
        thetaCosStrList + thetaSinStrList
    totalStrList = _tools.remove_duplicates(totalStrList)  # probably superfluous

    stringListD = {}
    stringListD['alpha', 'cos'] = alphaCosStrList
    stringListD['alpha', 'sin'] = alphaSinStrList
    stringListD['epsilon', 'cos'] = epsilonCosStrList
    stringListD['epsilon', 'sin'] = epsilonSinStrList
    stringListD['theta', 'cos'] = thetaCosStrList
    stringListD['theta', 'sin'] = thetaSinStrList
    stringListD['totalStrList'] = totalStrList
    return stringListD



