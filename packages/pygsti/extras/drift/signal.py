"""Signal analysis functions for time-series data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.fftpack import fft as _fft
from scipy.fftpack import ifft as _ifft
from scipy import convolve as _convolve
import warnings as _warnings
import numpy.random as _rnd

try: from astropy.stats import LombScargle as _LombScargle
except: pass

from scipy.stats import chi2 as _chi2
from ... import objects as _obj


def spectrum(x, times=None, null_hypothesis=None, counts=1, frequencies='auto', transform='dct',
             returnfrequencies=True):
    """
    todo
    """
    if transform == 'dct' or transform == 'dft':

        if transform == 'dct':
            modes = dct(x, null_hypothesis, counts)
            powers = modes**2

        if transform == 'dft':
            modes = dft(x, null_hypothesis, counts)
            powers = _np.abs(modes)**2

        if returnfrequencies:
            if isinstance(frequencies, str):

                if times is None: freqs = None
                else: freqs = fourier_frequencies_from_times(times)

            return freqs, modes, powers

        else:
            return modes, powers

    elif transform == 'lsp':
        freqs, powers = lsp(x, times, frequencies, null_hypothesis, counts)
        modes = None
        if returnfrequencies:
            return freqs, modes, powers
        else:
            return modes, powers

    else:
        raise ValueError("Input `transform` type invalid!")


def standardizer(x, null_hypothesis=None, counts=1):
    """
    todo
    """
    mean = _np.mean(x)
    if null_hypothesis is None:
        null_hypothesis = mean / counts
        if null_hypothesis <= 0 or null_hypothesis >= 1:
            return None

    normalizer = _np.sqrt(counts * null_hypothesis * (1 - null_hypothesis))
    standardized_x = (x - counts * null_hypothesis) / normalizer

    return standardized_x


def unstandardizer(z, null_hypothesis, counts=1):
    """
    todo
    """
    return z * _np.sqrt(counts * null_hypothesis * (1 - null_hypothesis)) + counts * null_hypothesis


def dct(x, null_hypothesis=None, counts=1):
    """
    Returns the Type-II discrete cosine transform of y, with an orthogonal normalization, where
    y is an array with elements related to the x array by

    y[k] = (x[k] - null_hypothesis[k])/normalizer;
    normalizer = sqrt(counts*null_hypothesis[k]*(1-null_hypothesis[k])).

    If null_hypothesis is None, then null_hypothesis[k] is mean(x)/counts, for all k. This is
    with the exception that when mean(x)/counts = 0 or 1 (when the above y[k] is ill-defined),
    in which case the zero vector is returned.

    Parameters
    ----------
    x : array
        Data string, on which the normalization and discrete cosine transformation is performed. If
        counts is not specified, this must be a bit string.

    null_hypothesis : array, optional
        If not None, an array to use in the normalization before the dct. If None, it is
        taken to be an array in which every element is the mean of x.

    counts : int, optional
        A factor in the normalization, that should correspond to the counts-per-timestep (so
        for full time resolution this is 1).

    Returns
    -------
    array
        The dct modes described above.

    """
    standardized_x = standardizer(x, null_hypothesis, counts)

    if standardized_x is None:
        out = _np.ones(len(x))
        out[0] = 0.
        return out

    modes = _dct(standardized_x, norm='ortho')

    return modes


def idct(modes, null_hypothesis, counts=1):
    """
    Inverts the dct function.

    Parameters
    ----------
    modes : array
        The fourier modes to be transformed to the time domain.

    null_hypothesis : array
        The array that was used in the normalization before the dct. This is
        commonly the mean of the time-domain data vector. All elements of this
        array must be in (0,1).

     counts : int, optional
        A factor in the normalization, that should correspond to the counts-per-timestep (so
        for full time resolution this is 1).

    Returns
    -------
    array
        Inverse of the dct function

    """
    z = _idct(modes, norm='ortho')
    x = unstandardizer(z, null_hypothesis, counts)
    return x


def dft(x, null_hypothesis=None, counts=1):
    """
    todo
    """
    standardized_x = standardizer(x, null_hypothesis, counts)

    if standardized_x is None:
        out = _np.ones(len(x))
        out[0] = 0.
        return out

    modes = _fft(standardized_x) / _np.sqrt(len(x))

    return modes


def idft(modes, null_hypothesis, counts=1):
    """
    Inverts the dft function.

    Parameters
    ----------
    modes : array
        The fourier modes to be transformed to the time domain.

    null_hypothesis : array
        The array that was used in the normalization before the dct. This is
        commonly the mean of the time-domain data vector. All elements of this
        array must be in (0,1).

     counts : int, optional
        A factor in the normalization, that should correspond to the counts-per-timestep (so
        for full time resolution this is 1).

    Returns
    -------
    array
        Inverse of the dft function

    """
    z = _np.sqrt(len(x)) * _ifft(modes)
    x = unstandardizer(z, null_hypothesis, counts)
    return x


def lsp(x, times, frequencies='auto', null_hypothesis=None, counts=1):
    """
    Performs a Lomb-Scargle periodogram (lsp) on the input data, returning powers
    and frequencies.

    *** This function uses astropy, which is not a required dependency for pyGSTi ***

    Parameters
    ----------
    todo

    Returns
    -------
    todo

    """
    numtimes = len(x)
    if isinstance(frequencies, str):
        freq = frequencies_from_timestep((max(times) - min(times)) / numtimes, numtimes)
    else:
        freq = frequencies

    standardized_x = standardizer(x, null_hypothesis, counts)

    if standardized_x is None:
        out = _np.ones(len(freq))
        out[0] = 0.
        return out

    power = _LombScargle(times, standardized_x).power(freq, normalization='psd')

    return freq, power


def bartlett_spectrum(x, numspectra, counts=1, null_hypothesis=None, transform='dct'):
    """
    Calculates the Bartlett power spectrum. This involves splitting the data into disjoint
    sections of the same length, and generating a power spectrum for each such section,
    and then averaging all these power spectra.

    Parameters
    ----------
    x : array
        The data to calculate the spectrum for.

    numspectra : int
        The number of "chunks" to split the data into, with a spectra calculated for each
        chunk. Note that if len(`x`) / num_spectra is not an integer, then not all of the
        data will be used.

    counts : int, optional
        The number of "clicks" per time-step in x, used for standarizing the data.

    null_hypothesis : array, optional
        The null hypothesis that we're looking for violations of. If left as None then this
        is the no-drift null hypothesis, with the static probability set to the mean of the
        data.

    transform : str, optional
        The transform to use the generate the power spectra.

    Returns
    -------
    array
        The Bartlett power spectrum

    """
    length = int(_np.floor(len(x) / numspectra))

    if null_hypothesis is None: null_hypothesis = _np.mean(x) * _np.ones(len(x)) / counts

    spectra = _np.zeros((numspectra, length))
    bartlett_spectrum = _np.zeros(length)

    for i in range(0, numspectra):
        spectra[i, :] = spectrum(x[i * length:((i + 1) * length)], counts=counts,
                                 null_hypothesis=null_hypothesis[i * length:((i + 1) * length)])

    bartlett_spectrum = _np.mean(spectra, axis=0)

    return bartlett_spectrum


def dct_basisfunction(omega, times, starttime, timedif):
    """
    todo
    """
    return _np.array([_np.cos(omega * _np.pi * (t - starttime + 0.5) / timedif) for t in times])


def power_significance_threshold(significance, numtests, dof):
    """
    todo
    """
    threshold = _chi2.isf(significance / numtests, dof) / dof

    return threshold


def power_to_pvalue(power, dof):
    """
    todo
    """
    pvalue = 1 - _chi2.cdf(dof * power, dof)

    return pvalue


def maxpower_pvalue(maxpower, numpowers, dof):
    """
    Todo: docstring
    """
    pvalue = 1 - _chi2.cdf(maxpower * dof, dof) ** (numpowers - 1)

    return pvalue


def power_significance_quasithreshold(significance, numstats, dof, procedure='Benjamini-Hochberg'):
    """
    Todo
    """
    if procedure == 'Benjamini-Hochberg':
        quasithreshold = _np.array([_chi2.isf((numstats - i) * significance / numstats, dof) / dof
                                    for i in range(numstats)])
    else:
        raise ValueError("Can only obtain a quasithreshold for the Benjamini-Hochberg procedure!")

    return quasithreshold


def get_auto_frequencies(ds, transform='dct'):
    """
    Returns a set of frequencies to create spectra for, for the input data. These frequencies are
    in units of 1/unit where `unit` is the units of the time-stamps in the DataSet.
    Depending on the type of transform, what this function is doing has different interpretations:
    .....

    Parameters
    ----------
    transform: str, optional

    ds: DataSet or MultiDataset
        Contains time-series data that the "auto" frequencies are calculated for.

    Returns
    -------
    frequencies : list
        A list of lists, where each of the consituent lists is a set of frequencies that are the
        are the derived frequencies for one or more of the dataset rows (i.e., for one or circuits).
        It is common for this to be a length-1 list, containing a single set of frequencies that are
        the frequencies this function has chosen for every circuit.

    pointers : dict
        A dictionary that "points" from the index of a circuit (indexed in the order of ds.keys())
        to the index of the list of frequencies for that circuit. No entry in the dictionary for
        an index should be interpretted as 0. So this can be an empty dictionary (with the frequencies
        list then necessarily of length 1, containing a single set of frequencies for every circuit).

    """
    # future: make this function for the lsp.

    assert(transform in ('dct', 'dft', 'lsp')), "The type of transform is invalid!"
    # todo : This is only reasonable with data that is equally spaced per circuit and with the same 
    # time-step over circuits
    if isinstance(ds, _obj.MultiDataSet):
        dskey = list(ds.keys())[0]
        timestep = ds[dskey].get_meantimestep()
        rw = ds[dskey][list(ds[dskey].keys())[0]]
    elif isinstance(ds, _obj.DataSet):
        timestep = ds.get_meantimestep()
        rw = ds[list(ds.keys())[0]]
    else:
        raise ValueError("Input data must be a DataSet or MultiDataSet!")
    numtimes = rw.get_number_of_times()
    freqs = frequencies_from_timestep(timestep, numtimes)
    freqslist = [freqs, ]
    # This pointers list should have keys based that are indices, as required by the stabilityanalyzer object.
    freqpointers = {}

    return freqslist, freqpointers


def frequencies_from_timestep(timestep, numtimes):
    """
    Calculates the Fourier frequencies associated with a timestep and a total number of times. These frequencies
    are in 1/units, where `units` is the units of time in `times`.

    Parameters
    ----------
    timestep : float
        The time difference between each data point.

    numtimes : int
        The total number of times.

    Returns
    -------
    array
        The frequencies associated with Fourier analysis on data with these timestamps.

    """
    return _np.arange(0, numtimes) / (2 * timestep * numtimes)


def fourier_frequencies_from_times(times):
    """
    Calculates the Fourier frequencies from a set of times. These frequencies are in 1/units, where
    `units` is the units of time in `times`. Note that if the times are not exactly equally spaced,
    then the Fourier frequencies are ill-defined, and this returns the frequencies based on assuming
    that the time-step is the mean time-step. This is reasonable for small deviations from equally
    spaced times, but not otherwise.

    Parameters
    ----------
    times : list
        The times from which to calculate the frequencies

    Returns
    -------
    array
        The frequencies associated with Fourier analysis on data with these timestamps.

    """
    timestep = _np.mean(_np.diff(times))  # The average time step.
    T = times[-1] - times[0]  # The total time difference

    return frequencies_from_timestep(timestep, T)


def amplitudes_at_frequencies(freqInds, timeseries, times=None, transform='dct'):
    """
    todo
    """
    amplitudes = {}
    for o in timeseries.keys():

        if transform == 'dct':
            temp = _dct(timeseries[o], norm='ortho')[freqInds] / _np.sqrt(len(timeseries[o]) / 2)
            if 0. in freqInds:
                temp[0] = temp[0] / _np.sqrt(2)
            amplitudes[o] = list(temp)

        else:
            raise NotImplementedError("This function only currently works for the DCT!")

    return amplitudes


def sparsity(p):
    """
    Returns the Hoyer sparsity index of the input vector p. This is defined to be:

    HoyerIndex = (sqrt(l) - (|p|_1 / |p|_2)) / (sqrt(l) - 1)

    where l is the length of the vector and |.|_1 and |.|_2 are the 1-norm and 2-norm of the vector, resp.

    """
    n = len(p)
    return (_np.sqrt(n) - _np.linalg.norm(p, 1) / _np.linalg.norm(p, 2)) / (_np.sqrt(n) - 1)


def renormalizer(p, method='logistic'):
    """
    Takes an arbitrary input vector `p` and maps it to a vector bounded within [0,1].

    -   If `method` = "sharp", then it maps any value in `p` below zero to zero and any value in `p` above one to one.

    -   If `method` = "logistic" then it 'squashes' the vector around it's mean value using a logistic function. The
        exact transform for each element x of p is:

        x -> mean - nu + (2*nu) / (1 + exp(-2 * (x - mean) / nu))

        where mean is the mean value of p, and nu is min(1-mean,mean). This transformation is only sensible when the
        mean of p is within [0,1].

    Parameters
    ----------
    p : array of floats
        The vector with values to be mapped to [0,1]

    method : {'logistic', 'sharp'}
        The method for "squashing" the input vector.

    Returns
    -------
    numpy.array
        The input vector "squashed" using the specified method.

    """
    if method == 'logistic':

        mean = _np.mean(p)
        out = logistic_transform(p, mean)

    elif method == 'sharp':
        out = p.copy()
        out[p > 1] = 1.
        out[p < 0] = 0.

    else: raise ValueError("method should be 'logistic' or 'sharp'")

    return out


def logistic_transform(x, mean):
    """
    Transforms the input float `x` as:

        x ->  mean - nu + (2*nu) / (1 + exp(-2 * (x - mean) / nu))

    where nu is min(1-mean,mean). This is a form of logistic transform, and maps x to a value in [0,1].
    """
    nu = min([1 - mean, mean])
    out = mean - nu + (2 * nu) / (1 + _np.exp(-2 * (x - mean) / nu))
    return out


def lowpass_filter(data, max_freq=None, transform='dct'):
    """
    Implements a low-pass filter on the input array, by Fourier transforming the input, mapping all but the lowest
    `max_freq` modes to zero, and then inverting the transform.

    Parameters
    ----------
    data : numpy.array,
        The vector to low-pass filter

    max_freq : None or int, optional
        The highest frequency to keep. If None then it keeps the minimum of 50 or l/10 frequencies, where l is the
        length of the data vector

    transform : str in ('dct','dft')
        The type of transform to use: the type-II discrete cosine transform or the discrete Fourier transform.

    Returns
    -------
    numpy.array
        The low-pass-filtered data.
    """
    n = len(data)

    if max_freq is None:
        max_freq = min(int(_np.ceil(n / 10)), 50)

    if transform == 'dct':
        modes = _dct(data, norm='ortho')
    if transform == 'dft':
        modes = _fft(data, norm='ortho')

    if max_freq < n - 1:
        modes[max_freq + 1:] = _np.zeros(len(data) - max_freq - 1)

    if transform == 'dct':
        out = _idct(modes, norm='ortho')
    if transform == 'dft':
        out = _ifft(modes, norm='ortho')
    return out


def moving_average(sequence, width=100):
    """
    Implements a moving average on `sequence` with an averaging width of `width`.
    """
    seq_length = len(sequence)
    base = _convolve(_np.ones(seq_length), _np.ones((int(width),)) / float(width), mode='same')
    signal = _convolve(sequence, _np.ones((int(width),)) / float(width), mode='same')

    return signal / base


def generate_flat_signal(power, nummodes, n, candidatefreqs=None, base=0.5, method ='sharp'):
    """
    Generates a minimal sparsity probability trajectory for the specified power, and a specified number of modes
    containing non-zero power. This is a probability trajectory that has a signal power of `power` (where, for a
    probability trajectory p, the signal vector is s = p - mean(p)), and has that power equally spread over `nummodes`
    different frequencies. The phases on the amplitudes are randomized.

    Parameters
    ----------
    power : float
        The amount of power in the signal of the generated probability trajectory

    nummodes : int
        The number of modes to equally spread that power over.

    n : int
        The number of sample times that the probability trajectory is being created for.

    candidatefreqs : list, optional
        A list containing a subset of 1,2,...,n-1. If not None, then all frequencies are included.


    base : float in (0,1), optional
        The mean of the generated probability trajectory.

    method : str, optional
        The method used to guarantee that the created probability trajectory is a valid probability,
        i.e., is within [0,1]. Options:

            - None. In this case nothing is done to guarantee that the output is a valid probability
                    trajectory, but the power in the signal is guaranteed to be the input power.

            - 'sharp' or 'logistic'. The 'sharp' or 'logistic' methods with the `renormalizer()`
                    function are used to map the generated vector onto [0,1]. When the renormalizer
                    does a non-trivial action on the generated vector, the power in the output will
                    be *below* the requested power. Moreover, the power will not be distributed only
                    over the requested number of modes (i.e., it impacts the spectrum of the output
                    vector).

    Returns
    -------
    array
        A probability trajectory.

    """
    amppermode = _np.sqrt(power / nummodes)
    if candidatefreqs is None: candidatefreqs = _np.arange(1, n)

    freqs = _np.random.choice(candidatefreqs, size=nummodes, replace=False, p=None)
    modes = _np.zeros(n, float)
    random_phases = _np.random.binomial(1, 0.5, size=nummodes)

    for i in range(0, nummodes):
        modes[freqs[i]] = amppermode * (-1)**random_phases[i]

    p = idct(modes, base * _np.ones(n))

    if method is not None:
        p = renormalizer(p, method=method)

    return p


def generate_gaussian_signal(power, center, spread, N, base=0.5, method='sharp'):
    """
    Generates a minimal sparsity probability trajectory for the specified power, and a specified number of modes
    containing non-zero power. This is a probability trajectory that has a signal power of `power` (where, for a
    probability trajectory p, the signal vector is s = p - mean(p)), and has that power equally spread over `nummodes`
    different frequencies. The phases on the amplitudes are randomized.

    Parameters
    ----------
    power : float
        The amount of power in the signal of the generated probability trajectory

    nummodes : int
        The number of modes to equally spread that power over.

    n : int
        The number of sample times that the probability trajectory is being created for.

    candidatefreqs : list, optional
        A list containing a subset of 1,2,...,n-1. If not None, then all frequencies are included.


    base : float in (0,1), optional
        The mean of the generated probability trajectory.

    method : str, optional
        The method used to guarantee that the created probability trajectory is a valid probability,
        i.e., is within [0,1]. Options:

            - None. In this case nothing is done to guarantee that the output is a valid probability
                    trajectory, but the power in the signal is guaranteed to be the input power.

            - 'sharp' or 'logistic'. The 'sharp' or 'logistic' methods with the `renormalizer()`
                    function are used to map the generated vector onto [0,1]. When the renormalizer
                    does a non-trivial action on the generated vector, the power in the output will
                    be *below* the requested power. Moreover, the power will not be distributed only
                    over the requested number of modes (i.e., it impacts the spectrum of the output
                    vector).

    Returns
    -------
    array
        A probability trajectory.

    """
    modes = _np.zeros(N)
    modes[0] = 0.
    modes[1:] = _np.exp(-1 * (_np.arange(1, N) - center)**2 / (2 * spread**2))
    modes = modes * (-1)**_np.random.binomial(1, 0.5, size=N)
    modes = _np.sqrt(power) * modes / _np.sqrt(sum(modes**2))

    p = idct(modes, base * _np.ones(N))

    if method is not None:
        p = renormalizer(p, method=method)

    return p
