"""Signal analysis functions for time-series data"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.fftpack import fft as _fft
from scipy.fftpack import ifft as _ifft
from scipy import convolve as _convolve
import warnings as _warnings
import numpy.random as _rnd

try: from astropy.stats import LombScargle as _LombScargle
except: _LombScargle = None

from scipy.stats import chi2 as _chi2
from ... import objects as _obj


def spectrum(x, times=None, null_hypothesis=None, counts=1, frequencies='auto', transform='dct',
             returnfrequencies=True):
    """
    Generates a power spectrum from the input time-series data. Before converting to a power
    spectrum, x is rescaled as

    x - >  (x - counts * null_hypothesis)  / sqrt(counts * null_hypothesis * (1-null_hypothesis)),

    where the arithmetic is element-wise, and `null_hypothesis` is a vector in (0,1).
    If `null_hypothesis` is None it is set to the mean of x. If that mean is 0 or 1 then
    the power spectrum returned is (0,1,1,1,...).

    Parameters
    ----------
    x: array
        The time-series data to convert into a power spectrum

    times: array, optional
        The times associated with the data in `x`. This is not optional for the `lsp` transform

    null_hypothesis: None or array, optional
        Used to normalize the data, and should be the null hypothesis that is being tested for
        the probability trajectory from which `x` is drawn. If `null_hypothesis` is None it is
        set to the mean of x.

    counts: int, optional
        The number of counts per time-step, whereby all values of `x` are within [0,counts].
        In the main usages for drift detection, `x` is the clickstream for a single measurement
        outcome -- so `x` contains integers between 0 and the number of measurements at a (perhaps
        coarse-grained) time. `counts` is this number of measurements per time.

    frequencies: 'auto' or array, optional
        The frequencies to generate the power spectrum for. Only relevant for transform=`lsp`.

    transform: 'dct', 'dft' or 'lsp', optional
        The transform to use to generate power spectrum. 'dct' is the Type-II discrete cosine transform
        with an orthogonal normalization; 'dft' is the discrete Fourier transform with a unitary
        normalization; 'lsp' is the float-meaning Lomb-Scargle periodogram with an orthogonal-like
        normalization.

    returnfrequencies: bool, optional
        Whether to return the frequencies corrsponding to the powers

    Returns
    -------
    if returnfrequencies:
        array or None
            The frequencies corresponding to the power spectrum. None is returned if the frequencies
            cannot be ascertained (when `times` is not specified).

    array or None
        The amplitudes, that are squared to obtain the powers. None is returned when the transform
        does not generate amplitudes (this is the case for `lsp`)

    array
        The power spectrum

    """
    if transform == 'dct' or transform == 'dft':

        if transform == 'dct':
            modes = dct(x, null_hypothesis, counts)
            powers = modes**2

        elif transform == 'dft':
            modes = dft(x, null_hypothesis, counts)
            powers = _np.abs(modes)**2

        if returnfrequencies:
            if isinstance(frequencies, str):

                if times is None: freqs = None
                else: freqs = fourier_frequencies_from_times(times)

            else:
                freqs = frequencies

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
    Maps the vector x over [0, counts] as

    x - >  (x - counts * null_hypothesis)  / sqrt(counts * null_hypothesis * (1-null_hypothesis)),

    where the arithmetic is element-wise, and `null_hypothesis` is a vector in (0,1).
    If `null_hypothesis` is None it is set to the mean of x. If that mean is 0 or 1 then
    None is returned.

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
    Inverts the `standardizer` function.
    """
    return z * _np.sqrt(counts * null_hypothesis * (1 - null_hypothesis)) + counts * null_hypothesis


def dct(x, null_hypothesis=None, counts=1):
    """
    Returns the Type-II discrete cosine transform of y, with an orthogonal normalization, where

    y = (x - counts * null_hypothesis)  / sqrt(counts * null_hypothesis * (1-null_hypothesis)),

    where the arithmetic is element-wise, and `null_hypothesis` is a vector in (0,1).
    If `null_hypothesis` is None it is set to the mean of x. If that mean is 0 or 1 then
    the vector of all ones, except for the first element which is set to zero, is returned.

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
        The DCT modes described above.

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
    Returns the discrete Fourier transform of y, with a unitary normalization, where
    y is an array with elements related to the x array by

    y = (x - counts * null_hypothesis)  / sqrt(counts * null_hypothesis * (1-null_hypothesis)),

    where the arithmetic is element-wise, and `null_hypothesis` is a vector in (0,1).
    If `null_hypothesis` is None it is set to the mean of x. If that mean is 0 or 1 then
    the vector of all ones, except for the first element which is set to zero, is returned.

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
        The DFT modes described above.

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
    z = _np.sqrt(len(modes)) * _ifft(modes)  # TIM CHECK THIS: len(*modes*) correct?
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

    if freq[0] == 0.:
        lspfreq = freq[1:]
    else:
        lspfreq = freq

    if _LombScargle is None:
        power = [0]  # TIM CHECK THIS??
    else:
        power = _LombScargle(times, standardized_x, fit_mean=True, center_data=False).power(lspfreq,
                                                                                            normalization='psd')

    if freq[0] == 0.: power = _np.array([0, ] + list(power))

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
        junk, powers = spectrum(x[i * length:((i + 1) * length)], counts=counts,
                                null_hypothesis=null_hypothesis[i * length:((i + 1) * length)],
                                returnfrequencies=False)
        spectra[i, :] = powers

    bartlett_spectrum = _np.mean(spectra, axis=0)

    return bartlett_spectrum


def dct_basisfunction(omega, times, starttime, timedif):
    """
    The `omega`th DCT basis function, for a initial time of `starttime` and a time-step of `timedif`,
    evaluated at the times `times`.

    """
    return _np.array([_np.cos(omega * _np.pi * (t - starttime + 0.5) / timedif) for t in times])


def power_significance_threshold(significance, numtests, dof):
    """
    The multi-test adjusted `signficance` statistical significance threshold for
    testing `numtests` test statistics that all have a marginal distribution that
    is chi2 with `dof` degrees of freedom.

    """
    threshold = _chi2.isf(significance / numtests, dof) / dof

    return threshold


def power_to_pvalue(power, dof):
    """
    Converts a power to a p-value, under the assumption that the power is chi2
    distribution with `dof` degrees of freedom.
    """
    pvalue = 1 - _chi2.cdf(dof * power, dof)

    return pvalue


def maxpower_pvalue(maxpower, numpowers, dof):
    """
    The p-value of the test statistic max(lambda_i) where there are `numpowers`
    lambda_i test statistics, and they are i.i.d. as chi2 with `dof` degrees
    of freedom. This approximates the p-value of the largest power in "clickstream"
    power spectrum (generated from `spectrum`), with the DOF given by the number of
    clicks per times.

    """
    pvalue = 1 - _chi2.cdf(maxpower * dof, dof) ** (numpowers - 1)

    return pvalue


def power_significance_quasithreshold(significance, numstats, dof, procedure='Benjamini-Hochberg'):
    """
    The Benjamini-Hockberg quasi-threshold for finding the statistically significant powers in
    a power spectrum.

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
    in units of 1 / unit where `unit` is the units of the time-stamps in the DataSet.
    What this function is doing has a fundmentally different meaning depending on whether the
    transform is time-stamp aware (here, the LSP) or not (here, the DCT and DFT).

    Time-stamp aware transforms take the frequencies to calculate powers at *as an input*, so this
    chooses these frequencies, which are, explicitly, the frequencies associated with the powers. The task
    of choosing the frequencies amounts to picking the best set of frequencies at which to interogate
    the true probability trajectory for components. As there are complex factors involved in this
    choice that the code has no way of knowing, sometimes it is best to choose them yourself. E.g.,
    if different frequencies are used for different circuits it isn't possible to (meaningfully)
    averaging power spectra across circuits, but this might be preferable if the time-step is
    sufficiently different between different circuits -- it depends on your aims.

    For time-stamp unaware transforms, these are the frequencies that, given that we're implementing
    the, e.g., DCT, the generated power spectrum is *implicitly* with respect to. In the case of data
    on a fixed time-grid, i.e., equally spaced data, then there is a precise set of frequencies implicit
    in the transform. Otherwise, these frequencies are explicitly at least slightly ad hoc, and choosing
    these frequencies amounts to choosing those frequencies that "best" approximate the properties being
    interogatted with fitting each, e.g., DCT basis function to the (timestamp-free) data.

    Parameters
    ----------
    ds: DataSet or MultiDataset
        Contains time-series data that the "auto" frequencies are calculated for.

    transform: str, optional
        The type of transform that the frequencies are being chosen for.


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


# Currently this is a trivial wrap-around for `frequencies_from_timestep, but in the future it might
# do something more subtle.
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
    numtimes = len(times)  # The number of times steps

    return frequencies_from_timestep(timestep, numtimes)


def amplitudes_at_frequencies(freq_indices, timeseries, times=None, transform='dct'):
    """
    Finds the amplitudes in the data at the specified frequency indices.
    Todo: better docstring. Currently only works for the DCT.
    """
    amplitudes = {}
    for o in timeseries.keys():

        if transform == 'dct':
            temp = _dct(timeseries[o], norm='ortho')[freq_indices] / _np.sqrt(len(timeseries[o]) / 2)
            if 0. in freq_indices:
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


def lowpass_filter(data, max_freq=None):
    """
    Implements a low-pass filter on the input array, by DCTing the input, mapping all but the lowest
    `max_freq` modes to zero, and then inverting the transform.

    Parameters
    ----------
    data : numpy.array,
        The vector to low-pass filter

    max_freq : None or int, optional
        The highest frequency to keep. If None then it keeps the minimum of 50 or l/10 frequencies, where l is the
        length of the data vector

    Returns
    -------
    numpy.array
        The low-pass-filtered data.
    """
    n = len(data)

    if max_freq is None:
        max_freq = min(int(_np.ceil(n / 10)), 50)

    modes = _dct(data, norm='ortho')

    if max_freq < n - 1:
        modes[max_freq + 1:] = _np.zeros(len(data) - max_freq - 1)

    out = _idct(modes, norm='ortho')

    return out


def moving_average(sequence, width=100):
    """
    Implements a moving average on `sequence` with an averaging width of `width`.
    """
    seq_length = len(sequence)
    base = _convolve(_np.ones(seq_length), _np.ones((int(width),)) / float(width), mode='same')
    signal = _convolve(sequence, _np.ones((int(width),)) / float(width), mode='same')

    return signal / base


def generate_flat_signal(power, nummodes, n, candidatefreqs=None, base=0.5, method='sharp'):
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


def generate_gaussian_signal(power, center, spread, n, base=0.5, method='sharp'):
    """
    Generates a probability trajectory with the specified power that is approximately Gaussian distribution
    accross frequencies, centered in the frequency specified by `center`. The probability trajectory has a
    rescaled signal power of `power`, where, for a probability trajectory p, the signal vector is s = p - mean(p),
    and the rescaled signal vector is p - mean(p) / sqrt(mean(p) * (1 - mean(p)). The phases on the amplitudes
    are randomized.

    Parameters
    ----------
    power : float
        The amount of power in the rescaled signal of the generated probability trajectory.

    center : int
        The mode on which to center the Gaussian.

    spread : int
        The spread of the Gaussian

    n : int
        The number of sample times that the probability trajectory is being created for.

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
    modes = _np.zeros(n)
    modes[0] = 0.
    modes[1:] = _np.exp(-1 * (_np.arange(1, n) - center)**2 / (2 * spread**2))
    modes = modes * (-1)**_np.random.binomial(1, 0.5, size=n)
    modes = _np.sqrt(power) * modes / _np.sqrt(sum(modes**2))

    p = idct(modes, base * _np.ones(n))

    if method is not None:
        p = renormalizer(p, method=method)

    return p
