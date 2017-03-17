

def besttxtcolor( x, colormap ):
    """
    Determinining function for whether text should be white or black

    Parameters
    ----------
    x : float
        Value of the cell in question
    cmap : matplotlib colormap
        Colormap assigning colors to the cells
    norm : matplotlib normalizer
        Function to map cell values to the interval [0, 1] for use by a
        colormap

    Returns
    -------
    {"white","black"}
    """
    cell_color = cmap(norm(x))
    R, G, B = cell_color[:3]
    # Perceived brightness calculation from http://alienryderflex.com/hsp.html
    P = _np.sqrt(0.299*R**2 + 0.587*G**2 + 0.114*B**2)
    return "black" if 0.5 <= P else "white"

class LinLogNorm(_matplotlib.colors.Normalize):
    def __init__(self, trans=None, vmin=None, vmax=None, clip=False):
        super(LinLogNorm, self).__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.trans = trans

    def inverse(self, value):
        norm_trans = super(LinLogNorm, self).__call__(self.trans)
        deltav = self.vmax - self.vmin
        return_value = _np.where(_np.greater(0.5, value),
                                 2*value*(self.trans - self.vmin) + self.vmin,
                                 deltav*_np.power(norm_trans, 2*(1 - value)))
        if return_value.shape==():
            return return_value.item()
        else:
            return return_value.view(_np.ma.MaskedArray)

    def __call__(self, value, clip=None):

        if isinstance(value, _np.ma.MaskedArray) and value.count() == 0:
            # no unmasked elements, in which case a matplotlib bug causes the
            # __call__ below to fail (numpy.bool_ has no attribute '_mask')
            return_value = _np.ma.array( _np.zeros(value.shape),
                                         mask=_np.ma.getmask(value))
            # so just create a dummy return value with the correct size
            # that has all it's entries masked (like value does)
            if return_value.shape==(): return return_value.item()
            else: return return_value.view(_np.ma.MaskedArray)

        lin_norm_value = super(LinLogNorm, self).__call__(value)

        if self.trans is None:
            self.trans = (self.vmax - self.vmin)/10 + self.vmin

        norm_trans = super(LinLogNorm, self).__call__(self.trans)
        log10_norm_trans = _np.ma.log10(norm_trans)
        with _np.errstate(divide='ignore'):
            # Ignore the division-by-zero error that occurs when 0 is passed to
            # log10 (the resulting NaN is filtered out by the where and is
            # harmless).

            #deal with numpy bug in handling masked nan values (nan still gives
            # "invalid value" warnings/errors even when masked)
            if _np.ma.is_masked(lin_norm_value):
                lin_norm_value = _np.ma.array(lin_norm_value.filled(1e100),
                                              mask=_np.ma.getmask(lin_norm_value))
            return_value = _np.ma.where(_np.ma.greater(norm_trans, lin_norm_value),
                                        lin_norm_value/(2*norm_trans),
                                        (log10_norm_trans -
                                         _np.ma.log10(lin_norm_value)) /
                                        (2*log10_norm_trans) + 0.5)

        if return_value.shape==():
            return return_value.item()
        else:
            return return_value.view(_np.ma.MaskedArray)

class MidPointNorm(_matplotlib.colors.Normalize):
    """
    A class for normalizing data which takes on
    positive and negative values.

    Taken from http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        super(MidPointNorm, self).__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = _np.ma.getmask(result)
                result = _np.ma.array(_np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.filled(0) #masked entries to 0 to avoid nans

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = _np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if _matplotlib.cbook.iterable(value):
            val = _np.ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0] *= abs(vmax - midpoint) #pylint: disable=unsubscriptable-object
            val[val<0] *= abs(vmin - midpoint) #pylint: disable=unsubscriptable-object
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

def splice_cmaps(cmaps, name=None, splice_points=None):
    """
    Take a list of cmaps and create a new cmap that joins them at specified
    points.

    Parameters
    ----------
    cmaps : list of matplotlib.colors.Colormap
        The colormaps ordered according to how they should appear in the final
        colormap

    name : string, optional
        The name for the colormap. If no name is given, the name
        ``"spliced_cmap1name_cmap2name_..."`` is assigned to the colormap.

    splice_points : ordered list of floats in (0, 1), optional
        The transition points when one colormap should end and the next should
        begin. Should have one less point than the number of cmaps provided. If
        no list is provided, the splice points will be arranged to split the
        interval (0, 1) up into equal seqments.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A cmap combining the provided cmaps
    """
    if name is None:
        name = '_'.join(['spliced'] + [cmap.name for cmap in cmaps])

    n_cmaps = len(cmaps)

    if splice_points is None:
        splice_points = _np.linspace(0, 1, n_cmaps + 1)[1:-1].tolist()

    n_sps = len(splice_points)

    if n_sps != n_cmaps - 1:
        raise ValueError(('The number of splice points, {0}, is not one less' +
            ' than the number of colormaps, {1}.').format(n_sps, n_cmaps))

    ranges = list(zip([0.0] + splice_points, splice_points + [1.0]))

    red_list = []
    green_list = []
    blue_list = []
    alpha_list = []

    # First segment
    cmap = cmaps[0]
    N = cmap.N
    low_val, high_val = ranges[0]
    input_values = _np.linspace(0.0, 1.0, N)
    scaled_values = _np.linspace(low_val, high_val, N)
    colors = cmap(input_values)
    for color, value in zip(colors[:-1], scaled_values[:-1]):
        r, g, b, a = color
        red_list.append((value, r, r))
        green_list.append((value, g, g))
        blue_list.append((value, b, b))
        alpha_list.append((value, a, a))

    # Middle segments
    for cmap, prev_cmap, rng in zip(cmaps[1:-1], cmaps[:-2], ranges[1:-1]):
        N = cmap.N
        low_val, high_val = rng
        input_values = _np.linspace(0.0, 1.0, N)
        scaled_values = _np.linspace(low_val, high_val, N)
        colors = cmap(input_values)
        prev_r, prev_g, prev_b, prev_a = prev_cmap(1.0)
        r, g, b, a = colors[0]
        red_list.append((low_val, prev_r, r))
        green_list.append((low_val, prev_g, g))
        blue_list.append((low_val, prev_b, b))
        alpha_list.append((low_val, prev_a, a))
        for color, value in zip(colors[1:-1], scaled_values[1:-1]):
            r, g, b, a = color
            red_list.append((value, r, r))
            green_list.append((value, g, g))
            blue_list.append((value, b, b))
            alpha_list.append((value, a, a))

    # Final segment
    cmap = cmaps[-1]
    prev_cmap = cmaps[-2]
    N = cmap.N
    low_val, high_val = ranges[-1]
    input_values = _np.linspace(0.0, 1.0, N)
    scaled_values = _np.linspace(low_val, high_val, N)
    colors = cmap(input_values)
    prev_r, prev_g, prev_b, prev_a = prev_cmap(1.0)
    r, g, b, a = colors[0]
    red_list.append((low_val, prev_r, r))
    green_list.append((low_val, prev_g, g))
    blue_list.append((low_val, prev_b, b))
    alpha_list.append((low_val, prev_a, a))
    for color, value in zip(colors[1:], scaled_values[1:]):
        r, g, b, a = color
        red_list.append((value, r, r))
        green_list.append((value, g, g))
        blue_list.append((value, b, b))
        alpha_list.append((value, a, a))

    cdict = {'red': red_list, 'green': green_list, 'blue': blue_list,
             'alpha': alpha_list}
    spliced_cmap = _matplotlib.colors.LinearSegmentedColormap(name, cdict)

    # return name, splice_points, cdict, spliced_cmap

    return spliced_cmap

def make_linear_cmap(start_color, final_color, name=None):
    """
    Make a color map that simply linearly interpolates between a start color
    and final color in RGB(A) space.

    Parameters
    ----------
    start_color : 3- (or 4-) tuple
        The (r, g, b[, a]) values for the start color.

    final_color : 3- (or 4-) tuple
        The (r, g, b[, a]) values for the final color.

    name : string
        A name for the colormap. If not provided, a name will be constructed
        from the colors at the two endpoints.

    Returns
    -------
    A cmap that interpolates between the endpoints in RGB(A) space.
    """
    labels = ['red', 'green', 'blue', 'alpha']
    cdict = {label: [(0, start_color[idx], start_color[idx]),
                     (1, final_color[idx], final_color[idx])]
             for label, idx in zip(labels, list(range(len(start_color))))}

    if name is None:
        name = 'linear_' + str(start_color) + '-' + str(final_color)

    return _matplotlib.colors.LinearSegmentedColormap(name, cdict)


def get_transition(N, eps=.1):
    '''
    Computes the transition point for the LinLogNorm class.

    Parameters
    ----------
    N : int
      number of chi2_1 random variables

    eps : float
      The quantile

    Returns
    -------
    trans : float
       An approximate 1-eps quantile for the maximum of N chi2_1 random
       variables.
    '''

    trans = _np.ceil(_chi2.ppf(1 - eps / N, 1))

    return trans



class StdColormapFactory(object):
    """
    Class used to create a standard GST colormap.
    """

    def __init__(self, kind, vmin=None, vmax=None, n_boxes=None, linlg_pcntle=.05, dof=1,\
                        midpoint=0):

        assert kind in ['linlog', 'div', 'seq'],\
            'Please instantiate the StdColormapFactory with a valid kind of colormap.'

        if kind != 'linlog':
            if (vmin is None) or (vmax is None):
                raise ValueError('vmin and vmax must both not be None for non-linlog colormap types.')
        else:
            if n_boxes is None:
                raise ValueError('linlog colormap type requires a non-None value for n_boxes.')

        self.kind = kind
        self.vmin = vmin
        self.vmax = vmax
        self.N = n_boxes
        self.percentile = linlg_pcntle
        self.dof = dof
        self.midpoint = midpoint

    def get_norm(self):
        #Creates the normalization class
        if self.kind == 'seq':
            norm = _matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax, clip=False)
        elif self.kind == 'div':
            norm = MidPointNorm(midpoint=self.midpoint, vmin=self.vmin, vmax=self.vmax)
        else:
            N = max(self.N,1) #don't divide by N == 0 (if there are no boxes)
            linlog_trans = _np.ceil(_chi2.ppf(1 - self.percentile / N, self.dof))
            norm = LinLogNorm(trans=linlog_trans)

        return norm

    def get_cmap(self):
        #Creates the colormap
        if self.kind == 'seq':
            cmap = _matplotlib.cm.get_cmap('Greys')
        elif self.kind == 'div':
            cmap = _matplotlib.cm.get_cmap('bwr')
        else:
            # Colors ranging from white to gray on [0.0, 0.5) and pink to red on
            # [0.5, 1.0] such that the perceived brightness of the pink matches the
            # gray.
            grayscale_cmap = make_linear_cmap((1, 1, 1), (0.5, 0.5, 0.5))
            red_cmap = make_linear_cmap((.698, .13, .133), (1, 0, 0))
            cmap = splice_cmaps([grayscale_cmap, red_cmap], 'linlog')

        cmap.set_bad('w',1)

        return cmap
