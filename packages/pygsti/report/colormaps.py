from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Colormap and derived class definitions """

import numpy as _np
from scipy.stats import chi2 as _chi2

def _vnorm(x, vmin, vmax):
    #Perform linear mapping from [vmin,vmax] to [0,1]
    # (which is just a *part* of the full mapping performed)
    if _np.isclose(vmin,vmax): return _np.ma.zeros(x.shape,'d')
    return _np.clip( (x-vmin)/ (vmax-vmin), 0.0, 1.0)


class Colormap(object):
    def __init__(self, rgb_colors, hmin, hmax):
        self.rgb_colors = rgb_colors
        self.hmin = hmin
        self.hmax = hmax
        
    def _brightness(self,R,G,B):
        # Perceived brightness calculation from http://alienryderflex.com/hsp.html
        return _np.sqrt(0.299*R**2 + 0.587*G**2 + 0.114*B**2)

    def besttxtcolor(self, value):
        z = _vnorm( self.normalize(value), self.hmin, self.hmax) # norm_value <=> color
        for i in range(1,len(self.rgb_colors)):
            if z < self.rgb_colors[i][0]:
                z1,rgb1 = self.rgb_colors[i-1]
                z2,rgb2 = self.rgb_colors[i]
                alpha = (z-z1)/(z2-z1)
                R,G,B = [rgb1[i] + alpha*(rgb2[i]-rgb1[i]) for i in range(3)]
                break
        else: R,G,B = self.rgb_colors[-1][1] #just take the final color

        # Perceived brightness calculation from http://alienryderflex.com/hsp.html
        P = self._brightness(R,G,B)
        #print("DB: value = %f (%s), RGB = %f,%f,%f, P=%f (%s)" % (value,z,R,G,B,P,"black" if 0.5 <= P else "white"))
        return "black" if 0.5 <= P else "white"

    def get_colorscale(self):
        plotly_colorscale = [ [z, 'rgb(%d,%d,%d)' %
                               (round(r*255),round(g*255),round(b*255))]
                              for z,(r,g,b) in self.rgb_colors ]
        return plotly_colorscale


    

class LinlogColormap(Colormap):
    def __init__(self, vmin, vmax, n_boxes, pcntle, dof_per_box, color="red"):
        self.N = n_boxes
        self.percentile = pcntle
        self.dof = dof_per_box
        self.vmin = vmin
        self.vmax = vmax
        hmin = 0  #we'll normalize all values to [0,1] and then
        hmax = 1  # plot.ly will map this range linearly to (also) [0,1]
                  # range of our (and every) colorscale.
        
        N = max(self.N,1) #don't divide by N == 0 (if there are no boxes)
        self.trans = _np.ceil(_chi2.ppf(1 - self.percentile / N, self.dof))
          # the linear-log transition point

        # Colors ranging from white to gray on [0.0, 0.5) and pink to red on
        # [0.5, 1.0] such that the perceived brightness of the pink matches the
        # gray.
        gray = (0.4,0.4,0.4)
        if color == "red":
            c = (0.77, 0.143, 0.146); mx = (1.0, 0, 0)
        elif color == "blue":
            c = (0,0,0.7); mx = (0,0,1.0)
        elif color == "green":
            c = (0.0, 0.483, 0.0); mx = (0, 1.0, 0)
        elif color == "cyan":
            c = (0.0, 0.46, 0.46); mx = (0.0, 1.0, 1.0)
        elif color == "yellow":
            c = (0.415, 0.415, 0.0); mx = (1.0, 1.0, 0)
        elif color == "purple":
            c = (0.72, 0.0, 0.72); mx = (1.0, 0, 1.0)
        else:
            raise ValueError("Unknown color: %s" % color)

        super(LinlogColormap, self).__init__(
            [ [0.0, (1.,1.,1.)], [0.499999999, gray],
              [0.5, c], [1.0, mx] ], hmin,hmax)


    def normalize(self, value):
        """ 
        Scale value to a value between self.hmin and self.hmax (heatmap endpoints).

        Parameters
        ----------
        value : float or ndarray

        Returns
        -------
        float or ndarray
        """
        #Safety stuff -- needed anymore? TODO
        if isinstance(value, _np.ma.MaskedArray) and value.count() == 0:
            # no unmasked elements, in which case a matplotlib bug causes the
            # __call__ below to fail (numpy.bool_ has no attribute '_mask')
            return_value = _np.ma.array( _np.zeros(value.shape),
                                         mask=_np.ma.getmask(value))
            # so just create a dummy return value with the correct size
            # that has all it's entries masked (like value does)
            if return_value.shape==(): return return_value.item()
            else: return return_value.view(_np.ma.MaskedArray)

        #deal with numpy bug in handling masked nan values (nan still gives
        # "invalid value" warnings/errors even when masked)
        if _np.ma.is_masked(value):
            value = _np.ma.array(value.filled(1e100),
                                 mask=_np.ma.getmask(value))

        lin_norm_value = _vnorm(value, self.vmin, self.vmax)
        norm_trans = _vnorm(self.trans, self.vmin, self.vmax)
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

            if norm_trans == 1.0:
                #then transition is at highest possible normalized value (1.0)
                # and the call to greater(...) below will always be True.
                # To avoid the False-branch getting div-by-zero errors, set:
                log10_norm_trans = 1.0 # because it's never used.

            return_value = _np.ma.where(_np.ma.greater(norm_trans, lin_norm_value),
                                        lin_norm_value/(2*norm_trans),
                                        (log10_norm_trans -
                                         _np.ma.log10(lin_norm_value)) /
                                        (2*log10_norm_trans) + 0.5)

        if return_value.shape==():
            return return_value.item()
        else:
            return return_value.view(_np.ma.MaskedArray)



class DivergingColormap(Colormap):
    def __init__(self, vmin, vmax, midpoint=0.0, color="RdBu"):
        hmin = vmin
        hmax = vmax
        self.midpoint = midpoint #midpoint doesn't work yet!

        if color == "RdBu": # blue -> white -> red
            rgb_colors = [ [0.0, (0.0,0.0,1.0)],
                           [0.5, (1.0,1.0,1.0)],
                           [1.0, (1.0,0.0,0.0)] ]
        else:
            raise ValueError("Unknown color: %s" % color)

        super(DivergingColormap, self).__init__(rgb_colors, hmin, hmax)

        
    def normalize(self, value):
        #no normalization is done automatically by plotly,
        # (using zmin and zmax values of heatmap)
        return value

        #vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
        #
        #is_scalar = False
        #if isinstance(value, float) or _compat.isint(value, int):
        #    is_scalar = True
        #result = _np.ma.array(value)
        #
        #if not (vmin < midpoint < vmax):
        #    raise ValueError("midpoint must be between maxvalue and minvalue.")
        #elif vmin == vmax:
        #    result.fill(0) # Or should it be all masked? Or 0.5?
        #elif vmin > vmax:
        #    raise ValueError("maxvalue must be bigger than minvalue")
        #else:
        #    # ma division is very slow; we can take a shortcut
        #    resdat = result.filled(0) #masked entries to 0 to avoid nans
        #
        #    #First scale to -1 to 1 range, than to from 0 to 1.
        #    resdat -= midpoint
        #    resdat[resdat>0] /= abs(vmax - midpoint)
        #    resdat[resdat<0] /= abs(vmin - midpoint)
        #
        #    resdat /= 2.
        #    resdat += 0.5
        #    result = _np.ma.array(resdat, mask=result.mask, copy=False)
        #
        #if is_scalar:
        #    result = float(result)
        #return result


        

class SequentialColormap(Colormap):
    def __init__(self, vmin, vmax, color="whiteToBlack"):
        hmin = vmin
        hmax = vmax

        if color == "whiteToBlack":
            rgb_colors = [ [0, (1.,1.,1.)], [1.0, (0.0,0.0,0.0)] ]
        elif color == "blackToWhite":
            rgb_colors = [ [0, (0.0,0.0,0.0)], [1.0, (1.,1.,1.)] ]
        else:
            raise ValueError("Unknown color: %s" % color)

        super(SequentialColormap, self).__init__(rgb_colors, hmin,hmax)

    def normalize(self, value):
        #no normalization is done automatically by plotly,
        # (using zmin and zmax values of heatmap)
        return value

        #is_scalar = False
        #if isinstance(value, float) or _compat.isint(value, int):
        #    is_scalar = True
        #
        #result = _np.ma.array(value)
        #
        #if self.vmin == self.vmax:
        #    result.fill(0) # Or should it be all masked? Or 0.5?
        #elif self.vmin > self.vmax:
        #    raise ValueError("maxvalue must be bigger than minvalue")
        #else:
        #    resdat = result.filled(0) #masked entries to 0 to avoid nans
        #    resdat = _vnorm(resdat, self.vmin, self.vmax)
        #    result = _np.ma.array(resdat, mask=result.mask, copy=False)
        #
        #if is_scalar:
        #    result = result[0]
        #return result
        


