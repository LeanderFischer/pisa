# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Map class to contain 2D histogram, error, and metadata about the contents.
MapSet class to contain a set of maps.

Also provide basic mathematical operations that user applies directly to the
containers but that get passed down to operate on the contained data.
"""


from __future__ import division

from collections import OrderedDict, Mapping, Sequence
from copy import deepcopy, copy
from operator import getitem, setitem
import re

import numpy as np
from scipy.stats import poisson
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp

from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numerical import normQuant
from pisa.utils.stats import chi2, llh


HASH_SIGFIGS = 12

def type_error(value):
    raise TypeError('Type of argument not supported: "%s"' % type(value))


def strip_outer_parens(value):
    value = value.strip()
    m = re.match(r'^\{\((.*)\)\}$', value)
    if m is not None:
        value = m.groups()[0]
    m = re.match(r'^\((.*)\)$', value)
    if m is not None:
        value = m.groups()[0]
    return value


class Map(object):
    """Class to contain a multi-dimensional histogram, error, and metadata
    about the histogram. Also provides basic mathematical operations for the
    contained data. See Examples below for how to use a Map object.


    Parameters
    ----------
    name : string
        Name for the map. Used to identify the map.

    hist : numpy.ndarray (incl. obj array from uncertainties.unumpy.uarray)
        The "data" (counts, etc.) in the map.  The shape of `hist` must be
        compatible with the `binning` specified.

    binning : MultiDimBinning
        Describes the binning of the Map.

    error_hist : numpy ndarray
        Must be same shape as `hist`. If specified, sets the error standard
        deviations for the contained `hist`, replacing any stddev information
        that might be contained in the passed `hist` arg.

    hash : None, or immutable object (typically an integer)
        Hash value to attach to the map.

    tex : None or string
        TeX string that can be used for e.g. plotting.

    full_comparison : bool
        Whether to perform full (recursive) comparisons when testing the
        equality of this map with another. See `__eq__` method.


    Examples
    --------
    >>> import pint; ureg = pint.UnitRegistry()
    >>> from pisa.core.binning import MultiDimBinning
    >>> binning = MultiDimBinning([dict(name='energy', is_log=True, num_bins=4,
    ...                                 domain=[1,80]*ureg.GeV),
    ...                            dict(name='coszen', is_lin=True, num_bins=5,
    ...                                 domain=[-1,0])])
    >>> m0 = Map(name='x', binning=binning, hist=np.zeros(binning.shape))
    >>> m0
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> m0.binning
    energy: 4 logarithmically-uniform bins spanning [1.0, 80.0] GeV
    coszen: 5 equally-sized bins spanning [-1.0, 0.0]
    >>> m0.hist[0:4, 0] = 1
    >>> m0
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.]])
    >>> m1 = m0[0:3, 0:2]
    >>> m1.binning
    energy: 3 logarithmically-uniform bins spanning [1.0, 26.7496121991]
    coszen: 2 equally-sized bins spanning [-1.0, -0.6]
    >>> m1
    array([[ 1.,  0.],
           [ 1.,  0.],
           [ 1.,  0.]])
    >>> for bin in m1.iterbins():
    ...     print '({0:~.2f}, {1:~.2f}): {2:0.1f}'.format(
    ...             bin.binning.energy.midpoints[0],
    ...             bin.binning.coszen.midpoints[0],
    ...             bin.hist[0,0])
    (2.00 GeV, -0.90 ): 1.0
    (2.00 GeV, -0.70 ): 0.0
    (5.97 GeV, -0.90 ): 1.0
    (5.97 GeV, -0.70 ): 0.0
    (17.85 GeV, -0.90 ): 1.0
    (17.85 GeV, -0.70 ): 0.0

    """
    _slots = ('name', 'hist', 'binning', 'hash', 'tex',
               'full_comparison', 'parent_indexer')
    _state_attrs = ('name', 'hist', 'binning', 'hash', 'tex',
                    'full_comparison')

    def __init__(self, name, hist, binning, error_hist=None, hash=None,
                 tex=None, full_comparison=True):
        # Set Read/write attributes via their defined setters
        super(Map, self).__setattr__('_name', name)
        # TeX dict for some common map names
        tex_dict = {'nue':r'\nu_e',
                    'numu':r'\nu_{\mu}',
                    'nutau':r'\nu_{\tau}',
                    'nuebar':r'\bar{\nu}_e',
                    'numubar':r'\bar{\nu}_{\mu}',
                    'nutaubar':r'\bar{\nu}_{\tau}',
                    'nue_cc':r'\nu_e\ CC',
                    'numu_cc':r'\nu_{\mu}\ CC',
                    'nutau_cc':r'\nu_{\tau}\ CC',
                    'nue_nc':r'\nu_e\ NC',
                    'numu_nc':r'\nu_{\mu}\ NC',
                    'nutau_nc':r'\nu_{\tau}\ NC',
                    'nuebar_cc':r'\bar{\nu}_e\ CC',
                    'numubar_cc':r'\bar{\nu}_{\mu}\ CC',
                    'nutaubar_cc':r'\bar{\nu}_{\tau}\ CC',
                    'nuebar_nc':r'\bar{\nu}_e\ NC',
                    'numubar_nc':r'\bar{\nu}_{\mu}\ NC',
                    'nutaubar_nc':r'\bar{\nu}_{\tau}\ NC'}
        if tex is None:
            if tex_dict.has_key(name):
                tex = tex_dict[name]
            else:
                tex = r'\rm{%s}' % name
        super(Map, self).__setattr__('_tex', tex)
        super(Map, self).__setattr__('_hash', hash)
        super(Map, self).__setattr__('_full_comparison', full_comparison)

        if not isinstance(binning, MultiDimBinning):
            if isinstance(binning, Sequence):
                binning = MultiDimBinning(dimensions=binning)
            elif isinstance(binning, Mapping):
                binning = MultiDimBinning(**binning)
            else:
                raise ValueError('Do not know what to do with `binning`=%s'
                                 %binning)
        self.parent_indexer = None

        # Do the work here to set read-only attributes
        super(Map, self).__setattr__('_binning', binning)
        binning.assert_array_fits(hist)
        super(Map, self).__setattr__('_hist', hist)
        if error_hist is not None:
            self.set_errors(error_hist)

    def __repr__(self):
        argstrs = [('%s=%s' %item) for item in self._serializable_state]
        return '%s(%s)' %(self.__class__.__name__, ', '.join(argstrs))

    def set_poisson_errors(self):
        """Approximate poisson errors using sqrt(n)."""
        super(Map, self).__setattr__('_hist', unp.uarray(self._hist,
                                                         np.sqrt(self._hist)))

    def set_errors(self, error_hist):
        """Manually define the error with an array the same shape as the
        contained histogram. Can also remove errors by passing None.

        Parameters
        ----------
        error_hist : None or ndarray (same shape as hist)
            Standard deviations to apply to `self.hist`.
            If None is passed, any errors present are removed, making
            `self.hist` a bare numpy array.

        """
        if error_hist is None:
            super(Map, self).__setattr__('_hist',
                                         unp.nominal_values(self._hist))
            return
        self.assert_compat(error_hist)
        super(Map, self).__setattr__('_hist', unp.uarray(self._hist,
                                                         error_hist))

    def new_obj(original_function):
        """ decorator to deepcopy unaltered states into new object """
        def new_function(self, *args, **kwargs):
            new_state = OrderedDict()
            state_updates = original_function(self, *args, **kwargs)
            for slot in self._state_attrs:
                if state_updates.has_key(slot):
                    new_state[slot] = state_updates[slot]
                else:
                    new_state[slot] = deepcopy(self.__getattr__(slot))
            return Map(**new_state)
        return new_function

    @new_obj
    def fluctuate(self, method, seed=None):
        orig = method
        method = str(method).lower()
        if method in ['poisson']:
            if seed is not None:
                np.random.seed(seed)
            return {'hist': unp.uarray(poisson.rvs(self.hist),
                                       np.sqrt(self.hist))}
        elif method in ['', 'none', 'false']:
            return
        else:
            raise Exception('fluctuation method %s not implemented' %orig)

    @property
    def shape(self):
        return self.hist.shape

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['hist'] = unp.nominal_values(self.hist)
        state['binning'] = self.binning._serializable_state
        stddevs = unp.std_devs(self.hist)
        stddevs = None if np.all(stddevs == 0) else stddevs
        state['error_hist'] = stddevs
        state['hash'] = self.hash
        state['tex'] = self.tex
        state['full_comparison'] = self.full_comparison
        return state

    @property
    def _hashable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['hist'] = normQuant(unp.nominal_values(self.hist),
                                  sigfigs=HASH_SIGFIGS)
        state['binning'] = self.binning._hashable_state
        stddevs = normQuant(unp.std_devs(self.hist), sigfigs=HASH_SIGFIGS)
        # TODO: better check here to see if the contained datatype is unp, as
        # opposed to 0 stddev (which could be the case but the user wants for
        # uncertainties to propagate)
        if np.all(stddevs == 0):
            stddevs = None
        else:
            stddevs = normQuant(stddevs, sigfigs=HASH_SIGFIGS)
        state['error_hist'] = stddevs
        state['tex'] = self.tex
        state['full_comparison'] = self.full_comparison
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new Map object from a JSON file.

        The format of the JSON is generated by the `Map.to_json` method, which
        converts a Map object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    def assert_compat(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            return
        elif isinstance(other, np.ndarray):
            self.binning.assert_array_fits(other)
        elif isinstance(other, Map):
            self.binning.assert_compat(other.binning)
        else:
            assert False, 'Unrecognized type'

    @new_obj
    def index(self, idx):
        if not isinstance(idx, Sequence) and len(idx) == 2:
            raise ValueError('Map is 2D; 2D indexing is required')
        # Indexing a single element e.g. hist[1,3] returns a 0D array while
        # e.g. hist[1,3:8] returns a 1D array; but we need 2D... so reshape
        # after indexing.
        new_hist = (self.hist)
        new_binning = (self.binning[idx])
        return {'binning': self.binning[idx],
                'hist': np.reshape(new_hist[idx], new_binning.shape)}

    def iterbins(self):
        """Returns a bin iterator which yields a map containing a single bin
        each time. Modifications to that map will be reflected in this (the
        parent) map.

        Note that the returned map has the attribute `parent_indexer` for
        indexing directly into to the parent map (or to a similar map).

        Yields
        ------
        Map object containing one of each bin of this Map

        """
        shape = self.shape
        for i in xrange(self.hist.size):
            idx_item = np.unravel_index(i, shape)
            idx_view = tuple([slice(x, x+1) for x in idx_item])
            single_bin_map = Map(
                name=self.name, hist=self.hist[idx_view],
                binning=self.binning[idx_item], hash=None, tex=self.tex,
                full_comparison=self.full_comparison
            )
            single_bin_map.parent_indexer = idx_item
            yield single_bin_map

    def iterindices(self):
        """Iterator that yields the index for accessing each bin in
        the map.

        >>> map = Map('x', binning=[dict('E', )

        """
        shape = self.shape
        for i in xrange(self.hist.size):
            idx_item = np.unravel_index(i, shape)
            yield idx_item

    def __repr__(self):
        return np.array_repr(self._hist)

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in slots"""
        if attr not in self._slots:
            raise ValueError('Attribute "%s" not allowed to be set.' % attr)
        super(Map, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        return super(Map, self).__getattribute__(attr)

    def __getitem__(self, idx):
        return self.index(idx)

    def llh(self, expected_values):
        """Calculate the total log-likelihood value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_llh : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(llh(actual_values=self.hist,
                          expected_values=expected_values))

    def chi2(self, expected_values):
        """Calculate the total chi-squared value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        Returns
        -------
        total_chi2 : float

        """
        if isinstance(expected_values, Map):
            expected_values = expected_values.hist
        return np.sum(chi2(actual_values=self.hist,
                           expected_values=expected_values))

    def __setitem__(self, idx, val):
        return setitem(self.hist, idx, val)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_name', value)

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_tex', value)

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        """Hash must be an immutable type (i.e., have a __hash__ method)"""
        assert hasattr(value, '__hash__')
        super(Map, self).__setattr__('_hash', value)

    @property
    def hist(self):
        return self._hist

    @property
    def nominal_values(self):
        return unp.nominal_values(self._hist)

    @property
    def std_devs(self):
        return unp.std_devs(self._hist)

    @property
    def binning(self):
        return self._binning

    @property
    def full_comparison(self):
        """Compare element-by-element instead of just comparing hashes."""
        return self._full_comparison

    @full_comparison.setter
    def full_comparison(self, value):
        assert isinstance(value, bool)
        super(Map, self).__setattr__('_full_comparison', value)

    # Common mathematical operators

    @new_obj
    def __abs__(self):
        return {#'name': "|%s|" % (self.name,),
                #'tex': r"{\left| %s \right|}" % strip_outer_parens(self.tex),
                'hist': np.abs(self.hist)}

    @new_obj
    def __add__(self, other):
        """Add `other` to self"""
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "(%s + %s)" % (self.name, other),
                #'tex': r"{(%s + %s)}" % (self.tex, other),
                'hist': self.hist + other
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "(%s + array)" % self.name,
                #'tex': r"{(%s + X)}" % self.tex,
                'hist': self.hist + other
            }
        elif isinstance(other, Map):
            dict = {
                #'name': "(%s + %s)" % (self.name, other.name),
                #'tex': r"{(%s + %s)}" % (self.tex, other.tex),
                'hist': self.hist + other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return dict

    #def __cmp__(self, other):

    @new_obj
    def __div__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "(%s / %s)" % (self.name, other),
                #'tex': r"{(%s / %s)}" % (self.tex, other),
                'hist': self.hist / other
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "(%s / array)" % self.name,
                #'tex': r"{(%s / X)}" % self.tex,
                'hist': self.hist / other
            }
        elif isinstance(other, Map):
            dict = {
                #'name': "(%s / %s)" % (self.name, other.name),
                #'tex': r"{(%s / %s)}" % (self.tex, other.tex),
                'hist': self.hist / other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return dict

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        raise NotImplementedError('floordiv not implemented for type Map')

    # TODO: figure out what we actually want to overload "==" with, and how
    # to implement all the possible kinds of "==" that might be useful for the
    # user, possibly with different methods altogether
    def __eq__(self, other):
        """Check if full state of maps are equal. *Not* element-by-element
        equality as for a numpy array. Call this.hist == other.hist for the
        nominal value and the error

        If `full_comparison` is true for *both* maps, or if either map lacks a
        hash, performs a full comparison of the contents of each map.

        Otherwise, simply checks that the hashes are equal.
        """
        if np.isscalar(other):
            # in case comparing with just with a scalar ignore the errors:
            return np.all(unp.nominal_values(self.hist) == other)
        if type(other) is uncertainties.core.Variable \
                or isinstance(other, np.ndarray):
            return (np.all(unp.nominal_values(self.hist)
                           == unp.nominal_values(other))
                    and np.all(unp.std_devs(self.hist)
                               == unp.std_devs(other)))
        elif isinstance(other, Map):
            if (self.full_comparison or other.full_comparison
                or self.hash is None or other.hash is None):
                return recursiveEquality(self._hashable_state,
                                         other._hashable_state)
            return self.hash == other.hash
        else:
            type_error(other)

    @new_obj
    def log(self):
        return {
            #'name': "log(%s)" % self.name,
            #'tex': r"\ln\left( %s \right)" % self.tex,
            'hist': np.log(self.hist)
        }

    @new_obj
    def log10(self):
        return {
            #'name': "log10(%s)" % self.name,
            #'tex': r"\log_{10}\left( %s \right)" % self.tex,
            'hist': np.log10(self.hist)
        }

    @new_obj
    def __mul__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "%s * %s" % (other, self.name),
                #'tex': r"%s \cdot %s" % (other, self.tex),
                'hist': self.hist * other
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "array * %s" % self.name,
                #'tex': r"X \cdot %s" % self.tex,
                'hist': self.hist * other,
            }
        elif isinstance(other, Map):
            dict = {
                #'name': "%s * %s" % (self.name, other.name),
                #'tex': r"%s \cdot %s" % (self.tex, other.tex),
                'hist': self.hist * other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return dict

    def __ne__(self, other):
        return not self == other

    @new_obj
    def __neg__(self):
        return {
            #'name': "-%s" % self.name,
            #'tex': r"-%s" % self.tex,
            'hist': -self.hist,
        }

    @new_obj
    def __pow__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "%s**%s" % (self.name, other),
                #'tex': "%s^{%s}" % (self.tex, other),
                'hist': np.power(self.hist, other)
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "%s**(array)" % self.name,
                #'tex': r"%s^{X}" % self.tex,
                'hist': np.power(self.hist, other),
            }
        elif isinstance(other, Map):
            dict = {
                #'name': "%s**(%s)" % (self.name, strip_outer_parens(other.name)),
                #'tex': r"%s^{%s}" % (self.tex, strip_outer_parens(other.tex)),
                'hist': np.power(self.hist, other.hist),
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return dict

    def __radd__(self, other):
        return self + other

    def __rdiv__(self, other):
        if isinstance(other, Map):
            return other / self
        else:
            return self.__rdiv(other)

    @new_obj
    def __rdiv(self,oher):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "(%s / %s)" % (other, self.name),
                #'tex': "{(%s / %s)}" % (other, self.tex),
                'hist': other / self.hist,
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "array / %s" % self.name,
                #'tex': "{(X / %s)}" % self.tex,
                'hist': other / self.hist,
            }
        else:
            type_error(other)
        return dict

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        if isinstance(other, Map):
            return other - self
        else:
            return self.__rsub(other)

    @new_obj
    def __rsub(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "(%s - %s)" % (other, self.name),
                #'tex': "{(%s - %s)}" % (other, self.tex),
                'hist': other - self.hist,
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "(array - %s)" % self.name,
                #'tex': "{(X - %s)}" % self.tex,
                'hist': other - self.hist,
            }
        else:
            type_error(other)
        return dict

    @new_obj
    def sqrt(self):
        return {
            #'name': "sqrt(%s)" % self.name,
            #'tex': r"\sqrt{%s}" % self.tex,
            'hist': np.sqrt(self.hist),
        }

    @new_obj
    def __sub__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            dict = {
                #'name': "(%s - %s)" % (self.name, other),
                #'tex': "{(%s - %s)}" % (self.tex, other),
                'hist': self.hist - other,
            }
        elif isinstance(other, np.ndarray):
            dict = {
                #'name': "(%s - array)" % self.name,
                #'tex': "{(%s - X)}" % self.tex,
                'hist': self.hist - other,
            }
        elif isinstance(other, Map):
            dict = {
                #'name': "%s - %s" % (self.name, other.name),
                #'tex': "{(%s - %s)}" % (self.tex, other.tex),
                'hist': self.hist - other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return dict

# TODO: instantiate individual maps from dicts if passed as such, so user
# doesn't have to instantiate each map. Also, check for name collisions with
# one another and with attrs (so that __getattr__ can retrieve the map by name)

# TODO: add docstrings

class MapSet(object):
    """
    Set of maps.


    Parameters
    ----------
    maps : one Map or a sequence of Map

    name : string

    tex : string

    collate_by_name : bool
        If True, when this MapSet is passed alongside another MapSet to a
        function that operates on the maps, contained maps in each will be
        accessed by name. Hence, only maps with the same names will be operated
        on simultaneously.

        If false, the contained maps in each MapSet will be accessed by their
        order in each MapSet. This behavior is useful if maps are renamed
        through some operation but their order is maintained, and then
        comparisons are sought with their progenitors with the original
        (different) name.

    """
    __slots = ('_name')
    __state_attrs = ('name', 'maps', 'tex', 'collate_by_name')
    def __init__(self, maps, name=None, tex=None, collate_by_name=True):
        maps_ = []
        for m in maps:
            if isinstance(m, Map):
                maps_.append(m)
            else:
                maps_.append(Map(**m))
        tex = (r'{\rm %s}' %name) if tex is None else tex
        super(MapSet, self).__setattr__('maps', tuple(maps_))
        super(MapSet, self).__setattr__('name', name)
        super(MapSet, self).__setattr__('tex', tex)
        super(MapSet, self).__setattr__('collate_by_name', collate_by_name)
        super(MapSet, self).__setattr__('collate_by_num', not collate_by_name)

    def __repr__(self):
        argstrs = [('%s=%s' %item) for item in self._serializable_state]
        return '%s(%s)' %(self.__class__.__name__, ', '.join(argstrs))

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['maps'] = [m._serializable_state for m in self.maps]
        state['name'] = self.name
        state['tex'] = self.tex
        state['collate_by_name'] = self.collate_by_name
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new MapSet object from a JSON file.

        The format of the JSON is generated by the `MapSet.to_json` method,
        which converts a MapSet object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    def __eq__(self, other):
        return recursiveEquality(self._hashable_state, other._hashable_state)

    @property
    def name(self):
        return super(MapSet, self).__getattribute__('_name')

    @name.setter
    def name(self, name):
        return super(MapSet, self).__setattr__('_name', name)

    @property
    def hash(self):
        hashes = self.hashes
        if all([(h is not None) for h in hashes]):
            return hash_obj(hashes)
        return None

    @property
    def names(self):
        return tuple([mp.name for mp in self])

    @property
    def hashes(self):
        return tuple([mp.hash for mp in self])

    def hash_maps(self, map_names=None):
        if map_names is None:
            map_names = [m.name for m in self]
        hashes = [m.hash for m in self if m.name in map_names]
        if all([(h != None) for h in hashes]):
            return hash_obj(hashes)
        return None

    def collate_with_names(self, vals):
        ret_dict = OrderedDict()
        [setitem(ret_dict, name, val) for name, val in zip(self.names, vals)]
        return ret_dict

    def find_map(self, value):
        idx = None
        if isinstance(value, Map):
            pass
        elif isinstance(value, basestring):
            try:
                idx = self.names.index(value)
            except ValueError:
                pass
        if idx is None:
            raise ValueError('Could not find map name "%s" in %s' %
                             (value, self))
        return self[idx]

    def apply_to_maps(self, attr, *args, **kwargs):
        if len(kwargs) != 0:
            raise NotImplementedError('Keyword arguments are not handled')

        if not all([hasattr(mp, attr) for mp in self]):
            raise AttributeError('All maps do not have attribute "%s"' % attr)

        # Retrieve the corresponding callables from contained maps
        val_per_map = [getattr(mp, attr) for mp in self]
        if not all([hasattr(meth, '__call__') for meth in val_per_map]):
            # If all results are maps, populate a new map set & return that
            if all([isinstance(r, Map) for r in val_per_map]):
                return MapSet(val_per_map)
            # Otherwise put in an ordered dict with <name>: <val> pairs ordered
            # according to the map ordering in the set
            return self.collate_with_names(val_per_map)

        # Rename for clarity
        method_name = attr
        method_per_map = val_per_map

        # Create a set of args for *each* map in this map set: If an arg is a
        # MapSet, convert that arg into the map in that set corresponding to
        # the same map in this set.
        args_per_map = []
        for map_num, mp in enumerate(self):
            map_name = mp.name
            this_map_args = []
            for arg in args:
                if (np.isscalar(arg) or
                        type(arg) is uncertainties.core.Variable or
                        isinstance(arg, (basestring, np.ndarray))):
                    this_map_args.append(arg)
                elif isinstance(arg, MapSet):
                    if self.collate_by_name:
                        this_map_args.append(arg[map_name])
                    elif self.collate_by_num:
                        this_map_args.append(arg[map_num])
                else:
                    raise TypeError('Unhandled arg %s / type %s' %
                                    (arg, type(arg)))
            args_per_map.append(tuple(this_map_args))

        # Make the method calls and collect returned values
        returned_vals = [meth(*args)
                         for meth, args in zip(method_per_map, args_per_map)]

        # If all results are maps, put them into a new map set & return
        if all([isinstance(r, Map) for r in returned_vals]):
            return MapSet(tuple(returned_vals))

        # If None returned by all, return a single None
        if all([(r is None) for r in returned_vals]):
            return

        # Otherwise put into an ordered dict with name: val pairs
        return self.collate_with_names(returned_vals)

    def __str__(self):
        if self.name is not None:
            my_name = "'" + self.name + "'"
        else:
            my_name = super(MapSet, self).__repr__()
        return "MapSet %s containing maps %s" % (my_name, self.names)

    def __repr__(self):
        return str(self)

    def __contains__(self, name):
        return name in [m.name for m in self]

    def __setattr__(self, attr, val):
        if attr in MapSet.__slots:
            object.__setattr__(attr, val)
        else:
            returned_vals = [setattr(mp, attr, val) for mp in self]
            if all([(r is None) for r in returned_vals]):
                return
            return self.collate_with_names(returned_vals)

    def __getattr__(self, attr):
        if attr in [m.name for m in self]:
            return self[attr]
        return self.apply_to_maps(attr)

    def __iter__(self):
        return iter(self.maps)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, item):
        """Retrieve a map by name or retrieve maps' histogram values by index
        or slice.

        If `item` is a string, retrieve map by name.
        If `item is an integer or one-dim slice, retrieve maps by sequence
        If `item` is length-2 tuple or two-dim slice, retrieve value(s) of all
            contained maps, each indexed by map[`item`]. The output is returned
            in an ordered dict with format {<map name>: <values>, ...}

        """
        if isinstance(item, basestring):
            return self.find_map(item)
        elif isinstance(item, (int, slice)):
            rslt = self.maps[item]
            if hasattr(rslt, '__len__') and len(rslt) > 1:
                return MapSet(rslt)
            return rslt
        elif isinstance(item, Sequence):
            if len(item) == 1:
                return self.maps[item]
            elif len(item) == 2:
                return MapSet([getitem(m, item) for m in self])
            else:
                raise IndexError('too many indices for 2D hist')
        #elif isinstance(item, Sequence):
        #    assert len(item) == 2, 'Maps are 2D, and so must be indexed as such'
        #    return self.collate_with_names([getitem(m, item) for m in self])
        else:
            raise TypeError('getitem does not support `item` of type %s'
                            % type(item))

    def __abs__(self):
        return self.apply_to_maps('__abs__')

    def __add__(self, val):
        return self.apply_to_maps('__add__', val)

    def __truediv__(self, val):
        return self.apply_to_maps('__truediv__', val)

    def __div__(self, val):
        return self.apply_to_maps('__div__', val)

    def log(self):
        return self.apply_to_maps('log')

    def log10(self):
        return self.apply_to_maps('log10')

    def __mul__(self, val):
        return self.apply_to_maps('__mul__', val)

    def __neg__(self):
        return self.apply_to_maps('__neg__')

    def __pow__(self, val):
        return self.apply_to_maps('__pow__', val)

    def __radd__(self, val):
        return self.apply_to_maps('__radd__', val)

    def __rdiv__(self, val):
        return self.apply_to_maps('__rdiv__', val)

    def __rmul__(self, val):
        return self.apply_to_maps('__rmul__', val)

    def __rsub__(self, val):
        return self.apply_to_maps('__rsub__', val)

    def sqrt(self):
        return self.apply_to_maps('sqrt')

    def __sub__(self, val):
        return self.apply_to_maps('__sub__', val)

    def metric_per_map(self, expected_values, metric):
        assert isinstance(metric, basestring)
        metric = metric.lower()
        if metric in ['chi2', 'llh']:
            return self.apply_to_maps(metric, expected_values)
        else:
            raise ValueError('`metric` "%s" not recognized; use either'
                             ' "chi2" or "llh".' %metric)

    def metric_total(self, expected_values, metric):
        return np.sum(self.metric_per_map(expected_values, metric).values())

    def chi2_per_map(self, expected_values):
        return self.apply_to_maps('chi2', expected_values)

    def chi2_total(self, expected_values):
        return np.sum(self.chi2_per_map(expected_values))

    def fluctuate(self, method):
        return self.apply_to_maps('fluctuate', method)

    def llh_per_map(self, expected_values):
        return self.apply_to_maps('llh', expected_values)

    def llh_total(self, expected_values):
        return np.sum(self.llh(expected_values))

    def set_poisson_errors(self):
        return self.apply_to_maps('set_poisson_errors')

## Now dynamically add all methods from Map to MapSet that don't already exist
## in MapSet (and make these methods distribute to contained maps)
##for method_name, method in sorted(Map.__dict__.items()):
#add_methods = '''__abs__ __add__ __div__ __mul__ __neg__ __pow__ __radd__
#__rdiv__ __rmul__ __rsub__ __sub__'''.split()
#
#for method_name in add_methods:
#    #if not hasattr(method, '__call__') or method_name in MapSet.__dict__:
#    #    continue
#    disallowed = ('__getattr__', '__setattr__', '__getattribute__',
#                  '__getitem__', '__eq__', '__ne__', '__str__', '__repr__')
#    if method_name in disallowed:
#        continue
#    print 'adding method "%s" to MapSet as an apply func' % method_name
#    arg_str = ', *args' # if len(args) > 0 else ''
#    eval('def {method_name}(self{arg_str}):\n'
#         '    return self.apply_to_maps({method_name}{arg_str})'.format(method_name=method_name, arg_str=arg_str))
#    #f.__doc__ = 'Apply method %s to all contained maps' % method_name
#    #method = getattr(Map, method_name)
#    #if method.__doc__:
#    #    f.__doc__ += '... ' + method.__doc__
#    setattr(MapSet, method_name, MethodType(eval(method_name), None, MapSet))


# TODO: add tests for llh, chi2 methods
def test_Map():
    import os
    import shutil
    import tempfile
    import pint; ureg = pint.UnitRegistry()

    n_ebins = 10
    n_czbins = 5
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1,80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1,0), is_lin=True)
    # set directly unumpy array with errors
    #m1 = Map(name='x', hist=unp.uarray(np.ones((40,20)),np.sqrt(np.ones((40,20)))), binning=(e_binning, cz_binning))
    # or call init poisson error afterwards
    m1 = Map(name='x', hist=np.ones((n_ebins, n_czbins)), binning=(e_binning,
                                                                   cz_binning))
    m1.set_poisson_errors()
    # or no errors at all
    m2 = Map(name='y', hist=2*np.ones((n_ebins, n_czbins)),
             binning=(e_binning, cz_binning))

    testdir = tempfile.mkdtemp()
    try:
        for m in [m1, m2, m1+m2, m1-m2, m1/m2, m1*m2]:
            m_file = os.path.join(testdir, m.name + '.json')
            m.to_json(m_file)
            m_ = Map.from_json(m_file)
            assert m_ == m, 'm=\n%s\nm_=\n%s' %(m, m_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    print m1, m1.binning
    print m2, m2.binning
    print m1.nominal_values
    print m1.std_devs
    r = m1 + m2
    # compare only nominal val
    assert r == 3
    print r
    print 'm1+m2=3:', r, r[0,0]
    r = m2 + m1
    # or compare including errors
    assert r == ufloat(3,1)
    print 'm2+m1=3:', r, r[0,0]
    r = 2*m1
    assert r == ufloat(2,2)
    print '2*m1=2:', r, r[0,0]
    r = (2*m1 + 8) / m2
    assert r == ufloat(5,1)
    print '(2*m1 + 8) / m2=5:', r, r.hist[0,0]
    #r[:,1] = 1
    #r[2,:] = 2
    print 'r[0:2,0:5].hist:', r[0:2,0:5].hist
    print 'r[0:2,0:5].binning:', r[0:2,0:5].binning
    r = m1 / m2
    assert r == ufloat(0.5,0.5)
    print r, '=', r[0,0]
    print [b.binning.energy.midpoints[0] for b in m1.iterbins()][0:2]


# TODO: add tests for llh, chi2 methods
# TODO: make tests use assert rather than rely on printouts!
def test_MapSet():
    import os
    import shutil
    import tempfile
    import pint; ureg = pint.UnitRegistry()

    n_ebins = 5
    n_czbins = 3
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1,80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1,0), is_lin=True)
    binning = MultiDimBinning([e_binning, cz_binning])
    m1 = Map(name='ones', hist=np.ones(binning.shape), binning=binning)
    m1.set_poisson_errors()
    m2 = Map(name='twos', hist=2*np.ones(binning.shape), binning=binning)
    ms01 = MapSet((m1, m2))
    ms01 = MapSet((m1, m2), name='ms01')
    ms02 = MapSet((m1, m2), name='map set 1')
    ms1 = MapSet(maps=(m1, m2), name='map set 1', collate_by_name=True)
    assert ms1.name == 'map set 1'
    ms1.hash

    assert ms1.maps == (m1, m2)
    assert ms1.names == ('ones', 'twos')
    assert ms1.tex == r'{\rm map set 1}'
    # Check the Poisson errors
    assert np.all(unp.nominal_values(ms1[0].hist) == np.ones(binning.shape))
    assert np.all(unp.std_devs(ms1[0].hist) == np.ones(binning.shape))
    assert np.all(ms1[1].hist == 2*np.ones(binning.shape))
    print 'ms1[0:2].hist:', ms1[0:2].hist
    print 'ms1[0:2,0:2].hist:', ms1[0:2,0:2].hist
    assert np.all(ms1.apply_to_maps('__add__', 1).ones == 2)

    m1 = Map(name='threes', hist=3*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='fours', hist=4*np.ones((n_ebins,n_czbins)), binning=binning)
    ms2 = MapSet(maps=(m1, m2), name='map set 2', collate_by_name=False)

    try:
        print ms1.__add__(ms2)
    except ValueError:
        pass
    else:
        raise Exception('Should have errored out!')

    m1 = Map(name='fives', hist=5*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='sixes', hist=6*np.ones((n_ebins,n_czbins)), binning=binning)
    ms3 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    ms4 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    assert ms3 == ms4

    print 'ms2.maps:', ms2.maps
    print "(ms2 + ms3).names", (ms2 + ms3).names
    print "(ms2 + ms3)[0,0].hist", (ms2 + ms3)[0,0].hist
    print "ms1['ones'][0,0]:", ms1['ones'][0,0]
    print 'ms1.__mul__(2)[0,0]:', ms1.__mul__(2)[0,0]
    print '(ms1 * 2)[0,0]:', (ms1 * 2)[0,0]
    print 'ms1.__add__(ms1)[0,0]:', ms1.__add__(ms1)[0,0]
    print '(ms1 + ms1)[0,0]:', (ms1 + ms1)[0,0]
    print ms1.names
    print '(ms1/ ms1)[0,0]:', (ms1 / ms1)[0,0]
    print '(ms1/ms1 - 1)[0,0]:', (ms1/ms1 - 1)[0,0]
    #print "ms1.log10()['ones']:", ms1.log10()['ones']
    #print "ms1.log10()[0,0]['ones']:", ms1.log10()[0,0]['ones']
    #print 'np.log10(ms1):', np.log10(ms1)
    print '(ms1 * np.e).binning:', (ms1 * np.e).binning
    #print 'np.log(ms1 * np.e)[0][0,0]:', (np.log(ms1 * np.e))[0][0,0]
    #print 'np.sqrt(ms1)[0][0:4,0:2].hist:', np.sqrt(ms1)[0][0:4,0:2].hist
    print 'str(ms1)', str(ms1)
    print 'str(ms4)', str(ms4)
    print 'ms3', ms3
    print 'ms4', ms4

    testdir = tempfile.mkdtemp()
    try:
        for ms in [ms01, ms02, ms1, ms2, ms3, ms4]:
            ms_file = os.path.join(testdir, ms.name + '.json')
            ms.to_json(ms_file)
            ms_ = MapSet.from_json(ms_file)
            assert ms_ == ms, 'ms=\n%s\nms_=\n%s' %(ms, ms_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)


if __name__ == "__main__":
    test_Map()
    test_MapSet()
