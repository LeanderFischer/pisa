# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import math

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE

# Note that the filename as well as the class name should be all lower case
# (the convention in Python is for class names to be UpperCamelCase).

class select_santa(PiStage):
    """
    Select MS or SS SANTA fit with the corresponding LEERA energy.

    Parameters
    ----------
    data
    params
        ms_chi2 : maximum reduced chi-square for MS fits
        ss_chi2 : maximum reduced chi-square for SS fits
    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    """

    # this is the constructor with default arguments
    
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        # here we register our expected parameters foo and bar so that PISA knows what to expect
        expected_params = ('ms_cut', 'ss_cut')

        # any in-/output names could be specified here, but we won't need that for now
        input_names = ()
        output_names = ()

        # register any variables that are used as inputs or new variables generated
        # (this may seem a bit abstract right now, but hopefully will become more clear later)

        # These keys would be the ones that are converted into binned mode if the calc_mode 
        # is binned. There are no keys for which binning makes sense here.
        input_apply_keys = ()
        # We simply store the calculation output 
        output_calc_keys = ('selected_coszen',
                            'selected_energy',
                            'selected_pid')
        # When the calculation is applied, the old values are simply replaced with the selected 
        # ones. 
        output_apply_keys = ('reco_coszen',
                             'reco_energy',
                             'pid')

        # init base class
        super(select_santa, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_apply_keys=input_apply_keys,
                                       output_apply_keys=output_apply_keys,
                                       output_calc_keys=output_calc_keys,
                                       )

        # this module only makes sense when computed event by event
        assert self.input_mode is not None
        assert self.calc_mode == 'events'
        assert self.output_mode is not None

    def setup_function(self):
        """Setup the stage"""
        # in case we need to initialize sth, like reading in an external file,
        # or add variables to the data object that we can later populate
        
        # do that in the right representation
        self.data.data_specs = self.calc_specs
        for container in self.data:
            # also notice that PISA uses strict typing for arrays
            container['selected_coszen'] = np.empty((container.size), dtype=FTYPE)
            container['selected_energy'] = np.empty((container.size), dtype=FTYPE)
            container['selected_pid'] = np.empty((container.size), dtype=FTYPE)

    def compute_function(self):
        """Perform computation"""
        # this function is called when parameters of this stage are changed (and the first time the
        # pipeline is run). Otherwise it is skipped.
        
        # reduced chi-square has no units.
        ms_cut = self.params.ms_cut.m_as('dimensionless')
        ss_cut = self.params.ss_cut.m_as('dimensionless')
        
        for container in self.data:
            # the `.get(WHERE)` statements are necessary for numba to know if these arrays should be read from the host (CPU)
            # or the device (GPU).
            # No worries, this will work without a GPU too
            select_fit(ms_cut,
                       ss_cut,
                       container['ms_trk_chi2dof'],
                       container['ss_trk_chi2dof'],
                       container['ms_coszen'],
                       container['ss_coszen'],
                       0.,
                       out=container['selected_coszen'].get(WHERE))
            container['selected_coszen'].mark_changed(WHERE)
            select_fit(ms_cut,
                       ss_cut,
                       container['ms_trk_chi2dof'],
                       container['ss_trk_chi2dof'],
                       container['ms_energy'],
                       container['ss_energy'],
                       0.,
                       out=container['selected_energy'].get(WHERE))
            container['selected_energy'].mark_changed(WHERE)
            select_fit(ms_cut,
                       ss_cut,
                       container['ms_trk_chi2dof'],
                       container['ss_trk_chi2dof'],
                       container['ms_pid'],
                       container['ss_pid'],
                       -1.,
                       out=container['selected_pid'].get(WHERE))
            container['selected_pid'].mark_changed(WHERE)

    def apply_function(self):
        # this function is called everytime the pipeline is run, so here we can just apply our factors
        # that we calculated before to the event weights
        
        for container in self.data:
            # we simply set the reconstruction values to the selected ones.
            vectorizer.set(container['selected_coszen'], out=container['reco_coszen'])
            vectorizer.set(container['selected_energy'], out=container['reco_energy'])
            vectorizer.set(container['selected_pid'], out=container['pid'])

# we will write a vectorized function here that works on both the CPU and the GPU,
# and in single and double precision using Numba's `guvectorize`:
# https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html#numba.guvectorize

# For numba to know what to do, we need to define the data types
# of the arguments. These are called "signatures".

# -> Put the most specific (lowest precision, unsigned, etc.) types before
#    less specific types (otherwise the lower precision types will be upcast
#    and performance will suffer for those types)

# -> Use vector notation for signatures' datatypes, even if values will be
#    scalars (you specify shapes in `layout`)

signatures = [
    '(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:])',
    '(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])'
]

# `layout` dictates the _shape_ of the arguments.
# Here, all are scalars: (), and "->()" indicates the last arg is a scalar output which stores the result.
# See https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html#numba.guvectorize

layout = '(),(),(),(),(),(),()->()'

@guvectorize(signatures, layout, target=TARGET)
def select_fit(ms_cut, ss_cut, ms_chi2dof, ss_chi2dof, ms_result, ss_result, else_value, out):
    """This function selects between MS and SS results depending on cuts.
    
    Parameters
    ----------
    ms_cut : scalar
        maximum reduced chi-square for multi-string fits
    ss_cut : scalar
        maximum reduced chi-square for single-string fits
    ms_chi2dof : scalar
        reduced chi-square of this particular ms fit
    ss_chi2dof : scalar
        reduced chi-square of this particular ss fit 
    ms_result : scalar
        result to be taken in case the ms fit is selected
    ss_result : scalar
        result to be taken in case the ss fit is selected
    else_value : scalar
        value to be put in case no fit is selected
    out : scalar
        Result is stored here.

    """
    # Note that you have to "dereference" scalar arguments as if they are actually arrays
    if ms_chi2dof[0] <= ms_cut[0]:
        out[0] = ms_result[0]
    elif ss_chi2dof[0] <= ss_cut[0]:
        out[0] = ss_result[0]
    else:
        out[0] = else_value[0]