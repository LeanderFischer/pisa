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


    
class shift_pid(PiStage):
    """
    Select a pid cut. (Default cut has to be 1.0! and then pid variable is shifted accordingly)

    Parameters
    ----------
    data
    params
        pid_cut : cut value to be selected
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

        # here we register our expected parameters so that PISA knows what to expect
        expected_params = ('pid_cut')

        # any in-/output names could be specified here, but we won't need that for now
        input_names = ()
        output_names = ()

        # register any variables that are used as inputs or new variables generated
        # (this may seem a bit abstract right now, but hopefully will become more clear later)

        # what are the keys used from the inputs during apply
        input_apply_keys = ('pid',)
                            
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('shifted_pid',)
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('pid',)

        # init base class
        super(shift_pid, self).__init__(data=data,
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

        # make sure the user specified some modes
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
            container['shift_pid'] = np.empty((container.size), dtype=FTYPE)

    def compute_function(self):
        """Perform computation"""
        # this function is called when parameters of this stage are changed (and the first time the
        # pipeline is run). Otherwise it is skipped.

        # pid_cut has no units.
        pid_cut = self.params.pid_cut.m_as('dimensionless')

        for container in self.data:
            # the `.get(WHERE)` statements are necessary for numba to know if these arrays should be read from the host (CPU)
            # or the device (GPU).
            # No worries, this will work without a GPU too
            shift_pid_function(pid_cut,
                               container['pid'].get(WHERE),
                               out= container['shift_pid'].get(WHERE))
            container['shift_pid'].mark_changed(WHERE)

    def apply_function(self):
        # this function is called everytime the pipeline is run, so here we can just apply our factors
        # that we calculated before to the event weights
        
        for container in self.data:
            # set the pid value to the calculated one
            vectorizer.set(container['shift_pid'], 
                           out=container['pid'])
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
    '(f4[:], f4[:], f4[:])',
    '(f8[:], f8[:], f8[:])'
]

# `layout` dictates the _shape_ of the arguments.
# Here, all are scalars: (), and "->()" indicates the last arg is a scalar output which stores the result.
# See https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html#numba.guvectorize

layout = '(),()->()'

@guvectorize(signatures, layout, target=TARGET)
def shift_pid_function(pid_cut_value, pid, out):
    """This function selects a pid cut by shifting the pid variable so 
    the default cut at 1.0 is at the desired cut position.
    
    Parameters
    ----------
    pid_cut_value : scalar
        desired pid_cut_value
    pid : scalar
        pid variable
    out : scalar
        shifted pid values

    """

    out[0] = (pid[0] + (1. - pid_cut_value[0]))