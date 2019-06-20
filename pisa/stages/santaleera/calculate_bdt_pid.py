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

import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier

# Note that the filename as well as the class name should be all lower case
# (the convention in Python is for class names to be UpperCamelCase).


    
class calculate_bdt_pid(PiStage):
    """
    Apply pre-trained BDT to reconstructed variables to calculate new pid

    Parameters
    ----------
    data
    params
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
        expected_params = ()

        # any in-/output names could be specified here, but we won't need that for now
        input_names = ()
        output_names = ()

        # register any variables that are used as inputs or new variables generated
        # (this may seem a bit abstract right now, but hopefully will become more clear later)

        # what are the keys used from the inputs during apply
        input_apply_keys = ()
                            
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('calculated_pid',)
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('pid',)

        # init base class
        super(calculate_bdt_pid, self).__init__(data=data,
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
        
        # Load gbc version:
        bdt_path = os.path.join(os.environ['PISA_RESOURCES'], 'data/oscnext_santa_bdts/default_25_meters/alg.pckl')
        global bdt
        bdt = pickle.load(open(bdt_path,'rb'))

        # do that in the right representation
        self.data.data_specs = self.calc_specs
        for container in self.data:
            # also notice that PISA uses strict typing for arrays
            container['calculated_pid'] = np.empty((container.size), dtype=FTYPE)

    def compute_function(self):
        """Perform computation"""
        # this function is called when parameters of this stage are changed (and the first time the
        # pipeline is run). Otherwise it is skipped.

        for container in self.data:
            # the `.get(WHERE)` statements are necessary for numba to know if these arrays should be read from the host (CPU)
            # or the device (GPU).
            # No worries, this will work without a GPU too
            container['calculated_pid'] = self.calculate_pid(container['pid'].get(WHERE),
                                                             container['leera_pid'].get(WHERE),
                                                             container['track_reco'].get(WHERE),
                                                             container['rho36_start_reco'].get(WHERE),
                                                             container['z_start_reco'].get(WHERE),
                                                             container['rho36_end_reco'].get(WHERE),
                                                             container['z_end_reco'].get(WHERE),
                                                            )

    def apply_function(self):
        # this function is called everytime the pipeline is run, so here we can just apply our factors
        # that we calculated before to the event weights
        
        for container in self.data:
            # set the pid value to the calculated one
            vectorizer.set(container['calculated_pid'], out=container['pid'])

    def calculate_pid(pid, leera_pid, track_reco, rho36_start_reco, z_start_reco, rho36_end_reco, z_end_reco, out):
        """This function uses the pre-trained bdt to apply it to the events features

        Parameters
        ----------
        pid        : scalar
                chi^2 ratio (track/cascade)
        leera_pid        : scalar
                llh difference (track-cascade)
        track_reco       : scalar
                reconstructed tracklength
        rho36_start_reco : scalar
                xy distance to string 36 (starting point)
        rho36_end_reco   : scalar
                xy distance to string 36 (end point)
        z_start_reco     : scalar
                depth (starting point)
        z_end_reco       : scalar
                depth (end point)
        """

        feature_list = [pid[0], leera_pid[0], track_reco[0], rho36_start_reco[0], 
                        z_start_reco[0], rho36_end_reco[0], z_end_reco[0]]
        feature_array = np.ndarray(feature_list).reshape(1,-1)
        out[0] = bdt.predict_proba(feature_array)[:,1]