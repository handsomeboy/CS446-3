import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as spla
from scipy.interpolate import BSpline, splrep, splder, sproot
import scipy.integrate as spint
import os
import sys
import pandas as pd
import time
import sys
from datetime import datetime
import timeit
from collections import OrderedDict
import potential_class
import bead_class

def calculate_ds(Beads, Potential_list,g_cg, g_aa, r_rdf):
    ds_ = []
    Beads_list = Beads._name_list
    Potentials_type_list = Beads._beads_pot_type
    Den = Beads._density
    AtNum = Beads._atom_num_list
    beta = Beads._beta
    for cnt, bead in enumrate(Beads_list):
        pl = Potential_list[cnt]
        r = r_rdf[cnt]
        rdf_cg = g_cg[cnt]
        rdf_aa = g_aa[cnt]
        ds_.extend([2*np.pi*AtNum[cnt]*Den[cnt]*beta*spint.simps((rdf_cg-rdf_aa)*pl.d_potential(r)[i])*r**2, r) for i in range(pl._num_lambda)])
    ds_ = np.array(ds_)
    print("shape of dlamba_: ", ds_.shape)
    return ds_

def update_B(B, DS_k, DS_kp1, dlambda):
    gamma = DS_kp1 - DS_k
    B += (np.matmul(gamma,gamma.T) / (gamma.T.dot(dlambda))) - \
        (np.matmul(np.matmul(B,np.matmul(dlambda,dlambda.T)),B)/(dlambda.T.dot(B.dot(dlambda))))
    return B

def solve(B, DS):
    if np.all(B == B.T):
        L = spla.cholesky(B,lower=True)
        dlambda_ = - spla.cho_solve((L, True), DS)
        return dlambda_
    else:
        raise ValueError("Not a symmetric B")
