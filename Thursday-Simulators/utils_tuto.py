# -*- coding: utf-8 -*-

"""
@authors Thomas Ayral <thomas.ayral@atos.net>
@copyright 2021 Bull S.A.S.  -  All rights reserved.

    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
"""


import numpy as np
from qat.core.circuit_builder.matrix_util import get_predef_generator
def make_matrix(hamiltonian):
    mat = np.zeros((2**hamiltonian.nbqbits, 2**hamiltonian.nbqbits), np.complex_)
    for term in hamiltonian.terms:
        op_list = ["I"]*hamiltonian.nbqbits
        for op, qb in zip(term.op, term.qbits):
            op_list[qb] = op
        def mat_func(name): return np.identity(2) if name == "I" else get_predef_generator()[name]
        term_mat = mat_func(op_list[0])
        for op in op_list[1:]:
            term_mat = np.kron(term_mat, mat_func(op))
        mat += term.coeff * term_mat
    return mat
