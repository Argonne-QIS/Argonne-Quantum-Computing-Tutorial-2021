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



import random
import numpy as np
from itertools import product
from copy import deepcopy

from qat.comm.datamodel.ttypes import OpType
from qat.comm.exceptions.ttypes import PluginException, ErrorType
from qat.comm.shared.ttypes import ProcessingType
from qat.comm.datamodel.ttypes import Op
from qat.core.plugins import AbstractPlugin
from qat.core.util import extract_syntax
from qat.core import Batch
from qat.core import Result, BatchResult
from qat.core.wrappers.result import Sample

def make_pauli_op(pauli_string, qbits):
    """
    Args:
        pauli_string (str): Pauli string PP..P with P = I, X, Y, Z.
    """
    assert (len(pauli_string)==len(qbits))
    pauli_op = Op(gate=pauli_string,
                  qbits=qbits,
                  type=OpType.GATETYPE)
    return pauli_op

    

class DepolarizingPlugin(AbstractPlugin):
    r"""
    When used to build a stack with a perfect QPU, it is equivalent to a noisy QPU with a depolarizing noise model.

    Here, we define the depolarizing noise model by its action on the density matrix:
    .. math::
        \mathcal{E}(\rho) = (1 - p) \rho + \frac{p}{4^{n_\mathrm{qbits}}-1}\sum_{k = 1}^{4^{n_\mathrm{qbits}}} P_k \rho P_k

    where :math:`\lbrace P_k, k = 0 \dots 4^{n_\mathrm{qbits}} \rbrace` denotes
    the set of all products of Pauli matrices (including the identity) for
    :math:`n_\mathrm{qubits}` qubits. By convention, :math:`P_0 = I_{2^{n_\mathrm{qbits}}}`.

    Args:
        prob_1qb (float, optional): 1-qbit depolarizing probability.
            Defaults to 0.0.
        prob_2qb (float, optional): 2-qbit depolarizing probability.
            Defaults to 0.0.
        n_samples (int, optional): number of stochastic samples.
            Defaults to 1000.
        seed (int, optional): seed for random number generator.
            Defaults to 1425.
        verbose (bool, optional): for verbose output. Defaults to False.
    """
    def __init__(self, prob_1qb=0.0, prob_2qb=0.0, n_samples=1000,
                 seed=1425, verbose=False):
        self.prob_1qb = prob_1qb
        self.prob_2qb = prob_2qb
        self.n_samples = n_samples
        self.seed = seed
        self.verbose = verbose
        self.nbshots = None
        self.nbqbits = None
        self.job_type = None
        
    def compile(self, batch, harware_specs):
        if len(batch.jobs) != 1:
            raise PluginException(code=ErrorType.INVALID_ARGS,
                                  message="This plugin supports only single jobs"
                                  ", got %s instead"%len(batch.jobs))
        job = batch.jobs[0]
        self.nbshots = job.nbshots
        self.nbqbits = len(job.qubits)
        self.job_type = job.type
        list_2qb_paulis = ["%s%s"%(p1, p2)
                           for p1, p2 in product(["I", "X", "Y", "Z"],
                                                 repeat=2) 
                            if p1 != 'I' or p2 !='I']
        
        new_batch = []
        for _ in range(self.n_samples):
            job_copy = deepcopy(job)
            job_copy.nbshots = 0
            job_copy.circuit.ops = []
            for op in job.circuit:
            
                if op.type != OpType.GATETYPE:
                    raise PluginException(code=ErrorType.ILLEGAL_GATES,
                                          message="This plugin supports operators of type GATETYPE,"
                                                  " got %s instead"%op.type)
                if len(op.qbits) > 2:
                    gdef = job_copy.circuit.gateDic[op.gate]
                    gname = extract_syntax(gdef, job_copy.circuit.gateDic)[0]
                    if gname != "STATE_PREPARATION":
                        raise PluginException(code=ErrorType.NBQBITS,
                                              message="This plugin supports only 1 and 2-qbit gates,"
                                                      " got a gate acting on qbits %s instead"%op.qbits)
                    
                job_copy.circuit.ops.append(op)
                if len(op.qbits) == 1:
                    if random.random() < self.prob_1qb:
                        job_copy.circuit.ops.append(make_pauli_op(random.choice(["X", "Y", "Z"]),
                                                                  op.qbits))
                if len(op.qbits) == 2:
                    if random.random() < self.prob_2qb:
                        noise_gate = random.choice(list_2qb_paulis)
                        for gate, qb in zip(noise_gate, op.qbits):
                            if gate != "I":
                                job_copy.circuit.ops.append(make_pauli_op(gate, [qb]))
            new_batch.append(job_copy)
        return Batch(new_batch)
    
    def post_process(self, batch_result):
        if self.job_type == ProcessingType.SAMPLE:
            final_distrib = None
            for result in batch_result.results:
                probs = np.zeros(2**self.nbqbits)
                for sample in result:
                    probs[sample.state.int] = sample.probability
                    
                if final_distrib is None:
                    final_distrib = probs
                else:
                    final_distrib += probs
            final_distrib /= len(batch_result.results)
           
            if self.verbose:
                print("norm final distrib=", np.sum(final_distrib))

            if self.nbshots == 0:
                res = Result()
                # res.has_statevector = True
                # res.statevector = final_distrib
                res.raw_data = []
                for int_state, val in enumerate(final_distrib):
                    sample = Sample(state=int_state,
                                    probability=val)
                    res.raw_data.append(sample)
                
                return BatchResult(results=[res])
            raise Exception("nbshots > 0 not yet implemented")

        elif self.job_type == ProcessingType.OBSERVABLE:
            vals = []
            for result in batch_result.results:
                if self.verbose:
                    print("result=", result)
                vals.append(result.value)
            val = np.mean(vals)
            err = np.std(vals)/np.sqrt(len(vals)-1)
            res = Result(value=val, error=err)
            return BatchResult(results=[res])

        else:
            raise Exception("Unknown job type")
        
