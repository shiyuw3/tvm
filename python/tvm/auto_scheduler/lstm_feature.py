# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Tuple, Union, Optional
import struct

import numpy as np

from .loop_state import State, StateObject
from . import _ffi_api

_vocabulary = {}
MAX_NUM_TREE = 40
MAX_NUM_CHILDREN = 20
# MAX_DIM_ADDITIONAL = 24
MAX_DIM_ADDITIONAL = 64

def unpack_lstm_feature(byte_arr: bytearray, take_log: Optional[bool] = None, keep_name: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the flatten feature (in byte array format) from c++
    """
    n_tree, offset_child, offset_names, offset_addfea, = \
        struct.unpack('4i', byte_arr[: 4 * 4])

    # unpack byte stream
    child_nums = struct.unpack("%di" % n_tree, byte_arr[offset_child : (offset_child + 4 * n_tree)])
    children = []
    offset_child += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = child_nums[i]
        children.append(struct.unpack("%di" % num,
                                      byte_arr[(offset_child + ct * 4) : (offset_child + (ct + num) * 4)]))
        ct += num

    name_nums = struct.unpack("%di" % n_tree, byte_arr[offset_names : (offset_names + 4 * n_tree)])
    names = []
    offset_names += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = name_nums[i]
        names.append(struct.unpack("%ds" % num,
                                   byte_arr[(offset_names + ct) : (offset_names + ct + num)])[0].decode())
        ct += num

    addfea_nums = struct.unpack("%di" % n_tree, byte_arr[offset_addfea : (offset_addfea + 4 * n_tree)])
    add_feas = []
    offset_addfea += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = addfea_nums[i]
        add_feas.append(struct.unpack("%df" % num,
                                      byte_arr[(offset_addfea + ct * 4) : (offset_addfea + (ct + num) * 4)]))
        ct += num

    # copy children relation to numpy array (with padding)
    # children_np = np.empty((n_tree, MAX_NUM_CHILDREN + 1), dtype=np.int16)
    children_np = np.zeros((MAX_NUM_TREE, MAX_NUM_CHILDREN + 1), dtype=np.int16)
    for i, chi in enumerate(children):
        n_chi = len(chi)
        children_np[i, 0] = n_chi
        children_np[i, 1 : (1 + n_chi)] = chi

    # transform string name to integer index
    # emb_idx_np = np.zeros((n_tree, ), dtype=object if keep_name else np.int16)
    emb_idx_np = np.zeros((MAX_NUM_TREE, ), dtype=object if keep_name else np.int16)
    for i, name in enumerate(names):
        if name not in _vocabulary:
            _vocabulary[name] = len(_vocabulary)
        emb_idx_np[i] = (name if keep_name else _vocabulary[name])

    # copy additional feature to numpy array (with padding)
    # add_feas_np = np.zeros((n_tree, MAX_DIM_ADDITIONAL), dtype=np.float32)
    add_feas_np = np.zeros((MAX_NUM_TREE, MAX_DIM_ADDITIONAL), dtype=np.float32)
    for i, fea in enumerate(add_feas):
        add_feas_np[i, : addfea_nums[i]] = fea

    if take_log:
        log_index = add_feas_np > 0
        add_feas_np[log_index] = np.log2(add_feas_np[log_index] + 1.0)

    return children_np, emb_idx_np, add_feas_np

def get_lstm_feature_from_state(
    task: "SearchTask", state: [Union[State, StateObject]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get LSTM feature from the given search task and state.
    """
    if isinstance(state, State):
        state_object = state.state_object
    elif isinstance(state, StateObject):
        state_object = state
    byte_arr = _ffi_api.GetLSTMFeatureFromState(task, state)
    return unpack_lstm_feature(byte_arr)

