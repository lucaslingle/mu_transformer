# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import numpy as np


def get_namedsharding(*, axis_names, device_mesh):
    return jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec(*axis_names)
    )


def sharding_constraint(x, axis_names, device_mesh, enabled):
    if not enabled:
        return x
    ns = get_namedsharding(axis_names=axis_names, device_mesh=device_mesh)
    return jax.lax.with_sharding_constraint(x, ns)


def to_global_array(array, global_mesh):
    # based on
    # https://github.com/google/maxtext/blob/main/MaxText/multihost_dataloading.py#L47
    assert array.ndim == 2
    assert global_mesh.axis_names == ("data", "model")
    # first temporarily put the arrays on each local device, they may get moved later
    arrays = jax.device_put(
        np.split(array, len(global_mesh.local_devices), axis=0),
        global_mesh.local_devices,
    )
    # now use jax.make_array_from_single_device_arrays
    global_shape = (jax.process_count() * array.shape[0], array.shape[1])
    sharding = get_namedsharding(
        axis_names=(global_mesh.axis_names,), device_mesh=global_mesh
    )
    arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    return arr
