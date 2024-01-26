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
import jax.numpy as jnp


def coord_check_l1(act):
    # https://github.com/microsoft/mup?tab=readme-ov-file#coord-check
    stat = jax.lax.stop_gradient(jnp.mean(jnp.abs(act)))
    return stat


def split_and_name(name, stat_tensor):
    # for logging. splits the sown stats by layer when using remat_scan or scan(remat),
    # after they've been stacked into a single output tensor
    stats = jnp.split(stat_tensor, stat_tensor.shape[0], axis=0)
    return {f"{name}_{i:02}": stats[i] for i in range(stat_tensor.shape[0])}
