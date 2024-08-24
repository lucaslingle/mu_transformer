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
from mu_transformer.configs.base import compute_ff
from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()

    config.v_type = "linear"
    config.g_type = "conv"
    config.ff_multiple = compute_ff(
        ff_act_glu=config.act_name.endswith("glu"),
        kv_group_sz=config.kv_group_size,
        v_type=config.v_type,
        g_type=config.g_type,
    )

    return config
