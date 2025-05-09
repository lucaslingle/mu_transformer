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
from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "dm2048"

    # mesh
    config.n_mesh_rows = 32
    config.n_mesh_cols = 4

    # width
    config.d_model = 2048

    return config
