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
import torch


def coord_check_l1(act):
    # https://github.com/microsoft/mup?tab=readme-ov-file#coord-check
    # note this is not actually an l1 norm, since we avg over coordinates instead of sum
    stat = torch.detach(torch.mean(torch.abs(act)))
    return stat


class Intermediates:
    def __init__(self):
        self.kvs = dict()

    def set(self, key, value, layer):
        self.kvs[f"{key}_{layer:02}"] = value

    def get(self, key):
        return self.kvs[key]

    def items(self):
        return self.kvs.items()
