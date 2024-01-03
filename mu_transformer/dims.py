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
import typing


class Dimensions:
    def __init__(self, **kvs) -> None:
        for k, v in kvs.items():
            self._set_kv(k, v)

    def _set_kv(self, key: str, value: typing.Any) -> None:
        assert isinstance(key, str)
        assert len(key) == 1
        assert key.isalpha()
        assert not hasattr(self, key)
        setattr(self, key, value)

    def _get_kv(self, key: str) -> typing.Any:
        assert hasattr(self, key)
        return getattr(self, key)

    def __getitem__(self, keys):
        assert isinstance(keys, str)
        return [self._get_kv(k) for k in keys]
