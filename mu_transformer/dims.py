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
