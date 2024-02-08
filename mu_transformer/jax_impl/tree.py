from flax import traverse_util


def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""

    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask
