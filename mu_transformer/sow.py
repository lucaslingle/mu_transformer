import jax
import jax.numpy as jnp


def coord_check_l1(act):
    # https://github.com/microsoft/mup?tab=readme-ov-file#coord-check
    stat = jax.lax.stop_gradient(jnp.mean(jnp.abs(act)))
    return stat


def split_coord_checks(name, stat_tensor):
    # for logging. splits the sown stats by layer when using remat_scan or scan(remat),
    # after they've been stacked into a single output tensor
    stats = jnp.split(stat_tensor, stat_tensor.shape[0], axis=0)
    return {f"{name}_{i:02}": stats[i] for i in range(stat_tensor.shape[0])}
