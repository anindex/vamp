from fire import Fire

import numpy as np

import jax
import jax.numpy as jnp

import vamp


for name, target in vamp.jax.registrations().items():
    jax.ffi.register_ffi_target(name, target)


def validate_motion_pairwise(a, b):
    print(a.shape)
    print(b.shape)

    out_type = jax.ShapeDtypeStruct(a.shape, b.dtype)

    call = jax.ffi.ffi_call(
        "validate_motion_pairwise",
        out_type,
        vmap_method = "broadcast_all",
        )

    return call(a, b)


def main(n = 10000):
    jax.config.update('jax_platform_name', 'cpu')

    halton = vamp.panda.halton()
    config_a = np.vstack([halton.next().numpy() for i in range(n)])
    config_b = np.vstack([halton.next().numpy() for i in range(n)])

    jca = jnp.array(config_a)
    jcb = jnp.array(config_b)

    validate_motion_pairwise(jca, jcb)

if __name__ == "__main__":
    Fire(main)
