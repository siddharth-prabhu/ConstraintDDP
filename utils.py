import os
import time
from typing import Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def check_dir(dir : str) -> None :

    if not os.path.exists(dir):
        os.makedirs(dir)


def cpu_time(afunc : Callable):
    # Get CPU time of decorated functions 
    # Time the scan function in jax.lax.scan and sum the times. Dont time jax.lax.scan directly   
    def _cpu_time(*args, **kwargs):
        start = jax.experimental.io_callback(lambda : jnp.array(time.process_time()), jax.ShapeDtypeStruct((), jnp.dtype("float64")), ordered = True)
        sol = jax.block_until_ready(afunc(*args, **kwargs))
        end = jax.experimental.io_callback(lambda : jnp.array(time.process_time()), jax.ShapeDtypeStruct((), jnp.dtype("float64")), ordered = True)
        return sol, end - start
    
    return _cpu_time
