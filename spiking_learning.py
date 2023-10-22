import functools


from typing import Any, Callable, Sequence

import jax
from jax import dtypes
from jax import random
from flax import linen as nn
import jax.numpy as jnp

import numpy as np

from jax._src.nn.initializers import lecun_normal


Array = jnp.ndarray
DType = Any


def uniform(scale=1e-2, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns real uniformly-distributed random
   arrays.
Args:
  scale: optional; the upper and lower bound of the random distribution.
  dtype: optional; the initializer's default dtype.
Returns:
  An initializer that returns arrays whose values are uniformly distributed
  in the range ``[-scale, scale)``.
"""

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape,dtype)*scale
    #return random.uniform(key, shape, dtype) * scale * 2 - scale

  return init


def static_init(val=1.0, dtype: DType = jnp.float_) -> Callable:
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtype) * val

  return init


def normal_shift(
    bias=0, scale=1e-2, no_sign_flip=True, dtype: DType = jnp.float_
) -> Callable:
  """Builds an initializer that returns real uniformly-distributed random
   arrays.
Args:
  scale: optional; the upper and lower bound of the random distribution.
  dtype: optional; the initializer's default dtype.
Returns:
  An initializer that returns arrays whose values are uniformly distributed
  in the range ``[-scale, scale)``.
"""

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)

    x = random.normal(key, shape, dtype) * scale + bias
    if no_sign_flip:
      x = jnp.abs(x)
    return x

  return init

def fs(slope):

    @jax.custom_vjp
    def fast_sigmoid(x):
      # if not dtype float grad ops wont work
      return jnp.array(x >= 0.0, dtype=x.dtype)
    
    
    def fast_sigmoid_fwd(x):
      return fast_sigmoid(x), x
    
    
    def fast_sigmoid_bwd(res, g):
      x = res
      alpha = slope
    
      scale = 1 / (alpha * jnp.abs(x) + 1.0) ** 2
      return (g * scale,)
    
    
    fast_sigmoid.defvjp(fast_sigmoid_fwd, fast_sigmoid_bwd)
    return fast_sigmoid
  
class subtraction_LIF(nn.Module):
  init_tau: float
  spike_fn: Callable
  v_threshold: float = 1.0
  v_reset: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, u, s_in):
    #tau = self.param("tau", uniform(self.init_tau,dtype=self.dtype), (u.shape[-1],))
    v_threshold = jnp.array([self.v_threshold], dtype=self.dtype)

    u = u * jax.nn.sigmoid(self.init_tau) + s_in
    

    s = self.spike_fn(u - v_threshold)
    u = u - s*v_threshold

    return u, s

class SpikingBlock(nn.Module):
  connection_fn: Callable
  neural_dynamics: Callable
  norm_fn: Callable = None

  # @nn.remat
  # @functools.partial(
  #     nn.transforms.scan,
  #     variable_broadcast="params",
  #     variable_carry="batch_stats",
  #     split_rngs={"params": False},
  # )
  @nn.compact
  def __call__(self, u, inputs):
    x = self.connection_fn(inputs)

    if self.norm_fn:
      x = self.norm_fn(x)

    u, s = self.neural_dynamics(u, x)

    return u, s

  @staticmethod
  def initialize_carry(
      inputs, connection_fn, norm_fn=None, dtype=jnp.float32
  ):
    x = connection_fn(inputs[:, :])
    if norm_fn:
      x = norm_fn(x)

    return jnp.zeros_like(x, dtype=dtype)

class SpikingBlockMod(nn.Module):
  connection_fn: Any
  output_sz: int
  neural_dynamics: Any
  init_tau: float
  spike_fn: Callable
  v_threshold: float = 1.0
  v_reset: float = 0.0
  dtype: Any = jnp.float32

  def setup(self):
    self.cf = self.connection_fn(self.output_sz,use_bias=True)
    self.nd = self.neural_dynamics(init_tau=self.init_tau,spike_fn=self.spike_fn)

  @nn.compact
  def __call__(self, u, inputs):
    x = self.cf(inputs)
    u, s = self.nd(u, x)

    return u, s

  @staticmethod
  def initialize_carry(
      inputs, connection_fn, norm_fn=None, dtype=jnp.float32
  ):
    x = connection_fn(inputs[:, :])
    if norm_fn:
      x = norm_fn(x)

    return jnp.zeros_like(x, dtype=dtype)

