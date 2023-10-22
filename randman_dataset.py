import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from randman.jax_randman import JaxRandman as Randman

def standardize(x,eps=1e-7):
    mi = x.min(0)
    ma = x.max(0)
    return (x-mi)/(ma-mi+eps)

@Partial(jax.jit,static_argnames=('nb_classes','nb_units','nb_steps','dim_manifold','nb_spikes','nb_samples','alpha','shuffle','time_encode','dtype'))
def make_spiking_dataset(nb_classes=10, nb_units=100, nb_steps=100, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True, time_encode=True, seed=None, seed2=None, dtype=jnp.float32):
    """ Generates event-based generalized spiking randman classification/regression dataset. 
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work. 
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args: 
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)
    """
  
    data = []
    labels = []
    targets = []

    uniform_key, shuffle_key, sample_key = jax.random.split(seed2,3)
    
    max_value = jnp.iinfo(jnp.int32).max
    randman_seeds = jax.random.randint(seed, shape=(nb_classes,nb_spikes),maxval=max_value,minval=0)

    for k in range(nb_classes):
        x = jax.random.uniform(uniform_key,(nb_samples,dim_manifold))#*0.95 + 0.025
        submans = [ Randman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k,i]) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(jnp.repeat(jnp.arange(nb_units).reshape(1,-1),nb_samples,axis=0))
            times.append(y)

        units = jnp.concatenate(units,axis=1)
        events = jnp.concatenate(times,axis=1)

        data.append(events)
        labels.append(k*jnp.ones(len(units)))
        targets.append(x)

    data = jnp.concatenate(data, axis=0)
    labels = jnp.array(jnp.concatenate(labels, axis=0), dtype=jnp.int32)
    targets = jnp.concatenate(targets, axis=0)

    idx = jnp.arange(len(data))
    if shuffle:
        #idx = jax.random.shuffle(shuffle_key,idx)
        idx = jax.random.permutation(shuffle_key,idx,independent=True)
    data = data[idx]
    labels = labels[idx]


    if time_encode:
        points = jnp.tile(jnp.int32(jnp.floor(data*nb_steps)),(nb_steps,1,1))
        vals = jnp.tile(jnp.arange(nb_steps),(nb_classes*nb_samples,nb_units,1)).transpose(2,0,1)
        data = jnp.where(vals==points,1,0)
        labels = jnp.tile(jax.nn.one_hot(labels,nb_classes),(nb_steps,1,1))


    else:
        points = jnp.tile(jnp.int32(jnp.floor(data*nb_steps)),(nb_steps,1,1))
        vals = jnp.tile(jnp.arange(nb_steps),(nb_classes*nb_samples,nb_units,1)).transpose(2,0,1)
        data = jnp.where(vals<=points,1,0)
        #data = jax.random.shuffle(sample_key,data,axis=0)
        data = jax.random.permutation(sample_key,data,axis=0,independent=True)
        labels = jnp.tile(jax.nn.one_hot(labels,nb_classes),(nb_steps,1,1))


    return dtype(data), dtype(labels)