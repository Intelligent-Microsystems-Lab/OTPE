import jax
import jax.numpy as jnp
from typing import Any, Callable
import flax.linen as nn
from jax.tree_util import tree_map
from jax.tree_util import Partial
from flax.core import freeze
import spiking_learning as sl


#######################################################################################################################################
#
#
#                                               Reimplementation of OSTL and OTTT
#
#
#######################################################################################################################################

class OSTL(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):

        def f(snn,carry,inputs):
            if carry['u'].size == 0:
                carry['u'] = jnp.zeros(self.output_sz,self.dtype)
                
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['E']) == 0:
                carry['E'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                ### for OTTT ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
            return carry,s
        
        ########################################################################
        # Because each weight belongs to only one neuron, the gradient
        # calculated with VJP on the sum returns a dense n^2 matrix equivalent
        # to the sparse n^3 matrix calculated with RTRL
        ########################################################################

        def summed_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def summed_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            ds_dtheta_cur,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(summed_carry,snn,carry['u'],inputs,has_aux=True)
            du_cur_dtheta,du_cur_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: ds_du_prev*x + y
            map_u = lambda x,y: du_cur_du_prev*x + y

            ds_dtheta = tree_map(map_s, carry['E'], ds_dtheta_cur)
            carry['E'] = tree_map(map_u, carry['E'], du_cur_dtheta)
            sig_tau = nn.sigmoid(self.init_tau)

            kernel = jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])

            return (carry,s),(ds_dtheta,ds_du_prev,sig_tau,kernel)

        def f_bwd(res,g):
            ds_dtheta,ds_du_prev,sig_tau,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,ds_dtheta)
            g_to_send = (g[1]*(ds_du_prev/sig_tau)).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)
################################

class OTTT(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):

        def f(snn,carry,inputs):
            carry['u'],s = snn(carry['u'],inputs)
            return carry,s

        def summed_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            _,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            sig_tau = nn.sigmoid(self.init_tau)
            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs

            p = jax.lax.stop_gradient({'params':snn.variables['params']})

            return (carry,s),(carry['a_hat'],p,ds_du_prev,sig_tau)

        def f_bwd(res,g):
            a_hat,p,ds_du_prev,sig_tau = res

            p_fu = Partial(fast_update,g[1].flatten()*(ds_du_prev/sig_tau).flatten(),a_hat)
            g_rec_params = tree_map(p_fu,p)
            g_to_send = (g[1]*(ds_du_prev/sig_tau)).dot(p['params']['cf']['kernel'].T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

#-------------------------------------------------------------------------------------------------------------------------------------#

#######################################################################################################################################
#
#
#                                                   OTPE and Approximate OTPE
#
#
#######################################################################################################################################

class OTPE(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):
        def f(snn,carry,inputs):
            if carry['u'].size == 0:
                carry['u'] = jnp.zeros(self.output_sz,self.dtype)

            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['E']) == 0:
                carry['g_bar'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['z_hat'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['E'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['R_hat'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def summed_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def summed_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            ds_dtheta_cur,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(summed_carry,snn,carry['u'],inputs,has_aux=True)
            du_cur_dtheta_cur,du_cur_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: ds_du_prev*x + y
            map_u = lambda x,y: du_cur_du_prev*x + y
            

            ds_dtheta = tree_map(map_s, carry['E'], ds_dtheta_cur)

            carry['E'] = tree_map(map_u, carry['E'], du_cur_dtheta_cur)

            sig_tau = nn.sigmoid(self.init_tau)

            map_r = lambda x,y: sig_tau*x + y

            carry['R_hat'] = tree_map(map_r,carry['R_hat'],ds_dtheta)

            #################################################################
            # "ratio" manages the ratio between current and previous effects
            # in g_bar. We get the effect of the current membrane potential
            # before spiking on the output spikes with ds_du/sig_tau
            #################################################################

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['g_bar'] = ratio*carry['g_bar'] + (1-ratio)*(ds_du_prev/sig_tau)

            kernel = jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])

            return (carry,s),(carry['R_hat'],carry['g_bar'],kernel)

        def f_bwd(res,g):
            R_hat,g_bar,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,R_hat)
            g_to_send = (g[1]*g_bar).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['E'] = jnp.zeros(0,dtype)
        carry['g_bar'] = jnp.zeros(0,dtype)
        carry['R_hat'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['z_hat'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['E'] = jnp.zeros(2,dtype)
        return carry

################################

class Approx_OTPE(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):

        def f(snn,carry,inputs):
            carry['u'],s = snn(carry['u'],inputs)
            return carry,s

        def summed_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            _,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))


            sig_tau = nn.sigmoid(self.init_tau)

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])

            carry['g_bar'] = ratio*carry['g_bar'] + (1-ratio)*(ds_du_prev/sig_tau)

            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs
            carry['z_hat'] = sig_tau*carry['z_hat'] + carry['a_hat']

            p = jax.lax.stop_gradient({'params':snn.variables['params']})

            return (carry,s),(carry['z_hat'],p,carry['g_bar'])

        def f_bwd(res,g):
            a_hat,p,J_u_x = res

            p_fu = Partial(fast_update,g[1].flatten()*J_u_x.flatten(),a_hat)
            g_rec_params = tree_map(p_fu,p)
            g_to_send = (g[1]*J_u_x).dot(p['params']['cf']['kernel'].T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

#-------------------------------------------------------------------------------------------------------------------------------------#

#######################################################################################################################################
#
#
#                                      Modification of OTPE and Approximate OTPE for the output layer
#
#
#######################################################################################################################################

class OTPE_front(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):
        def f(snn,carry,inputs):
            if carry['u'].size == 0:
                carry['u'] = jnp.zeros(self.output_sz,self.dtype)

            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['E']) == 0:
                carry['g_bar'] = jnp.zeros(self.output_sz,self.dtype)
                carry['out'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['z_hat'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['E'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['R_hat'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def summed_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def summed_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            ds_dtheta_cur,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(summed_carry,snn,carry['u'],inputs,has_aux=True)
            du_cur_dtheta_cur,du_cur_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: ds_du_prev*x + y
            map_u = lambda x,y: du_cur_du_prev*x + y
            
            ds_dtheta = tree_map(map_s, carry['E'], ds_dtheta_cur)

            carry['E'] = tree_map(map_u, carry['E'], du_cur_dtheta_cur)

            sig_tau = nn.sigmoid(self.init_tau)

            map_r = lambda x,y: sig_tau*x + y

            carry['R_hat'] = tree_map(map_r,carry['R_hat'],ds_dtheta)

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['g_bar'] = ratio*carry['g_bar'] + (1-ratio)*(ds_du_prev/sig_tau)

            carry['out'] = carry['out']*sig_tau + s

            kernel = jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])
            
            return (carry,carry['out']),(carry['R_hat'],carry['g_bar'],kernel)

        def f_bwd(res,g):
            R_hat,g_bar,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,R_hat)
            g_to_send = (g[1]*g_bar).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['E'] = jnp.zeros(0,dtype)
        carry['g_bar'] = jnp.zeros(0,dtype)
        carry['R_hat'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['z_hat'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['E'] = jnp.zeros(2,dtype)
        return carry

################################

class Approx_OTPE_front(nn.Module):
    connection_fn: Any
    output_sz: int
    neural_dynamics: Any
    init_tau: float
    spike_fn: Callable
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,inputs):

        def f(snn,carry,inputs):
            carry['u'],s = snn(carry['u'],inputs)
            return carry,s

        def summed_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(summed_spike,snn,carry['u'],inputs,has_aux=True)
            _,ds_du_prev,_ = bwd(jnp.ones(z.shape,self.dtype))


            sig_tau = nn.sigmoid(self.init_tau)

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['g_bar'] = ratio*carry['g_bar'] + (1-ratio)*(ds_du_prev/sig_tau)

            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs
            carry['z_hat'] = sig_tau*carry['z_hat'] + carry['a_hat']
            carry['out'] = sig_tau*carry['out'] + s

            p = jax.lax.stop_gradient({'params':snn.variables['params']})

            return (carry,carry['out']),(carry['z_hat'],p,carry['g_bar'])

        def f_bwd(res,g):
            z_hat,p,g_bar = res

            p_fu = Partial(fast_update,g[1].flatten()*g_bar.flatten(),z_hat)
            g_rec_params = tree_map(p_fu,p)
            g_to_send = (g[1]*g_bar).dot(p['params']['cf']['kernel'].T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)