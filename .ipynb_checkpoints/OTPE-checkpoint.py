import jax
import jax.numpy as jnp
from typing import Any, Callable
import flax.linen as nn
from jax.tree_util import tree_map
from jax.tree_util import Partial
from flax.core import freeze
import spiking_learning as sl

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
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                # for OTTT #
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                ############
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['J_u_params']) == 0:
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)
            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)
            sig_tau = nn.sigmoid(self.init_tau)#nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau']))
            grad_s_x = grad_s_u/sig_tau

            return (carry,s),(J_s_params,grad_s_x,jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            J_s_params,grad_s_x,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            g_to_send = (g[1]*grad_s_x).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        return carry
######################################
class OSTL_pass(nn.Module):
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
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                # for OTTT #
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['u_pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
                ############
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['J_u_params']) == 0:
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)
            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)
            sig_tau = nn.sigmoid(self.init_tau)
            grad_s_x = grad_s_u/sig_tau

            kernel = jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])

            s_pass = grad_s_x*kernel + grad_s_u*carry['u_pass']
            carry['u_pass'] = grad_u_u*carry['u_pass'] + (grad_u_u/sig_tau)*kernel

            return (carry,s),(J_s_params,s_pass)

        def f_bwd(res,g):
            J_s_params,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            g_to_send = g[1].dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry
######################################
class OTPE_pass(nn.Module):
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
            if len(carry['J_u_params']) == 0:
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['r2'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
                carry['u_pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
                
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat(snn,params,carry,x):
            u,s = snn.apply(params,carry,x)
            return (jnp.sum(u),jnp.sum(s)),(u,s)

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y
            

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)

            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)

            sig_tau = nn.sigmoid(self.init_tau)

            grad_s_x = grad_s_u/sig_tau

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x

            map_r = lambda x,y: ratio*x + (1-ratio)*y

            carry['r2'] = tree_map(map_r,carry['r2'],J_s_params)

            kernel = jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])

            s_pass = grad_s_x*kernel + grad_s_u*carry['u_pass']
            carry['u_pass'] = grad_u_u*carry['u_pass'] + (grad_u_u/sig_tau)*kernel

            carry['pass'] = sig_tau*carry['pass'] + s_pass
            
            return (carry,s),(carry['r2'],carry['J_u_x'],carry['pass'])

        def f_bwd(res,g):
            J_s_params,grad_s_x,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            #g_to_send = (g[1]*grad_s_x).dot(kernel.T)
            #g_to_send = tree_map(lambda x,y: jnp.sum(x*y,axis=1),g_rec_params['params']['cf']['kernel'],kernel)
            g_to_send = g[1].dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        carry['r2'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['a_hat2'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry

##########################################
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
            if len(carry['J_u_params']) == 0:
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['r2'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                #carry['pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat(snn,params,carry,x):
            u,s = snn.apply(params,carry,x)
            return (jnp.sum(u),jnp.sum(s)),(u,s)

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y
            

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)

            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)

            sig_tau = nn.sigmoid(self.init_tau)#nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau']))

            grad_s_x = grad_s_u/sig_tau

            map_r = lambda x,y: sig_tau*x + y

            carry['r2'] = tree_map(map_r,carry['r2'],J_s_params)

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x

            #carry['pass'] = ratio*carry['pass'] + (1-ratio)*grad_s_x*jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])
            
            return (carry,s),(carry['r2'],carry['J_u_x'],jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            J_s_params,grad_s_x,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            g_to_send = (g[1]*grad_s_x).dot(kernel.T)
            #g_to_send = tree_map(lambda x,y: jnp.sum(x*y,axis=1),g_rec_params['params']['cf']['kernel'],kernel)
            #g_to_send = g[1].dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        carry['r2'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['a_hat2'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry
##########################################
class OTPE_flat(nn.Module):
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
            if len(carry['J_u_params']) == 0:
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['r2'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                #carry['pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat(snn,params,carry,x):
            u,s = snn.apply(params,carry,x)
            return (jnp.sum(u),jnp.sum(s)),(u,s)

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y
            

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)

            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)

            sig_tau = nn.sigmoid(self.init_tau)#nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau']))

            grad_s_x = grad_s_u/sig_tau

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = ratio/carry['ratio']
            carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x

            #carry['pass'] = ratio*carry['pass'] + (1-ratio)*grad_s_x*jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])
            map_r = lambda x,y: ratio*x + (1-ratio)*y

            carry['r2'] = tree_map(map_r,carry['r2'],J_s_params)
            
            return (carry,s),(carry['r2'],carry['J_u_x'],jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            J_s_params,grad_s_x,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            g_to_send = (g[1]*grad_s_x).dot(kernel.T)
            #g_to_send = tree_map(lambda x,y: jnp.sum(x*y,axis=1),g_rec_params['params']['cf']['kernel'],kernel)
            #g_to_send = g[1].dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        carry['r2'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['a_hat2'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry
##########################################
class OTPE_mod2(nn.Module):
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
            if len(carry['J_u_params']) == 0:
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                ### for Approx_OTPE ###
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
                #######################
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                carry['r2'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
                #carry['pass'] = jnp.zeros((inputs.size,self.output_sz),self.dtype)
            return carry,s

        def flat_spike(model,carry,x):
            _,s = model(carry,x)
            return jnp.sum(s),s

        def flat(snn,params,carry,x):
            u,s = snn.apply(params,carry,x)
            return (jnp.sum(u),jnp.sum(s)),(u,s)

        def flat_carry(model,carry,x):
            u,_ = model(carry,x)
            return jnp.sum(u),u

        def f_fwd(snn,carry,inputs):

            z,bwd,s = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            z,bwd,carry['u'] = nn.vjp(flat_carry,snn,carry['u'],inputs,has_aux=True)
            grad_u_params,grad_u_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            map_s = lambda x,y: grad_s_u*x + y
            map_u = lambda x,y: grad_u_u*x + y
            

            J_s_params = tree_map(map_s, carry['J_u_params'], grad_s_params)

            carry['J_u_params'] = tree_map(map_u, carry['J_u_params'], grad_u_params)

            sig_tau = nn.sigmoid(self.init_tau)#nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau']))

            grad_s_x = grad_s_u/sig_tau

            map_r = lambda x,y: sig_tau*x + y

            carry['r2'] = tree_map(map_r,carry['r2'],J_s_params)

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['J_u_x'] = sig_tau*carry['J_u_x'] + grad_s_x

            #carry['pass'] = ratio*carry['pass'] + (1-ratio)*grad_s_x*jax.lax.stop_gradient(snn.variables['params']['cf']['kernel'])
            
            return (carry,s),(carry['r2'],carry['J_u_x'],jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            J_s_params,grad_s_x,kernel = res

            g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,J_s_params)
            g_to_send = (g[1]*grad_s_x).dot(kernel.T)
            #g_to_send = tree_map(lambda x,y: jnp.sum(x*y,axis=1),g_rec_params['params']['cf']['kernel'],kernel)
            #g_to_send = g[1].dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(0,dtype)
        carry['r2'] = jnp.zeros(0,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)

        carry['a_hat'] = jnp.zeros(0,dtype)
        carry['a_hat2'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry
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
            if carry['u'].size == 0:
                carry['u'] = jnp.zeros(self.output_sz,self.dtype)
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
            carry['u'],s = snn(carry['u'],inputs)
            return carry,s

        def flat_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))

            sig_tau = nn.sigmoid(self.init_tau)#jnp.mean(nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau'])))
            grad_s_x = grad_s_u/sig_tau
            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs

            return (carry,s),(carry['a_hat'],grad_s_params,grad_s_x,jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            a_hat,J_u_params,grad_s_x,kernel = res

            p_fu = Partial(fast_update,g[1].flatten()*grad_s_x.flatten(),a_hat)
            g_rec_params = tree_map(p_fu,J_u_params)
            g_to_send = (g[1]*grad_s_x).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['a_hat'] = jnp.zeros(0,dtype)

        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
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
            if carry['u'].size == 0:
                carry['u'] = jnp.zeros(self.output_sz,self.dtype)
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['J_u_params']) == 0:
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def flat_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))


            sig_tau = nn.sigmoid(self.init_tau)#jnp.mean(nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau'])))
            grad_s_x = grad_s_u/sig_tau

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x

            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs
            carry['a_hat2'] = sig_tau*carry['a_hat2'] + carry['a_hat']

            return (carry,s),(carry['a_hat2'],grad_s_params,carry['J_u_x'],grad_s_x,jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            a_hat,J_u_params,J_u_x,grad_s_x,kernel = res

            p_fu = Partial(fast_update,g[1].flatten()*J_u_x.flatten(),a_hat)
            g_rec_params = tree_map(p_fu,J_u_params)
            g_to_send = (g[1]*J_u_x).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(1,dtype)
        carry['a_hat'] = jnp.zeros(1,dtype)
        carry['a_hat2'] = jnp.zeros(1,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)
        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry
################################
class Approx_OTPE_flat(nn.Module):
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
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['J_u_params']) == 0:
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def flat_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))


            sig_tau = nn.sigmoid(self.init_tau)#jnp.mean(nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau'])))
            grad_s_x = grad_s_u/sig_tau

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            #carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x
            carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x

            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs
            carry['a_hat2'] = ratio*carry['a_hat2'] + (1-ratio)*carry['a_hat']

            return (carry,s),(carry['a_hat2'],grad_s_params,carry['J_u_x'],grad_s_x,jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            a_hat,J_u_params,J_u_x,grad_s_x,kernel = res

            p_fu = Partial(fast_update,g[1].flatten()*J_u_x.flatten(),a_hat)
            g_rec_params = tree_map(p_fu,J_u_params)
            g_to_send = (g[1]*J_u_x).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(1,dtype)
        carry['a_hat'] = jnp.zeros(1,dtype)
        carry['a_hat2'] = jnp.zeros(1,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)
        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry

class Approx_OTPE_mod2(nn.Module):
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
                carry['J_u_x'] = jnp.zeros(self.output_sz,self.dtype)
                carry['a_hat'] = jnp.zeros(inputs.size,self.dtype)
                carry['a_hat2'] = jnp.zeros(inputs.size,self.dtype)
            carry['u'],s = snn(carry['u'],inputs)
            if len(carry['J_u_params']) == 0:
                carry['J_u_params'] = {'params':freeze(tree_map(lambda x: jnp.zeros_like(x,self.dtype),snn.variables)['params'])}
            return carry,s

        def flat_spike(model,carry,x):
            u,s = model(carry,x)
            return jnp.sum(s),(s,u)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(snn,carry,inputs):

            z,bwd,(s,carry['u']) = nn.vjp(flat_spike,snn,carry['u'],inputs,has_aux=True)
            grad_s_params,grad_s_u,_ = bwd(jnp.ones(z.shape,self.dtype))


            sig_tau = nn.sigmoid(self.init_tau)#jnp.mean(nn.sigmoid(jax.lax.stop_gradient(snn.variables['params']['nd']['tau'])))
            grad_s_x = grad_s_u/sig_tau

            ratio = sig_tau*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])
            #carry['J_u_x'] = ratio*carry['J_u_x'] + (1-ratio)*grad_s_x
            carry['J_u_x'] = sig_tau*carry['J_u_x'] + grad_s_x

            carry['a_hat'] = sig_tau*carry['a_hat'] + inputs
            carry['a_hat2'] = ratio*carry['a_hat2'] + (1-ratio)*carry['a_hat']

            return (carry,s),(carry['a_hat2'],grad_s_params,carry['J_u_x'],grad_s_x,jax.lax.stop_gradient(snn.variables['params']['cf']['kernel']))

        def f_bwd(res,g):
            a_hat,J_u_params,J_u_x,grad_s_x,kernel = res

            p_fu = Partial(fast_update,g[1].flatten()*J_u_x.flatten(),a_hat)
            g_rec_params = tree_map(p_fu,J_u_params)
            g_to_send = (g[1]*J_u_x).dot(kernel.T)

            return (g_rec_params,None,g_to_send)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        return f_custom(sl.SpikingBlockMod(self.connection_fn,self.output_sz,self.neural_dynamics,self.init_tau,self.spike_fn,dtype=self.dtype),carry,inputs)

    @staticmethod
    def initialize_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(0,dtype)
        carry['J_u_x'] = jnp.zeros(1,dtype)
        carry['a_hat'] = jnp.zeros(1,dtype)
        carry['a_hat2'] = jnp.zeros(1,dtype)
        carry['ratio'] = jnp.zeros(1,dtype)
        return carry

    @staticmethod
    def test_carry(dtype=jnp.float32):
        carry = {}
        carry['u'] = jnp.zeros(0,dtype)
        carry['J_u_params'] = jnp.zeros(2,dtype)
        return carry