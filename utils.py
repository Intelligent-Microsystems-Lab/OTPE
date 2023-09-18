from typing import Any, Callable

import jax
import jax.numpy as jnp

import optax
import flax.linen as nn

from jax.tree_util import Partial, tree_map, tree_leaves, tree_structure, tree_unflatten
import spiking_learning as sl
from jax.flatten_util import ravel_pytree

def gen_test_data(gen_data,reps,seed):
    keys = jax.random.split(seed,num=reps)
    test_data = []
    test_labels = []
    for i in range(reps):
        d,l = gen_data(seed2=keys[i])
        test_data.append(d)
        test_labels.append(l)
    test_data = jnp.concatenate(test_data,axis=1)
    test_labels = jnp.concatenate(test_labels,axis=1)
    return test_data,test_labels


class custom_snn(nn.Module):
    mod1: Callable
    mod2: Callable
    spike_fn: Callable
    layer_sz: Callable
    n_layers: int = 3
    output_sz: int = 10
    tau: float = 3.
    dtype: Any = jnp.float32
    def setup(self):
        snns = list()
        snns.append(self.mod1(
            Partial(nn.Dense,dtype=self.dtype,param_dtype=self.dtype),
            self.output_sz,
            Partial(sl.subtraction_LIF,dtype=self.dtype),
            self.tau,
            self.spike_fn,
            dtype=self.dtype))
        for i in range(1,self.n_layers):
            snns.append(self.mod2(Partial(nn.Dense,
                                          dtype=self.dtype,
                                          param_dtype=self.dtype),
                                          self.layer_sz(i),
                                          Partial(sl.subtraction_LIF,dtype=self.dtype),
                                          self.tau,
                                          self.spike_fn,
                                          dtype=self.dtype))
        self.snns = snns
    def __call__(self,carry,s):
        if carry[0]['u'].size == 0:
            in_ax1 = None
        else:
            in_ax1 = 0

        for i in range(self.n_layers):
            carry[i],s = jax.vmap(self.snns[self.n_layers-(i+1)],in_axes=(in_ax1,0))(carry[i],s)

        return carry, s
    


class bp_snn(nn.Module):
    spike_fn: Callable
    layer_sz: Callable
    n_layers: int = 5
    output_sz: int = 10
    tau: float = 3.
    dtype: Any = jnp.float32
    def setup(self):
        snns = list()
        snns.append(sl.SpikingBlock(nn.Dense(self.output_sz,dtype=self.dtype,param_dtype=self.dtype),
                                    sl.subtraction_LIF(self.tau,self.spike_fn,dtype=self.dtype)))
        for i in range(1,self.n_layers):
            snns.append(sl.SpikingBlock(nn.Dense(self.layer_sz(i),dtype=self.dtype,param_dtype=self.dtype),
                                        sl.subtraction_LIF(self.tau,self.spike_fn,dtype=self.dtype)))
        self.snns = snns
    def __call__(self,carry,s):
        if carry[0]['u'].size == 0:
            in_ax1 = None
        else:
            in_ax1 = 0

        for i in range(self.n_layers):
            carry[i]['u'],s = jax.vmap(self.snns[self.n_layers-(i+1)],in_axes=(in_ax1,0))(carry[i]['u'],s)

        return carry, s
    



#################################################


def loss_fn(model,params,carry,b):
   carry, s = model.apply(params,carry,b[0])
   loss = jnp.mean(optax.softmax_cross_entropy(s,b[1]))
   return loss,(carry,s,loss)

def mix_loss_fn(model,params,carry,leak_s,b):
   carry, s = model.apply(params,carry,b[0])
   leak_s = leak_s + s
   loss = jnp.mean(optax.softmax_cross_entropy(leak_s,b[1]))
   return loss,(carry,s,loss,leak_s)

def offline_apply_grad(seq_len,model,params,c,b):
    p_loss = Partial(loss_fn,model)
    grad,(carry,s,loss) = jax.jacrev(p_loss,has_aux=True)(params,c[0],b)
    grad = tree_map(lambda x,y: x+(y/seq_len),c[1],grad)
    return (carry,grad),(s,loss)

def online_apply_grad(optimizer,model,c,b):
    o_params, o_opt_state, o_carry = c
    p_loss = Partial(loss_fn,model)
    grad,(o_carry,s,loss) = jax.jacrev(p_loss,has_aux=True)(o_params,o_carry,b)
    #grad = tree_map(lambda x,y: x+(y/seq_len),c[1],grad)
    updates, o_opt_state = optimizer.update(grad,o_opt_state,o_params)
    o_params = optax.apply_updates(o_params, updates)
    return (o_params,o_opt_state,o_carry),(s,loss)

def mix_apply_grad(optimizer,model,c,b):
    o_params, o_opt_state, o_carry, leak_s = c
    p_loss = Partial(mix_loss_fn,model)
    grad,(o_carry,s,loss,leak_s) = jax.jacrev(p_loss,has_aux=True)(o_params,o_carry,leak_s,b)
    #grad = tree_map(lambda x,y: x+(y/seq_len),c[1],grad)
    updates, o_opt_state = optimizer.update(grad,o_opt_state,o_params)
    o_params = optax.apply_updates(o_params, updates)
    return (o_params,o_opt_state,o_carry,leak_s),(s,loss)

def offline_train_func(optimizer,model,params,carry,batch,opt_state):
    grad = tree_map(jnp.zeros_like,params)
    carry = tree_map(lambda x: jnp.stack([jnp.zeros_like(x[0])]*batch[0].shape[1]),carry)
    p_apply_grad = Partial(offline_apply_grad,batch[0].shape[0],model,params)
    (carry,grad),(s,loss) = jax.lax.scan(p_apply_grad,(carry,grad),batch)
    
    #updates, opt_state = optimizer.update(grad,opt_state,params,extra_args={"loss": jnp.mean(loss)})
    updates, opt_state = optimizer.update(grad,opt_state,params)
    params = optax.apply_updates(params, updates)

    return jnp.mean(loss), grad, params, opt_state

def online_train_func(optimizer,model,params,carry,batch,opt_state):
    grad = tree_map(jnp.zeros_like,params)
    carry = tree_map(lambda x: jnp.stack([jnp.zeros_like(x[0])]*batch[0].shape[1]),carry)
    p_apply_grad = Partial(online_apply_grad,optimizer,model)
    (params,opt_state,carry),(s,loss) = jax.lax.scan(p_apply_grad,(params,opt_state,carry),batch)
    
    return jnp.mean(loss), grad, params, opt_state

def mix_train_func(optimizer,model,params,carry,batch,opt_state):
    grad = tree_map(jnp.zeros_like,params)
    carry = tree_map(lambda x: jnp.stack([jnp.zeros_like(x[0])]*batch[0].shape[1]),carry)
    p_apply_grad = Partial(mix_apply_grad,optimizer,model)
    (params,opt_state,carry,leak_s),(s,loss) = jax.lax.scan(p_apply_grad,(params,opt_state,carry,jnp.zeros_like(batch[1])),batch)
    
    return jnp.mean(loss), grad, params, opt_state

def bp_loss_fn(bp_model,params,carry,b):
    p_apply = Partial(bp_model.apply,params)
    def p_loss(c,data):
        c, s = p_apply(c,data[0])
        loss = jnp.mean(optax.softmax_cross_entropy(s,data[1]))
        return c, loss
    carry, loss = jax.lax.scan(p_loss,carry,b)
    loss = jnp.mean(loss)
    return loss,loss

def bp_train_func(optimizer,bp_model,params,carry,batch,opt_state):
    carry = tree_map(lambda x: jnp.stack([jnp.zeros_like(x[0])]*batch[0].shape[1]),carry)
    p_loss = Partial(bp_loss_fn,bp_model)
    grad,loss = jax.jacrev(p_loss,has_aux=True)(params,carry,(batch[0],batch[1]))
    updates, opt_state = optimizer.update(grad,opt_state,params)
    #updates, opt_state = optimizer.update(grad,opt_state,params,extra_args={"loss": loss})
    params = optax.apply_updates(params, updates)

    return loss, grad, params, opt_state


def test_func(model,params,carry,batch):
    #carry = tree_map(lambda x: jnp.stack([jnp.zeros_like(x[0],dtype)]*batch[0].shape[1]),carry)
    p_apply = Partial(model.apply,params)
    carry,s = jax.lax.scan(p_apply,carry,batch[0])
    score = jnp.mean(jnp.equal(jnp.argmax(jnp.sum(s,axis=0),axis=1) + -1*(jnp.sum(jnp.sum(s,axis=0),axis=1)==0),jnp.argmax(jnp.sum(batch[1],axis=0),axis=1)))
    return score

def layer_cosines(approx_grad,bp_grad):
    return optax.cosine_similarity(approx_grad.flatten(),bp_grad.flatten())


########################################

def cos_sim_train_func(OTTTmodel,
                       Approx_OTPEmodel,
                       OSTLmodel,
                       OTPEmodel,
                       bp_model,
                       optimizer,
                       carry,
                       test_carry,
                       test_data,
                       test_labels,
                       batch_sz,
                       gen_data,
                       all_params,
                       all_opt,
                       key):

    key,_ = jax.random.split(key,num=2)
    train_data, train_labels = gen_data(seed2=key)
    batch = (train_data[:,:batch_sz],train_labels[:,:batch_sz])

    OTTT_train = Partial(offline_train_func,optimizer,OTTTmodel)
    Approx_OTPE_train = Partial(offline_train_func,optimizer,Approx_OTPEmodel)
    OSTL_train = Partial(offline_train_func,optimizer,OSTLmodel)
    OTPE_train = Partial(offline_train_func,optimizer,OTPEmodel)
    
    struct = tree_structure(all_params[0])
    cross_params = tree_unflatten(struct,tree_leaves(all_params[4]))

    _,cross_OTTT_grad, _, _ = OTTT_train(cross_params,carry,batch,all_opt[0])
    _,cross_Approx_OTPE_grad, _, _ = Approx_OTPE_train(cross_params,carry,batch,all_opt[1])
    _,cross_OSTL_grad, _, _ = OSTL_train(cross_params,carry,batch,all_opt[2])
    _,cross_OTPE_grad, _, _ = OTPE_train(cross_params,carry,batch,all_opt[3])


    OTTT_loss, OTTT_grad, all_params[0], all_opt[0] = OTTT_train(all_params[0],carry,batch,all_opt[0])
    Approx_OTPE_loss, Approx_OTPE_grad, all_params[1], all_opt[1] = Approx_OTPE_train(all_params[1],carry,batch,all_opt[1])
    OSTL_loss, OSTL_grad, all_params[2], all_opt[2] = OSTL_train(all_params[2],carry,batch,all_opt[2])
    OTPE_loss, OTPE_grad, all_params[3], all_opt[3] = OTPE_train(all_params[3],carry,batch,all_opt[3])

    bp_loss, bp_grad, all_params[4], all_opt[4] = bp_train_func(optimizer,bp_model,all_params[4],test_carry,batch,all_opt[4])

    struct = tree_structure(OTTT_grad)
    cross_grad = tree_unflatten(struct,tree_leaves(bp_grad))

    OTTT_cosines = optax.cosine_similarity(ravel_pytree(cross_OTTT_grad)[0],ravel_pytree(cross_grad)[0])
    Approx_OTPE_cosines = optax.cosine_similarity(ravel_pytree(cross_Approx_OTPE_grad)[0],ravel_pytree(cross_grad)[0])
    OSTL_cosines = optax.cosine_similarity(ravel_pytree(cross_OSTL_grad)[0],ravel_pytree(cross_grad)[0])
    OTPE_cosines = optax.cosine_similarity(ravel_pytree(cross_OTPE_grad)[0],ravel_pytree(cross_grad)[0])

    all_cosines = (OTTT_cosines,Approx_OTPE_cosines,OSTL_cosines,OTPE_cosines)

    OTTT_cosines = ravel_pytree(tree_map(layer_cosines,cross_OTTT_grad,cross_grad))[0]
    Approx_OTPE_cosines = ravel_pytree(tree_map(layer_cosines,cross_Approx_OTPE_grad,cross_grad))[0]
    OSTL_cosines = ravel_pytree(tree_map(layer_cosines,cross_OSTL_grad,cross_grad))[0]
    OTPE_cosines = ravel_pytree(tree_map(layer_cosines,cross_OTPE_grad,cross_grad))[0]

    all_cosines_per = (OTTT_cosines,Approx_OTPE_cosines,OSTL_cosines,OTPE_cosines)

    OTTT_acc = test_func(OTTTmodel,all_params[0],test_carry,(test_data,test_labels))
    Approx_OTPE_acc = test_func(Approx_OTPEmodel,all_params[1],test_carry,(test_data,test_labels))
    OSTL_acc = test_func(OSTLmodel,all_params[2],test_carry,(test_data,test_labels))
    OTPE_acc = test_func(OTPEmodel,all_params[3],test_carry,(test_data,test_labels))
    bp_acc = test_func(bp_model,all_params[4],test_carry,(test_data,test_labels))

    all_acc = (OTTT_acc,Approx_OTPE_acc,OSTL_acc,OTPE_acc,bp_acc)
    all_loss = (OTTT_loss, Approx_OTPE_loss, OSTL_loss, OTPE_loss, bp_loss)

    return all_loss, all_cosines, all_cosines_per, all_acc, all_params, all_opt, key


def online_sim_train_func(OTTTmodel,
                       Approx_OTPEmodel,
                       OSTLmodel,
                       OTPEmodel,
                       optimizer,
                       carry,
                       test_carry,
                       test_data,
                       test_labels,
                       batch_sz,
                       gen_data,
                       all_params,
                       all_opt,
                       key):

    key,_ = jax.random.split(key,num=2)
    train_data, train_labels = gen_data(seed2=key)
    batch = (train_data[:,:batch_sz],train_labels[:,:batch_sz])

    OTTT_train = Partial(online_train_func,optimizer,OTTTmodel)
    Approx_OTPE_train = Partial(online_train_func,optimizer,Approx_OTPEmodel)
    OSTL_train = Partial(online_train_func,optimizer,OSTLmodel)
    OTPE_train = Partial(online_train_func,optimizer,OTPEmodel)

    OTTT_loss, _, all_params[0], all_opt[0] = OTTT_train(all_params[0],carry,batch,all_opt[0])
    Approx_OTPE_loss, _, all_params[1], all_opt[1] = Approx_OTPE_train(all_params[1],carry,batch,all_opt[1])
    OSTL_loss, _, all_params[2], all_opt[2] = OSTL_train(all_params[2],carry,batch,all_opt[2])
    OTPE_loss, _, all_params[3], all_opt[3] = OTPE_train(all_params[3],carry,batch,all_opt[3])

    OTTT_acc = test_func(OTTTmodel,all_params[0],test_carry,(test_data,test_labels))
    Approx_OTPE_acc = test_func(Approx_OTPEmodel,all_params[1],test_carry,(test_data,test_labels))
    OSTL_acc = test_func(OSTLmodel,all_params[2],test_carry,(test_data,test_labels))
    OTPE_acc = test_func(OTTTmodel,all_params[3],test_carry,(test_data,test_labels))

    all_acc = (OTTT_acc,Approx_OTPE_acc,OSTL_acc,OTPE_acc)
    all_loss = (OTTT_loss, Approx_OTPE_loss, OSTL_loss, OTPE_loss)

    return all_loss, all_acc, all_params, all_opt, key

def online_front_train_func(OTTTmodel,
                       Approx_OTPEmodel,
                       OSTLmodel,
                       OTPEmodel,
                       fApprox_OTPEmodel,
                       fOTPEmodel, 
                       optimizer,
                       carry,
                       test_carry,
                       test_data,
                       test_labels,
                       batch_sz,
                       gen_data,
                       all_params,
                       all_opt,
                       key):

    key,_ = jax.random.split(key,num=2)
    train_data, train_labels = gen_data(seed2=key)
    batch = (train_data[:,:batch_sz],train_labels[:,:batch_sz])

    OTTT_train = Partial(online_train_func,optimizer,OTTTmodel)
    Approx_OTPE_train = Partial(online_train_func,optimizer,Approx_OTPEmodel)
    OSTL_train = Partial(online_train_func,optimizer,OSTLmodel)
    OTPE_train = Partial(online_train_func,optimizer,OTPEmodel)
    fApprox_OTPE_train = Partial(online_train_func,optimizer,fApprox_OTPEmodel)
    fOTPE_train = Partial(online_train_func,optimizer,fOTPEmodel)
    

    OTTT_loss, _, all_params[0], all_opt[0] = OTTT_train(all_params[0],carry,batch,all_opt[0])
    Approx_OTPE_loss, _, all_params[1], all_opt[1] = Approx_OTPE_train(all_params[1],carry,batch,all_opt[1])
    OSTL_loss, _, all_params[2], all_opt[2] = OSTL_train(all_params[2],carry,batch,all_opt[2])
    OTPE_loss, _, all_params[3], all_opt[3] = OTPE_train(all_params[3],carry,batch,all_opt[3])
    fApprox_OTPE_loss, _, all_params[4], all_opt[4] = fApprox_OTPE_train(all_params[4],carry,batch,all_opt[4])
    fOTPE_loss, _, all_params[5], all_opt[5] = fOTPE_train(all_params[5],carry,batch,all_opt[5])

    OTTT_acc = test_func(OTTTmodel,all_params[0],test_carry,(test_data,test_labels))
    Approx_OTPE_acc = test_func(OTTTmodel,all_params[1],test_carry,(test_data,test_labels))
    OSTL_acc = test_func(OTTTmodel,all_params[2],test_carry,(test_data,test_labels))
    OTPE_acc = test_func(OTTTmodel,all_params[3],test_carry,(test_data,test_labels))
    fApprox_OTPE_acc = test_func(OTTTmodel,all_params[4],test_carry,(test_data,test_labels))
    fOTPE_acc = test_func(OTTTmodel,all_params[5],test_carry,(test_data,test_labels))

    all_acc = (OTTT_acc,Approx_OTPE_acc,OSTL_acc,OTPE_acc,fApprox_OTPE_acc,fOTPE_acc)
    all_loss = (OTTT_loss, Approx_OTPE_loss, OSTL_loss, OTPE_loss,fApprox_OTPE_loss,fOTPE_loss)

    return all_loss, all_acc, all_params, all_opt, key