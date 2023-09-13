#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import jax
import jax.numpy as jnp

import optax
from OTPE import OSTL, OTTT, OTPE, Approx_OTPE, OTPE_mod2, Approx_OTPE_mod2, OTPE_front, Approx_OTPE_front

from jax.tree_util import Partial, tree_map, tree_leaves, tree_structure, tree_unflatten
import spiking_learning as sl

import randman_dataset as rd
import numpy as np
from utils import gen_test_data, cos_sim_train_func, online_sim_train_func, test_func, custom_snn, bp_snn
import torch
#import tonic


# In[2]:


torch.manual_seed(0)
np.random.seed(0)
output_size = 20
nlayers = 5
dim = 3
seq_len = 50
slope = 25
lr = "0001"
manifold_seed_val = 0
init_seed_val = 2
manifold_seed = jax.random.PRNGKey(manifold_seed_val)
init_seed = jax.random.split(jax.random.PRNGKey(init_seed_val))[0]
dtype = jnp.float32#jnp.bfloat16
tau = dtype(2.)
batch_sz = 128
spike_fn = sl.fs(slope)
n_iter = 10000
layer_name = 512
update_time = 'online'
if layer_name == 128:
    layer_sz = lambda i: 128
elif layer_name == 512:
    layer_sz = lambda i: 512
elif layer_name == 256:
    layer_sz = lambda i: 256

optimizer = optax.adamax(dtype(0.0001))


# In[3]:


# sensor_size = tonic.datasets.SHD.sensor_size
# train = tonic.datasets.SHD('data',train=True,transform=tonic.transforms.ToFrame(sensor_size=sensor_size,n_time_bins=seq_len))
# test = tonic.datasets.SHD('data',train=False,transform=tonic.transforms.ToFrame(sensor_size=sensor_size,n_time_bins=seq_len))


# In[4]:


train_data = jnp.load('data/train_data.npy')
train_labels = jnp.load('data/train_labels.npy')
test_data = jnp.load('data/test_data.npy')
test_labels = jnp.load('data/test_labels.npy')
val_data = jnp.load('data/val_data.npy')
val_labels = jnp.load('data/val_labels.npy')


# In[5]:


# train_data = []
# train_labels = []
# for i in range(len(train)):
#     d,l = train[i]
#     train_data.append(d)
#     train_labels.append(l)
# train_data = dtype(jnp.concatenate(train_data,axis=1))
# train_labels = dtype(jax.nn.one_hot(jnp.stack(train_labels),output_size))
# train_labels = jnp.tile(jnp.expand_dims(train_labels,axis=2),seq_len).transpose(2,0,1)

# val_data = train_data[:,0:len(train)//10]
# val_labels = train_labels[:,0:len(train)//10]


# train_data = train_data[:,len(train)//10:]
# train_labels = train_labels[:,len(train)//10:]


# In[6]:


def gen_data(seed2):
    inds = jnp.arange(train_labels.shape[1])
    inds = jax.random.permutation(seed2,inds,independent=True)
    data = train_data[:,inds[0:batch_sz]]
    labels = train_labels[:,inds[0:batch_sz]]
    return data, labels


# In[7]:


# test_data = []
# test_labels = []
# for i in range(len(test)):
#     d,l = test[i]
#     test_data.append(d)
#     test_labels.append(l)
# test_data = dtype(jnp.concatenate(test_data,axis=1))
# test_labels = dtype(jax.nn.one_hot(jnp.stack(test_labels),output_size))
# test_labels = jnp.tile(jnp.expand_dims(test_labels,axis=2),seq_len).transpose(2,0,1)


# In[8]:


# jnp.save('data/train_data.npy',train_data)
# jnp.save('data/train_labels.npy',train_labels)
# jnp.save('data/test_data.npy',test_data)
# jnp.save('data/test_labels.npy',test_labels)
# jnp.save('data/val_data.npy',val_data)
# jnp.save('data/val_labels.npy',val_labels)


# In[9]:


carry = [OTPE.initialize_carry(dtype=dtype)]*nlayers


# In[10]:


OTTTmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OSTL, mod2=OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
#OSTLmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OSTL, mod2=OSTL, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
OTPEmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OTPE_front, mod2=OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
Approx_OTPEmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=Approx_OTPE_front, mod2=Approx_OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
Mixmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OTTT, mod2=Approx_OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
carry = [OTPE.initialize_carry(dtype=dtype)]*nlayers
params = OTPEmodel.init(init_seed,carry,train_data[0,:batch_sz])
carry,s = OTPEmodel.apply(params,carry,train_data[0,:batch_sz])
opt_state = optimizer.init(params)
orig_params = params


# In[11]:


val_carry = [OTPE.test_carry()]*nlayers
val_carry,_ = OTPEmodel.apply(params,val_carry,val_data[0])

test_carry = [OTPE.test_carry()]*nlayers
test_carry,_ = OTPEmodel.apply(params,test_carry,test_data[0])


# In[12]:


bp_model = bp_snn(output_sz=output_size, n_layers=nlayers, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
bp_carry = carry
bp_params = bp_model.init(init_seed,bp_carry,train_data[0,:batch_sz])
struct = tree_structure(bp_params)
bp_params = tree_unflatten(struct,tree_leaves(orig_params))

bp_carry,s = bp_model.apply(bp_params,bp_carry,train_data[0,:batch_sz])
bp_opt_state = optimizer.init(bp_params)


# In[13]:


carry = tree_map(lambda x: jnp.zeros_like(x,dtype),carry)
val_carry = tree_map(lambda x: jnp.zeros_like(x,dtype),val_carry)
test_carry = tree_map(lambda x: jnp.zeros_like(x,dtype),test_carry)


# In[14]:


key = jax.random.split(init_seed)[0]
cos = []
cos_per = []
val_acc = []
train_loss = []
all_params = [params]*4
all_params.append(bp_params)
all_opt = [opt_state]*4
all_opt.append(bp_opt_state)
best_params = tree_map(jnp.zeros_like,all_params)
best_val = [0]*4
if update_time == 'offline':
    best_val.append(0)


# In[15]:


offline_training = jax.jit(Partial(cos_sim_train_func,OTTTmodel,
                       Approx_OTPEmodel,
                       Mixmodel,
                       OTPEmodel,
                       bp_model,
                       optimizer,
                       carry,
                       val_carry,
                       val_data,
                       val_labels,
                       batch_sz,
                       gen_data
                       ))


# In[16]:


online_training = jax.jit(Partial(online_sim_train_func,OTTTmodel,
                       Approx_OTPEmodel,
                       Mixmodel,
                       OTPEmodel,
                       optimizer,
                       carry,
                       val_carry,
                       val_data,
                       val_labels,
                       batch_sz,
                       gen_data
                       ))


# In[17]:


# with open('SHD_data/models/model_{}layer_{}_{}dim_{}seqlen_{}iter_{}seed_{}_sub_{}fs_adamax_lr{}'.format(nlayers,layer_name,dim,seq_len,0,init_seed_val,update_time,slope,lr),'wb') as file:
#            pickle.dump(tree_map(jnp.float32,all_params),file,protocol=pickle.HIGHEST_PROTOCOL)


# In[18]:


for epoch in range(n_iter):        
    
    if update_time == 'offline':

        all_loss, all_cosines, all_cosines_per, all_acc, all_params, all_opt, key = offline_training(all_params,all_opt,key)
        

        cos.append(np.stack(list(tree_map(jnp.float32,all_cosines))))
        cos_per.append(np.stack(list((tree_map(jnp.float32,all_cosines_per)))))
    
    elif update_time == 'online':

        all_loss, all_acc, all_params, all_opt, key = online_training(all_params,all_opt,key)


    val_acc.append(np.stack(list(tree_map(jnp.float32,all_acc))))

    train_loss.append(np.stack(list(tree_map(jnp.float32,all_loss))))
    
    #print(epoch)
    #print(val_acc[-1])
        #print(cos[-1])
        #print(cos_per[-1])
        #print(all_cosines_per)
    
    truth = np.greater(val_acc[-1],best_val).squeeze()
    best_val = np.where(truth,val_acc[-1],best_val)
    
    for i in range(len(best_val)):
        if truth[i]:
            best_params[i] = all_params[i]
    
        # if (epoch+1)%200 == 0: #200
        #     with open('SHD_data/models/model_{}layer_{}_{}dim_{}seqlen_{}iter_{}seed_{}_sub_{}fs_adamax_lr{}'.format(nlayers,layer_name,dim,seq_len,epoch+1,init_seed_val,update_time,slope,lr),'wb') as file:
        #         pickle.dump(tree_map(jnp.float32,all_params),file,protocol=pickle.HIGHEST_PROTOCOL)


# In[19]:


# with open('SHD_data/models/model_{}layer_{}_{}dim_{}seqlen_best_{}seed_{}_sub_{}fs_adamax_lr{}'.format(nlayers,layer_name,dim,seq_len,init_seed_val,t_name,slope,lr),'wb') as file:
#     pickle.dump(tree_map(jnp.float32,best_params),file,protocol=pickle.HIGHEST_PROTOCOL)


# In[20]:


OTTT_acc = test_func(OTTTmodel,best_params[0],test_carry,(test_data,test_labels))
Approx_OTPE_acc = test_func(OTTTmodel,best_params[1],test_carry,(test_data,test_labels))
OSTL_acc = test_func(OTTTmodel,best_params[2],test_carry,(test_data,test_labels))
OTPE_acc = test_func(OTTTmodel,best_params[3],test_carry,(test_data,test_labels))
#Mix_acc = test_func(OTTTmodel,best_params[4],test_carry,(test_data,test_labels))

all_acc = (OTTT_acc,Approx_OTPE_acc,OSTL_acc,OTPE_acc)

val_acc.append(np.stack(list(tree_map(jnp.float32,all_acc))))
if update_time == 'online':
    val_acc[-1] = val_acc[-1][0:4]
print(val_acc[-1])


# In[21]:


np.save('SHD_data/layer_cosine_similarity/sim_{}layer_{}_{}dim_{}seqlen_{}iter_{}_sub_{}fs_adamax_mix_lr{}_{}seed'.format(nlayers,layer_name,dim,seq_len,n_iter,update_time,slope,lr,init_seed_val),cos_per)
np.save('SHD_data/model_cosine_similarity/sim_{}layer_{}_{}dim_{}seqlen_{}iter_{}_sub_{}fs_adamax_mix_lr{}_{}seed'.format(nlayers,layer_name,dim,seq_len,n_iter,update_time,slope,lr,init_seed_val),cos)
np.save('SHD_data/accuracy/sim_{}layer_{}_{}dim_{}seqlen_{}iter_{}_sub_{}fs_adamax_mix_lr{}_{}seed'.format(nlayers,layer_name,dim,seq_len,n_iter,update_time,slope,lr,init_seed_val),val_acc)
np.save('SHD_data/loss/sim_{}layer_{}_{}dim_{}seqlen_{}iter_{}_sub_{}fs_adamax_mix_lr{}_{}seed'.format(nlayers,layer_name,dim,seq_len,n_iter,update_time,slope,lr,init_seed_val),train_loss)

