
# In[1]:


import pickle
import sys

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import jax
import jax.numpy as jnp

import optax
from OTPE import OSTL, OTTT, OTPE, Approx_OTPE

from jax.tree_util import Partial, tree_map, tree_leaves, tree_structure, tree_unflatten
import spiking_learning as sl

import randman_dataset as rd
import numpy as np
from utils import gen_test_data, cos_sim_train_func, online_sim_train_func, custom_snn, bp_snn


# In[2]:
################# SETTINGS #######################################

#update_ind,enc_ind,seed_ind = np.unravel_index(int(sys.argv[1])-1,(2,2,4))

update_ind = 0

output_size = 10
nlayers = 3
dim = 3
seq_len = 50
lr = ["001","00025"][update_ind]
manifold_seed_val = 0
init_seed_val = 0#seed_ind
manifold_seed = jax.random.PRNGKey(manifold_seed_val)
init_seed = jax.random.split(jax.random.PRNGKey(init_seed_val))[0]
dtype = jnp.float32#jnp.bfloat16
slope = 25
tau = dtype(2.) ### change to 2. ####
batch_sz = 128
spike_fn = sl.fs(slope)
n_iter = 20000 # 2000
layer_name = 128
update_time = ['offline','online'][update_ind]
timing = ['rate','time'][0]
if timing=='time':
    t_name = 'time'
    t = True
elif timing == 'rate':
    t_name = 'rate'
    t = False
if layer_name == 128:
    layer_sz = lambda i: 128
elif layer_name == 512:
    layer_sz = lambda i: 512
elif layer_name == 256:
    layer_sz = lambda i: 256

optimizer = optax.adamax(dtype([0.001,0.00025][update_ind]))

#------------------------------------------------------#

# In[3]:


train_data,train_labels = rd.make_spiking_dataset(nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=1000, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,seed2=manifold_seed,shuffle=False,dtype=dtype)


# In[4]:


gen_data = Partial(rd.make_spiking_dataset,nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=1000, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,shuffle=True,time_encode=t,dtype=dtype)


# In[5]:


test_data,test_labels = gen_test_data(gen_data,1,manifold_seed)


# In[6]:


carry = [OTPE.initialize_carry(dtype=dtype)]*nlayers


# In[7]:


OTTTmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OTTT, mod2=OTTT, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
OSTLmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OSTL, mod2=OSTL, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
OTPEmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OSTL, mod2=OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
Approx_OTPEmodel = custom_snn(output_sz=output_size, n_layers=nlayers, mod1=OTTT, mod2=Approx_OTPE, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
carry = [OTPE.initialize_carry(dtype=dtype)]*nlayers
params = OTPEmodel.init(init_seed,carry,train_data[0,:batch_sz])
carry,s = OTPEmodel.apply(params,carry,train_data[0,:batch_sz])
opt_state = optimizer.init(params)
orig_params = params


# In[8]:


test_carry = [OTPE.test_carry()]*nlayers
test_carry,_ = OTPEmodel.apply(params,test_carry,train_data[0])


# In[9]:


bp_model = bp_snn(output_sz=output_size, n_layers=nlayers, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)
bp_carry = carry
bp_params = bp_model.init(init_seed,bp_carry,train_data[0,:batch_sz])
struct = tree_structure(bp_params)
bp_params = tree_unflatten(struct,tree_leaves(orig_params))

bp_carry,s = bp_model.apply(bp_params,bp_carry,train_data[0,:batch_sz])
bp_opt_state = optimizer.init(bp_params)


# In[10]:


carry = tree_map(lambda x: jnp.zeros_like(x,dtype),carry)
test_carry = tree_map(lambda x: jnp.zeros_like(x,dtype),test_carry)


# In[11]:


key = jax.random.split(init_seed)[0]
cos = []
cos_per = []
val_acc = []
train_loss = []
all_params = [params]*4
all_params.append(bp_params)
all_opt = [opt_state]*4
all_opt.append(bp_opt_state)
carry = tree_map(lambda x: jnp.zeros_like(x,dtype),carry)
test_carry = tree_map(lambda x: jnp.zeros_like(x,dtype),test_carry)
best_acc = 0
best_params = [0]


# In[12]:


offline_training = jax.jit(Partial(cos_sim_train_func,OTTTmodel,
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
                       gen_data
                       ))


# In[13]:


online_training = jax.jit(Partial(online_sim_train_func,OTTTmodel,
                       Approx_OTPEmodel,
                       OSTLmodel,
                       OTPEmodel,
                       optimizer,
                       carry,
                       test_carry,
                       test_data,
                       test_labels,
                       batch_sz,
                       gen_data
                       ))


# In[14]:


with open('randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,0,manifold_seed_val,t_name,slope,lr,init_seed_val),'wb') as file:
            pickle.dump(tree_map(jnp.float32,all_params),file,protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


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

    if (epoch+1)%200 == 0: #200
        with open('randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,epoch+1,manifold_seed_val,t_name,slope,lr,init_seed_val),'wb') as file:
            pickle.dump(tree_map(jnp.float32,all_params),file,protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


np.save('randman_data/layer_cosine_similarity/sim_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val),cos_per)
np.save('randman_data/model_cosine_similarity/sim_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val),cos)
np.save('randman_data/accuracy/sim_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val),val_acc)
np.save('randman_data/loss/sim_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val),train_loss)

