import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.tree_util import Partial, tree_flatten, tree_unflatten, tree_structure, tree_leaves
from randman_dataset import make_spiking_dataset
import randman_dataset as rd
from utils import gen_test_data, cos_sim_train_func, online_sim_train_func, custom_snn, bp_snn

import matplotlib as mpl
import matplotlib.font_manager as fm

import pickle
import spiking_learning as sl
import optax
import numpy as np
from sklearn.decomposition import PCA

from jax import random
import matplotlib.pyplot as plt
import os
import seaborn as sb
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


output_size = 10
nlayers = 3
dim = 3
seq_len = 50
lr = "001"
manifold_seed_val = 0
init_seed_val = 0
manifold_seed = jax.random.PRNGKey(manifold_seed_val)
init_seed = jax.random.split(jax.random.PRNGKey(init_seed_val))[0]
dtype = jnp.float32#jnp.bfloat16
slope = 25
tau = dtype(2.) ### change to 2. ####
batch_sz = 128
spike_fn = sl.fs(slope)
n_iter = 20000 # 2000
layer_name = 128
update_time = 'offline'
timing = 'rate'
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

#t = True
output_size = 10
dtype = jnp.float32
seed = 0
key = jax.random.PRNGKey(seed)
key2 = jax.random.split(key,num=1)[0]
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
marker_colors = [sb.set_hls_values('#e41a1c',l=0.2),
                 sb.set_hls_values('#377eb8',l=0.2),
                 sb.desaturate('#4daf4a',0.5),
                 sb.desaturate('#984ea3',0.5),
                 sb.desaturate('#ff7f00',0.5)]

model_indicator = 4

gen_data = Partial(rd.make_spiking_dataset,nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=1000, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,shuffle=True,time_encode=t,dtype=dtype)
#gen_data = Partial(make_spiking_dataset, nb_classes=10, nb_units=50, nb_steps=20, nb_samples=10000,
#                   dim_manifold=3, alpha=1., nb_spikes=1, seed=key, shuffle=True, one_hot=True, sp=1)


class bp_mlp_variable(nn.Module):
  n_layers: int = 5
  sz: int = 128

  def setup(self):
    snns = list()
    snns.append(sl.SpikingBlock(nn.Dense(10), sl.subLIF(tau, spike_fn)))
    for i in range(1, self.n_layers):
      snns.append(sl.SpikingBlock(nn.Dense(self.sz), sl.subLIF(tau, spike_fn)))
      # snns.append(osj.osjModel2(nn.Dense,300,sl.osjLIF,tau,sl.fast_sigmoid))
    self.snns = snns

  def __call__(self, carry, s):
    if carry[0]['u'].size == 0:
      in_ax1 = None
    else:
      in_ax1 = 0

    for i in range(self.n_layers):
      carry[i]['u'], s = jax.vmap(
          self.snns[self.n_layers - (i + 1)], in_axes=(in_ax1, 0))(carry[i]['u'], s)

    return carry, s


def load_params(file):
  with open(file, 'rb') as f:
    all_params = pickle.load(f)
  struct = tree_structure(all_params[-1])
  for i in range(len(all_params) - 1):
    all_params[i] = tree_unflatten(struct, tree_leaves(all_params[i]))
  return all_params


def npvec_to_tensorlist(pc, params):
  tree_val, tree_struct = jax.tree_util.tree_flatten(params)
  val_list = []
  counter = 0
  for i in [x.shape for x in tree_val]:
    increase = np.prod(i)
    val_list.append(pc[counter: int(counter + increase)].reshape(i))
    counter += increase

  return jax.tree_util.tree_unflatten(tree_struct, val_list)


def project1d(w, d):
  assert len(w) == len(d), "dimension does not match for w and "
  return jnp.dot(w, d) / np.linalg.norm(d)


def project2d(d, dx, dy, proj_method):
  if proj_method == "cos":
    # when dx and dy are orthorgonal
    x = project1d(d, dx)
    y = project1d(d, dy)
  elif proj_method == "lstsq":
    A = np.vstack([dx, dy]).T
    [x, y] = np.linalg.lstsq(A, d)[0]

  return x, y


#model = bp_mlp_variable(nlayers, sz=layer_sz)
model = bp_snn(output_sz=output_size, n_layers=nlayers, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)

carry = [{'u': jnp.zeros((batch_sz, 10))}] + \
    [{'u': jnp.zeros((batch_sz, layer_name))}] * (nlayers - 1)
carry = carry[::-1]


def loss_fn(params, carry, b):
  c, s = model.apply(params, carry, b[0])
  loss = jnp.mean(optax.softmax_cross_entropy(s, b[1]))
  return c, loss


@jax.jit
def get_loss(params, gen_key):
  data, logits = gen_data(seed2=gen_key)
  #data,logits = gen_test_data(gen_data,1,manifold_seed)
  p_loss = Partial(loss_fn, params)
  c, loss = jax.lax.scan(
      p_loss, carry, (data[:, :batch_sz], logits[:, :batch_sz]))
  return jnp.mean(loss)


def get_surface(x, y, xdirection, ydirection, variables):

  xv, yv = jnp.meshgrid(x, y)

  def surface_parallel(ix, iy):
    interpolate_vars = jax.tree_util.tree_map(
        lambda w, x, y: w + x * ix + y * iy,
        variables,
        xdirection,
        ydirection,
    )
    return get_loss(interpolate_vars, key2)


  zv_list = []
  for i in range(int(xv.flatten().shape[0] / 100)):
    zv = jax.vmap(surface_parallel)(
        jnp.array(xv.flatten())[(i * 100): (i + 1) * 100],
        jnp.array(yv.flatten())[(i * 100): (i + 1) * 100],
    )
    zv_list.append(zv)

  return xv, yv, np.stack(zv_list).flatten().reshape(xv.shape)


params_end = load_params(
'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val))
    #'data_final/models/model_{}layer_{}_3dim_20seqlen_{}iter_1sp_{}seed_time_sub_adamax'

matrix = []
for i in range(0,n_iter+200,200):
  for j in range(5):
    tmp = load_params(
'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,i,manifold_seed_val,t_name,slope,lr,init_seed_val))
    diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
    matrix.append(jnp.hstack([x.reshape(-1)
                  for x in jax.tree_util.tree_flatten(diff_tmp)[0]]))


pca = PCA(n_components=2)
pca.fit(np.array(matrix))

pc1 = np.array(pca.components_[0])
pc2 = np.array(pca.components_[1])

angle = jnp.dot(pc1, pc2) / (jnp.linalg.norm(pc1) * jnp.linalg.norm(pc2))

xdirection = npvec_to_tensorlist(pc1, params_end[model_indicator])
ydirection = npvec_to_tensorlist(pc2, params_end[model_indicator])

ratio_x = pca.explained_variance_ratio_[0]
ratio_y = pca.explained_variance_ratio_[1]

dx = pc1
dy = pc2

xcoord = {}
ycoord = {}
x_abs_max = 0
y_abs_max = 0
for j in range(5):
  xcoord[j] = []
  ycoord[j] = []
  for i in range(0,n_iter+200,200):
    tmp = load_params(
'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,i,manifold_seed_val,t_name,slope,lr,init_seed_val))
    diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
    diff_tmp = jnp.hstack([x.reshape(-1)
                          for x in jax.tree_util.tree_flatten(diff_tmp)[0]])

    tmp_x, tmp_y = project2d(diff_tmp, dx, dy, 'cos')
    xcoord[j].append(tmp_x)
    ycoord[j].append(tmp_y)

    if np.abs(tmp_x) > x_abs_max:
      x_abs_max = abs(tmp_x)
    if np.abs(tmp_y) > y_abs_max:
      y_abs_max = abs(tmp_y)


# buffer_y = (np.max(ycoord) - np.min(ycoord)) * 0.05
# buffer_x = (np.max(xcoord) - np.min(xcoord)) * 0.05

# x = np.linspace(
#     np.min(xcoord) - buffer_x,
#     np.max(xcoord) + buffer_x,
#     100,
# )
# y = np.linspace(
#     np.min(ycoord) - buffer_y,
#     np.max(ycoord) + buffer_y,
#     100,
# )


buffer_y = y_abs_max * 0.05
buffer_x = x_abs_max * 0.05

x = np.linspace(
    (-1*x_abs_max) - buffer_x,
    x_abs_max + buffer_x,
    100,
)
y = np.linspace(
    (-1*y_abs_max) - buffer_y,
    y_abs_max + buffer_y,
    100,
)


xv, yv, zv = get_surface(x, y, xdirection, ydirection, params_end[model_indicator])

font_size = 23
gen_lw = 8

#plt.rc("font", family="Helvetica", weight="bold")
plt.rc("font", weight="bold")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.4, 8.5))

#ax.set_title(
#    "Loss Landscape and Training Trajectory for 3-layer, 512 width",
#    fontdict={"weight": "bold", "size": font_size + 3},
#)


def fmt(x):
  s = f"{x:.3f}"
  if s.endswith("0"):
    s = f"{x:.0f}"
  return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


#CS = ax.contourf(xv, yv, zv, 100)
CS = ax.contour(xv, yv, zv, 100)
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

for j in range(5):
  ax.plot(
      xcoord[j],
      ycoord[j],
      label=str(j),
      color=colors[j],
      marker="o",
      markeredgecolor='black',
      markerfacecolor="None",
      markersize=8,
      # color="r",
      linewidth=gen_lw,
  )

ax.legend(['OTTT','Approx OTPE', 'OSTL', 'OTPE', 'BPTT'],fontsize=32)
ax.set_xlabel(
    "1st PC: %.2f %%" % (ratio_x * 100),
    fontdict={"weight": "bold", "size": font_size},
)
ax.set_ylabel(
    "2nd PC: %.2f %%" % (ratio_y * 100),
    fontdict={"weight": "bold", "size": font_size},
)

plt.tight_layout()
#plt.savefig("ll_offline_3_128"+str(model_indicator)+".png", dpi=300, bbox_inches="tight")
plt.savefig("plots/randman_ll_rate_offline_{}_{}_0".format(nlayers,layer_name)+".svg", dpi=300, bbox_inches="tight")
plt.close()
