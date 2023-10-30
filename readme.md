


<!-- You can do the same for SHD with "generate_SHD_data.py" and "generate_SHD_data-f.py"
Use "generate_SHD_data-f.py" for online learning and "generate_SHD_data.py" for offline learning.

For plotting, use "plots.ipynb".
For the loss landscape, set the matching settings of the generated Randman data in "gen_l.py" and run

To generate different seeds (0-3), change the value of "init_seed_val" in the scripts -->

# OTPE
Code for the paper "Incorporating Post-Synaptic Effects into Online Training of Feed-Forward SNNs". This repo contains code to reproduce all figures and experiments in the paper. If you find this code useful please cite the paper <!-- (insert ArXiv link here) -->

## Installation and Usage

To reproduce Randman results, run `python generate_randman_data.py`
Set the settings you wish to produce in the beginning of script.
Then execute the script, which will save the data.

You can do the same for SHD with `generate_SHD_data.py` and `generate_SHD_data-f.py`
Use "generate_SHD_data-f.py" for online learning and `generate_SHD_data.py` for offline learning.

For plotting, use `plots.ipynb`.
For the loss landscape, set the matching settings of the generated Randman data in `gen_l.py` before running

To generate different seeds (0-3), change the value of "init_seed_val" in the scripts

<!--

Simply `git clone` the repository to your home computer. The `numerical_results.py` file will recreate the numerical results figures in section 5.1. The `cnn.py` file contains the predictive coding and backprop CNNs used in section 5.2. The `lstm.py` and `rnn_names.py` files contain predictive coding and backprop LSTMs and RNNs used in section 5.3.

-->

## Requirements 

The code is written using Python 3.10 the following packages:

* [jax] version 0.4.14
* [flax] 0.7.2
* [tonic] for loading Spiking Heidelberg Digits
* [randman] For randman dataset (install Randman from https://github.com/fzenke/randman)
* [optax] 0.1.7
* [numpy]
* [matplotlib]

## Citation

<!-- For the original paper and citing: -->

<!-- ```
@article{placeholder,
  title={},
  author={},
  journal={},
  year={}
}
``` -->