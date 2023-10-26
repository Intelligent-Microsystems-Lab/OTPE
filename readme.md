


<!-- You can do the same for SHD with "generate_SHD_data.py" and "generate_SHD_data-f.py"
Use "generate_SHD_data-f.py" for online learning and "generate_SHD_data.py" for offline learning.

For plotting, use "plots.ipynb".
For the loss landscape, set the matching settings of the generated Randman data in "gen_l.py" and run

To generate different seeds (0-3), change the value of "init_seed_val" in the scripts -->

# OTPE
Code for the paper "Incorporating Post-Synaptic Effects into Online Training of Feed-Forward SNNs". This repo contains code to reproduce all figures and experiments in the paper. If you find this code useful please cite the paper (insert ArXiv link here)

## Installation and Usage

To reproduce Randman results, go to `generate_randman_data.py`
Set the settings you wish to produce in the beginning of script.
Then execute the script, which will save the data.

You can do the same for SHD with `generate_SHD_data.py` and `generate_SHD_data-f.py`
Use "generate_SHD_data-f.py" for online learning and `generate_SHD_data.py` for offline learning.

In addition to pypi packages, you will need to install Randman from https://github.com/fzenke/randman

<!--

Simply `git clone` the repository to your home computer. The `numerical_results.py` file will recreate the numerical results figures in section 5.1. The `cnn.py` file contains the predictive coding and backprop CNNs used in section 5.2. The `lstm.py` and `rnn_names.py` files contain predictive coding and backprop LSTMs and RNNs used in section 5.3.

-->

## Requirements 

The code is written in [Pyython 3.x] and uses the following packages:
* [NumPY]
* [JAX] version 1.3.1
* [Tonic] version 1.x (only for downloading shakespeare dataset)
* [matplotlib] for plotting figures
* [randman]

## Citation

If you enjoyed the paper or found the code useful, please cite as: 

<!-- ```
@article{millidge2020predictive,
  title={Predictive Coding Approximates Backprop along Arbitrary Computation Graphs},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2006.04182},
  year={2020}
}
``` -->