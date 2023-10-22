To reproduce Randman results, go to "generate_randman_data.py"
Set the settings you wish to produce in the beginning of script.
Then execute the script, which will save the data.
In addition to pypi packages, you will need to install Randman from https://github.com/fzenke/randman


You can do the same for SHD with "generate_SHD_data.py" and "generate_SHD_data-f.py"
we recommend using "generate_SHD_data-f.py" for online learning
use "generate_SHD_data.py" for offline learning.
The "Tonic" package should take care of retrieving the data

For plotting, use "plots.ipynb".
For the loss landscape, set the matching settings of the generated Randman data in "gen_l.py" and run

To generate different seeds (0-3), change the value of "init_seed_val" in the scripts