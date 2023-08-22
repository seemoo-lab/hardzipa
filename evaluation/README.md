# Description

This code (invoked from evaluate_all.py) is used to compute **entropy** of audio, illuminance, RGB, CO2, and TVOC data (all the data except raw audio recordings is inside the **hardzipa-data** folder on [Zenodo](https://zenodo.org/record/8263497)), as well as **dynamic time warping (DTW) distance** for illuminance, RGB, CO2, and TVOC data. 

The similarity of audio data is computed using [this code](https://github.com/seemoo-lab/ubicomp19_zero_interaction_security/tree/master/Schemes/audio) (please refer to [our paper](https://arxiv.org/abs/2306.04458) for more details about audio similarity metrics). 
