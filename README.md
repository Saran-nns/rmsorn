## Reward Modulated Self-Organizing Recurrent Neural Networks 

PyPi package of RM-SORN: a reward-modulated self-organizing recurrent neural network: https://doi.org/10.3389/fncom.2015.00036

[![Build Status](https://travis-ci.org/Saran-nns/rmsorn.svg?branch=master)](https://travis-ci.org/Saran-nns/rmsorn)
[![codecov](https://codecov.io/gh/Saran-nns/rmsorn/branch/master/graph/badge.svg)](https://codecov.io/gh/Saran-nns/rmsorn)
[![PyPI version](https://badge.fury.io/py/rmsorn.svg)](https://badge.fury.io/py/rmsorn)
![PyPI - Downloads](https://img.shields.io/github/downloads/saran-nns/rmsorn/total)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://img.shields.io/github/license/Saran-nns/rmsorn)

#### To install the latest release:

```python
pip install rmsorn
```

The library is still in alpha stage, so you may also want to install the latest version from the development branch:

```python
pip install git+https://github.com/Saran-nns/rmsorn
```
#### Usage:
##### Update Network configurations

Navigate to home/conda/envs/ENVNAME/Lib/site-packages/rmsorn

or if you are unsure about the directory of rmsorn

Run

```python
import rmsorn

rmsorn.__file__
```
to find the location of the rmsorn package

Then, update/edit the configuration.ini

```python
from rmsorn.tasks import PatternRecognition

inputs, targets = PatternRecognitionTask.generate_sequence()
train_plast_inp_mat,X_all_inp,Y_all_inp,R_all, frac_pos_active_conn = SimulateRMSorn(phase = 'Plasticity', 
                                                                                      matrices = None,
                                                                                      inputs = np.asarray(inputs),sequence_length = 4, targets = targets,
                                                                                      reward_window_sizes = [1,5,10,20],
                                                                                      epochs = 1).train_rmsorn()
```
