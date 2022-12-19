# POGGAN(POwerful Graph GAN)

![POGGAN Logo](https://github.com/rhythm-on-github/P7/blob/main/poggan.png)


A knowledge graph generator with a plethora of options

<!-- toc -->

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
    - [Setup](#setup)
    - [First run](#first-run)
    - [Settings](#settings)
- [Paper](#paper)
- [The Team](#the-team)
- [Credits](#credits)
- [Licence](#licence)

<!-- tocstop -->

## Features
Multiple modes to run the dataset
   - run - trains a model, generates and runs SDS
   - test - loads and tests all tests
   - tune - Uses RayTune, to optimize hyperparameters within a given search space, generates synth data and runs SDS
   - dataTest - loads data compare valid vs test for SDS

Wide range of options
    - Support for multiple workers thanks to RayTune
    - Many easy to access options including hyperparameters, choice of dataset, hardware assignment for each worker and more

## Requirements
   - Python 3.8.13 (other versions might work but have not been tested)
   - Preferabily one or more Nvidia GPUs


## Getting Started

### Setup
To get started we need to install some dependencies. This can be done with the following command.

Note: On other OS's than Linux, packages might be different (such as PyTorch), in which case consult the developers website for correct installation.
```
pip install -r requirements.txt
```


### First run
```
python Main.py
```

optionally you can use 

```
python Main.py --mode [mode]
```

where [mode] is one of the modes listed above


### Settings

You can change the default dataset (and much more!) by changing the string in default in Main.py to another dataset in the same folder 

```python
parser.add_argument("--dataset",			type=str,	default="nations",	help="Which dataset folder to use as input")
```

If you do not want the nations dataset or if you are offline, the download can be disabled with the following option:
```python
opt.dataset_download = False
```

You can also change the hyperparameter search space and more such as run modes parameters in Main.py
```python
#--- Search space settings
# Learning rate
tune_lr_min = 1e-4 #default: 1e-4
tune_lr_max = 1e-1 #default: 1e-1

tune_batch_sizes = [16, 32, 64, 128, 256] #default: [16, 32, 64, 128, 256]
tune_latent_dims = [32, 64, 128, 256, 512] #default: [32, 64, 128, 256, 512]

tune_n_critics = [1, 2, 3] #default: [1, 2, 3]

tune_f_loss_min_min = 1e-6 #default: 1e-6
tune_f_loss_min_max = 1e-1 #default: 1e-1
```

## Paper
The paper can be found [here](https://www.overleaf.com/project/6332b13b23a385a2ea10c941) !!change link when its done!!

## The Team
We are a group of 7th semester Computer Science students from Aalborg University, who were tasked at creating "secure, scalable and usable systems".

## Credits
Our thanks goes out to these libraries datasets and the logo

[Pytorch](https://github.com/pytorch/pytorch) and their [paper](https://arxiv.org/abs/1912.01703) for providing a general framework for machine learning

[Raytune](https://github.com/ray-project/ray/tree/master/python/ray/tune) and their [paper](https://arxiv.org/abs/1807.05118) for providing a framework for hyperparameter tuning

[SDMetrics](https://github.com/sdv-dev/SDMetrics) for providing some of the metrics used to evaluate the quality of the produced output

[AMIE](https://github.com/lajus/amie) and their [paper](https://suchanek.name/work/publications/eswc-2020-amie-3.pdf) for rule inferencing

The Nations dataset [ZhenfengLei/KGDatasets](https://github.com/ZhenfengLei/KGDatasets)

The POGGAN logo [@Shiosei](https://twitter.com/Shiosei_)


## Licence
Copyright 2022 Anders Martin Hansen, Frederik Stær, Frederik Marinus Trudslev, Silas Oliver Torup Bachmann

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
