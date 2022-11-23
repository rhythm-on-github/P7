# POGGAN(POwerful Graph GAN)

```
    ____  ____  _______________    _   __
   / __ \/ __ \/ ____/ ____/   |  / | / /
  / /_/ / / / / / __/ / __/ /| | /  |/ / 
 / ____/ /_/ / /_/ / /_/ / ___ |/ /|  /  
/_/    \____/\____/\____/_/  |_/_/ |_/  
```


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
- [Licence](#licence)

<!-- tocstop -->

## Features

Multiple modes to run the dataset
   - generate - simply generates triples
   - run - trains a model, generates and runs SDS
   - test - loads and tests all tests
   - tune - Uses RayTune, to optimize hyperparameters within a given search space, generates synth data and runs SDS
   - datatest - loads data compare valid vs test for SDS

Wide range of options
    - Support for multiple workers thanks to RayTune
    - Many easy to access options including hyperparameters, choice of dataset, hardware assignment for each worker and more

## Requirements
   - Python 3.8.13 (other versions might work but have not been tested)
   - Preferabily one or more Nvidia GPUs


## Getting Started

### Setup
To get startet we need to install some dependencies, this can be done with this command.

```
pip install -r requirements.txt
```

Download dataset from ...


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

You can change the default dataset(and much more!), by changing the string in default in Main.py to another dataset in same folder 

```python
parser.add_argument("--dataset",			type=str,	default="nations",	help="Which dataset folder to use as input")
```


You can also change the hyperparameters search space and and run modes parameters in Main.py
```python
#potentially run raytune, otherwise just train once
if opt.mode == "tune":
	config = {
		#"l1":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		#"l2":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		"lr":			tune.loguniform(1e-4, 1e-1),
		"batch_size":	tune.choice([16, 32, 64, 128, 256]),
		"latent_dim":	tune.choice([32, 64, 128, 256, 512]),
		"n_critic":		tune.choice([1, 2, 3]),
		"f_loss_min":	tune.loguniform(1e-6, 1e-1),
	}
```

## Paper

The paper can be found [here](https://www.overleaf.com/project/6332b13b23a385a2ea10c941) !!change link when its done!!

## The Team

We are a group of Computer Science students from 7th semester, who were tasked at creating "secure, scalable and usable systems".

## Licence
Copyright 2022 Anders Martin Hansen, Frederik Stær, Frederik Marinus Trudslev, Silas Oliver Torup Bachmann

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
