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

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Paper](#paper)
- [The Team](#the-team)
- [Licence](#licence)

<!-- tocstop -->

## Introduction

Loading of custom datasets
Includes multiple modes
   - generate - simply generates triples
   - run - trains a model, generates and runs SDS
   - test - loads and tests all tests
   - tune - Uses RayTune, to optimize hyperparameters within a given search space, generates synth data and runs SDS
   - datatest - loads data compare valid vs test for SDS
Wide range of options   

## Requirements
   - Python 3.8.13 (other versions might work but have not been tested)
   - preferabily one or more GPUs


## Getting Started

To get startet we need to install some dependencies, this can be done with this command.

```
pip install -r requirements.txt
```

Download dataset from ...

```
python Main.py
```

optionally you can use 

```
python Main.py --mode [mode]
```

where [mode] is one of the modes listed above

To change the default dataset change default to another dataset in same folder

```
parser.add_argument("--dataset",			type=str,	default="nations",	help="Which dataset folder to use as input")
```

to change hyperparemeters ...

## Paper

Paper goes here!

## The Team

We are a group of Computer Science students from 7th semester, who were tasked at creating "secure, scalable and usable systems"

## Licence
Copyright 2022 Anders Martin Hansen, Frederik Stær, Frederik Marinus Trudslev, Silas Oliver Torup Bachmann

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
