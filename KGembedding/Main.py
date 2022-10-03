# Documentation
# install for IPython: https://ipython.org/install.html 
# Argparse code modified from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py 
#
#
# end of documentation

import torch
import argparse
import os
import pathlib
from Classes.Triple import *



# Hyperparameters 
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",   type=int,   default=10,   help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=128,     help="size of the batches")
parser.add_argument("--lr",         type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu",      type=int,   default=8,      help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=64,     help="dimensionality of the latent space")
#parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
#parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
#parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
#parser.add_argument("--channels",   type=int,   default=3,      help="number of image channels")
parser.add_argument("--n_critic",   type=int,   default=1,      help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=-1,   help="lower and upper clip value for disc. weights. (-1 = no clipping)")
parser.add_argument("--beta1",      type=float, default=0.5,    help="beta1 hyperparameter for Adam optimizer")

# Output options 
parser.add_argument("--sample_interval", type=int,  default=200,    help="iters between image samples")
parser.add_argument("--update_interval", type=int,  default=50,    help="iters between terminal updates")
parser.add_argument("--epochs_per_save", type=int,  default=5,    help="epochs between model saves")
parser.add_argument("--split_disc_loss", type=bool,  default=False,    help="whether to split discriminator loss into real/fake")
opt = parser.parse_args()
print(opt)


# Dataset directory
workDir = pathlib.Path().resolve()
dataDir = os.path.join(workDir.parent.resolve(), 'datasets')
FB15Kdir = os.path.join(dataDir, 'FB15K-237')

trainDir = os.path.join(FB15Kdir, 'train.txt')
testDir = os.path.join(FB15Kdir, 'test.txt')
validDir = os.path.join(FB15Kdir, 'valid.txt')


# Seed
seed = torch.Generator().seed()
print("Current seed: " + str(seed))




# --- Dataset loading & formatting ---
trainFile = open(trainDir, 'r')
testFile = open(testDir, 'r')
validFile = open(validDir, 'r')

trainData = []
testData = []
validData = []

#only used to: 
# 1. get the number of unique entities and relations, for one-hot encoding 
# 2. get their individual indices (key: name, value: index)
entities = dict()
entityID = 0
relations = dict()
relationID = 0

# Each part of the dataset is loaded the same way
for (file, data) in [(trainFile, trainData), (testFile, testData), (validFile, validData)]:
	# Load the specific file's contents, line by line
	for line in file:
		#load tab-separated triple
		split = line.split('\t')
		h = split[0]
		r = split[1]
		t = split[2]
		triple = Triple(h, r, t)
		data.append(triple)
		#keep track of unique entities and relations + their indices
		if h not in entities:
			entities[h] = entityID
			entityID += 1
		if r not in relations:
			relations[r] = relationID
			relationID += 1
		if t not in entities:
			entities[t] = entityID
			entityID += 1

entitiesN = len(entities)
relationsN = len(relations)

trainFile.close()
testFile.close()
validFile.close()


