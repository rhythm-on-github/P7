# Documentation
# install for IPython: https://ipython.org/install.html 
# Argparse code modified from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py 
#
#
# end of documentation

# imports
import torch
import torch.nn as nn
import argparse
import os
import pathlib
import datetime 
from datetime import datetime
from tqdm import tqdm
# rayTune imports
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# local imports
from Classes.Triple import *
from Classes.Encoding import *
from Classes.Training import *
from Classes.Testing import *
from Classes.Graph import *



# --- Settings ---
# NN choice 
from NNs.simpGAN import *

# Hyperparameters 
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",   type=int,   default=1,   help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=512,     help="size of the batches")
parser.add_argument("--lr",         type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu",      type=int,   default=8,      help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=64,     help="dimensionality of the latent space")
#parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
#parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
#parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
#parser.add_argument("--channels",   type=int,   default=3,      help="number of image channels")
parser.add_argument("--n_critic",   type=int,   default=3,      help="max. number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=-1,   help="lower and upper clip value for disc. weights. (-1 = no clipping)")
parser.add_argument("--beta1",      type=float, default=0.5,    help="beta1 hyperparameter for Adam optimizer")
parser.add_argument("--fake_loss_min",  type=float, default=0.0002,    help="target minimum fake loss for D")

# Output options 
parser.add_argument("--sample_interval", type=int,  default=50,    help="iters between image samples")
#parser.add_argument("--update_interval", type=int,  default=50,    help="iters between terminal updates")
#parser.add_argument("--epochs_per_save", type=int,  default=5,    help="epochs between model saves")
#parser.add_argument("--split_disc_loss", type=bool,  default=False,    help="whether to split discriminator loss into real/fake")
parser.add_argument("--out_n_triples",	type=int,	default=10000,	help="Number of triples to generate after training")
parser.add_argument("--test_only",		type=bool,	default=False,	help="Skip training/generating/saving and just load generated data for testing?")
opt = parser.parse_args()
print(opt)


# Dataset directory
def path_join(p1, p2):
	return os.path.join(p1, p2)

workDir  = pathlib.Path().resolve()
dataDir  = path_join(workDir.parent.resolve(), 'datasets')
inDataDir = path_join(dataDir, 'FB15K237')
loss_graphDir = path_join(dataDir, "_loss_graph")

# filepath for storing loss graph
graphDirAndName = path_join(loss_graphDir, "loss_graph.png")

trainName = 'train.txt' #temporarily use smaller dataset
testName  = 'test.txt'
validName = 'valid.txt'


# Seed
seed = torch.Generator().seed()
print("Current seed: " + str(seed))


# Computing device
tryCuda = True 
cuda = tryCuda and torch.cuda.is_available()
device = 'cpu'
if cuda: device = 'cuda:0'

# use raytune flag
rtune = False

# --- Dataset loading & formatting ---
os.chdir(inDataDir)
trainFile = open(trainName, 'r')
testFile  = open(testName, 'r')
validFile = open(validName, 'r')

trainData = []
testData  = []
validData = []

dataToLoad = [(trainFile, trainData), (testFile, testData), (validFile, validData)]

#only used to: 
# 1. get the number of unique entities and relations, for one-hot encoding 
# 2. get their individual indices (key: name, value: index)
entities = dict()
entityID = 0
relations  = dict()
relationID = 0

#potentially load generated data
genDir = path_join(dataDir, "_gen")
os.chdir(genDir)
genData = []
if opt.test_only:
	genFile = open("triples.csv", 'r')
	dataToLoad.append((genFile, genData))

# Each part of the dataset is loaded the same way
for (file, data) in dataToLoad:
	# Load the specific file's contents, line by line
	for line in tqdm(file, desc="load"):
		#load tab-separated triple
		split = line.split('\t')
		h = split[0]
		r = split[1]
		t = split[2][:-2] #remove \n
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

# make dataset encoders
trainDataEncoder = Encoder(trainData, entities, entitiesN, relations, relationsN)
testDataEncoder  = Encoder(testData,  entities, entitiesN, relations, relationsN)
validDataEncoder = Encoder(validData, entities, entitiesN, relations, relationsN)

# make data loaders
trainDataloader = torch.utils.data.DataLoader(trainDataEncoder, batch_size=opt.batch_size, shuffle=True)
testDataloader  = torch.utils.data.DataLoader(testDataEncoder,  batch_size=opt.batch_size, shuffle=True)
validDataloader = torch.utils.data.DataLoader(validDataEncoder, batch_size=opt.batch_size, shuffle=True)


# raytune metrics and searchspace
config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}


# --- Training ---
real_epochs = opt.n_epochs
if opt.test_only:
	real_epochs = 0

trainStart = datetime.now()
# setup
generator	=		Generator(opt.latent_dim, entitiesN, relationsN)
discriminator = Discriminator(opt.latent_dim, entitiesN, relationsN)
if cuda:
	generator.to(device)
	discriminator.to(device)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

loss_func = torch.nn.BCELoss()
optim_gen =  torch.optim.Adam(generator.parameters(),		lr=opt.lr, betas=(opt.beta1, 0.999))
optim_disc = torch.optim.Adam(discriminator.parameters(),	lr=opt.lr, betas=(opt.beta1, 0.999))

# If we had checkpoints it would be here
epochsDone = 0

# misc data 
fake_data = [] #latest generated data
real_losses = []
fake_losses = []
discriminator_losses = []
generator_losses = []

# For checking time (0 epochs timestamp)
now = datetime.now()
currentTime = now.strftime("%H:%M:%S")
timeStamps = [currentTime]

# For tqdm bars 
iters_per_epoch = len(trainDataloader)
columns = 60

# run training loop 
if real_epochs > 0:
	print("Starting training Loop...")
	
D_trains_since_G_train = 0;
for epoch in tqdm(range(epochsDone, real_epochs), position=0, leave=False, ncols=columns):
	# run an epoch 
	print("")
	for i, batch in tqdm(enumerate(trainDataloader), position=0, leave=True, total=iters_per_epoch, ncols=columns):
		#run a batch

		# train discriminator
		disc_losses = (real_losses, fake_losses, discriminator_losses)
		real_batch_size = train_discriminator(opt, Tensor, batch, fake_data, device, discriminator, generator, optim_disc, loss_func, disc_losses)
		D_trains_since_G_train += 1

		# only train generator every n_critic iterations or if the discriminator is overperforming
		D_overperforming = False
		if epoch >= 1 or i >= 1:
			D_overperforming = fake_losses[-1] < opt.fake_loss_min
		if(D_trains_since_G_train >= opt.n_critic or D_overperforming or i == 0):
			train_generator(fake_data, device, discriminator, optim_gen, loss_func, real_batch_size, generator_losses)
			D_trains_since_G_train = 0
		else:
			generator_losses.append(generator_losses[-1])

		# print to terminal
		#if(i % opt.update_interval == 0):
			#print_update()
			#print(self.optimizer_D.param_groups[0]['betas'])

		fake_data = []

		desc = " - losses r/f/D/G:  "
		desc += "{:.3f}".format(real_losses[-1])
		desc += " / " + "{:.4f}".format(fake_losses[-1])
		desc += " / " + "{:.3f}".format(discriminator_losses[-1])
		desc += " / " + "{:.1f}".format(generator_losses[-1])
		print(desc, end='\r')

		# save graph every sample_interval iteration
		if (i % opt.sample_interval == 0):
			saveGraph(graphDirAndName, generator_losses, discriminator_losses)

	# save graph after each epoch
	saveGraph(graphDirAndName, generator_losses, discriminator_losses)

	

trainEnd = datetime.now()
if real_epochs > 0:
	trainTime = (trainEnd - trainStart).total_seconds()
	print("Training time: " + "{:.0f}".format(trainTime) + " seconds")
	tpsTrain = (len(trainData)*opt.n_epochs)/trainTime;
	print("Average triples/s:" + "{:.0f}".format(tpsTrain) + "\n")





# --- Generating synthetic data ---
real_out_triples = opt.out_n_triples
if opt.test_only:
	real_out_triples = 0

#flip key/value for dictionaries for fast decoding (clears prior dictionaries to save memory)
genStart = datetime.now()
entitiesRev = dict()
relationsRev = dict()
for i in range(len(entities)):
	(key, value) = entities.popitem()
	entitiesRev[value] = key
for i in range(len(relations)):
	(key, value) = relations.popitem()
	relationsRev[value] = key

syntheticTriples = []

for i in tqdm(range(real_out_triples), ncols=columns, desc="gen"):
	z = Variable(Tensor(np.random.normal(0, 1, (opt.latent_dim,))))

	start = datetime.now()

	tripleEnc = generator(z)

	mid = datetime.now()

	triple = decode(tripleEnc, entitiesRev, entitiesN, relationsRev, relationsN)

	end = datetime.now()
	time1 = (mid - start).total_seconds()
	time2 = (end - mid).total_seconds()
	print(" - times: " + "{:.2f}".format(time1*1000) + "ms gen / " + "{:.2f}".format(time2*1000) + "ms decode", end='\r')

	syntheticTriples.append(triple)
	
genEnd = datetime.now()
if real_out_triples > 0:
	genTime = (genEnd - genStart).total_seconds()
	print("\nGeneration time: " + "{:.0f}".format(genTime) + " seconds", end="")
	tpsGen = (len(syntheticTriples))/genTime;
	print("Average triples/s:" + "{:.0f}".format(tpsGen) + "\n")




# --- Data formatting & saving  (and maybe synthetic data saving) ---

if not opt.test_only:
	# make/overwrite generated files 
	nodesFile = open("nodes.csv", "w")
	edgesFile = open("edges.csv", "w")
	triplesFile = open("triples.csv", "w")

	# format & save data 
	#format data as nodes and edges
	nodes = entitiesRev
	edges = syntheticTriples

	#save
	nodesFile.write("Id,Label,timeset,modularity_class\n")
	edgesFile.write("Source,Target,Type,Id,Label,timeset,Weight\n")
	for i in tqdm(range(len(nodes)), desc="save"):
		nodesFile.write(str(i) + "," + nodes[i] + ",,1\n")

	nextEdgeID = 0;
	#flip entity dictionary again for fast formatting
	for i in range(len(entitiesRev)):
		(value, key) = entitiesRev.popitem()
		entities[key] = value
	for triple in tqdm(edges, desc="save"):
		(h, r, t) = (triple.h, triple.r, triple.t)
		hID, tID = entities[h], entities[t]
		edgesFile.write(str(hID) + "," + str(tID) + ",Directed," + str(nextEdgeID) + "," + r + ",,1\n")
		nextEdgeID += 1
		triplesFile.write(h + "\t" + r + "\t" + t + "\n")

	nodesFile.close()
	edgesFile.close()
	triplesFile.close()






# --- Testing ---
print("\nTesting:")
(score, results) = (0, [])
if not opt.test_only:
	(score, results) = SDS(validData, syntheticTriples)
else:
	(score, results) = SDS(validData, genData)

print("\nDetailed SDS results: (lower = better)")
for result in results:
	(name, n, sum) = result
	print(name + " result:")
	#print("n size: " + str(n))
	#print("sum: " + "{:.2f}".format(sum))
	if n != 0:
		print("avg.: " + "{:.2f}".format(sum/n) + "\n")
	else:
		print("avg.: NaN" + " (lower = better)\n")
print("Overall SDS score: (lower = better)")
print("{:.2f}".format(score))


print("Done!")