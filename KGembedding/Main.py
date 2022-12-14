# Documentation
# install for IPython: https://ipython.org/install.html 
# Argparse code modified from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py 
# Ray tune code modified from: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#the-train-function 
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
import json
import pandas as pd
from math import floor, ceil
import requests

#ray imports can be outcommented if not using raytune / checkpoints
#import ray 
#from ray import tune
#from ray.air import session
#from ray.air.checkpoint import Checkpoint
#from ray.tune.schedulers import ASHAScheduler

# local imports
from Classes.Triple import *
from Classes.Encoding import *
from Classes.Training import *
from Classes.Testing import *
from Classes.Graph import *



# --- Settings ---
# NN choice 
from NNs.convGAN3 import *

# Hyperparameters 
#tuning implemented
parser = argparse.ArgumentParser()
parser.add_argument("--lr",         type=float, default=0.0002, help="Learning rate")
parser.add_argument("--batch_size", type=int,   default=64,     help="Size of the batches")
parser.add_argument("--latent_dim", type=int,   default=64,     help="Dimensionality of the latent space")
parser.add_argument("--n_critic",   type=int,   default=2,      help="Max. number of training steps for discriminator per iter")
parser.add_argument("--f_loss_min", type=float, default=0.02,    help="Target minimum fake loss for D")
#tuning not explicitly implemented
parser.add_argument("--n_epochs",   type=int,   default=2,   help="Number of epochs of training")
#tuning not implemented for these
parser.add_argument("--clip_value", type=float, default=-1,   help="Lower and upper clip value for disc. weights. (-1 = no clipping)")
parser.add_argument("--beta1",      type=float, default=0.5,    help="Beta1 hyperparameter for Adam optimizer")

# Hyperparameter tuning options
parser.add_argument("--tune_n_valid_triples",	type=int,	default=10000,	help="With raytune, no. of triples to generate for validation (rounded up to nearest mult. of batch size)")
parser.add_argument("--tune_samples",			type=int,	default=10,	help="Total samples taken with raytune")
parser.add_argument("--max_concurrent_samples",	type=int,	default=2,	help="Max. samples to run at the same time with raytune. (use None for unlimited)")
parser.add_argument("--tune_max_epochs",		type=int,	default=2,	help="How many epochs at most per run with raytune")
parser.add_argument("--tune_cpus",				type=float,   default=2,      help="Number of cpu threads to use during batch generation")
parser.add_argument("--tune_gpus",				type=float,	default=1,	help="How many gpus to reserve per trial with raytune (does not influence total no. of gpus used)")
parser.add_argument("--tune_subset_size",		type=float,	default=0.1,	help="How large the subset of train data should be during tuning")

# General options
parser.add_argument("--dataset",			type=str,	default="FB15K237",	help="Which dataset folder to use as input")
parser.add_argument("--mode",				type=str,	default="test",	help="Which thing to do, overall (run/test/tune/dataTest)")
#"Booleans"
parser.add_argument("--use_gpu",			type=str,	default="True",	help="Use GPU for training (when without raytune)? (cuda)")
parser.add_argument("--disable_download",	type=str,	default="True",	help="Downloads the nations dataset from a GitHub repo")

# Output options 
parser.add_argument("--sample_interval",	type=int,  default=5000,    help="Iters between image samples")
parser.add_argument("--tqdm_columns",		type=int,  default=60,    help="Total text columns for tqdm loading bars")
#parser.add_argument("--epochs_per_save",	type=int,  default=5,    help="Epochs between model saves")
#parser.add_argument("--split_disc_loss",	type=str,  default="False",    help="Whether to split discriminator loss into real/fake")
parser.add_argument("--out_n_triples",		type=int,	default=10000,	help="Number of triples to generate after training (rounded up to nearest mult. of batch size)")
parser.add_argument("--use_sdmetrics",		type=str,	default="True",	help="Use sdmetrics for evaluation in test mode?")

opt = parser.parse_args()

#checkpoint options used and changed later
opt.load_checkpoint = True
opt.save_checkpoints = True

#convert "Booleans" to actual bools
if opt.use_gpu == "False":
	opt.use_gpu = False
else:
	opt.use_gpu = True

if opt.disable_download == "False":
	opt.disable_download = False
else:
	opt.disable_download = True

#if opt.split_disc_loss == "False":
#	opt.split_disc_loss = False
#else:
#	opt.split_disc_loss = True

if opt.use_sdmetrics == "False":
	opt.use_sdmetrics = False
else:
	opt.use_sdmetrics = True

print(opt)




# --- setup ---
# Dataset directory
def path_join(p1, p2):
	return os.path.join(p1, p2)

workDir  = pathlib.Path().resolve()
dataDir  = path_join(workDir.parent.resolve(), 'datasets')
inDataDir = path_join(dataDir, opt.dataset)
loss_graphDir = path_join(dataDir, "_loss_graph")
if not os.path.exists(loss_graphDir):
	os.makedirs(loss_graphDir)

# filepath for storing loss graph
graphDirAndName = path_join(loss_graphDir, "loss_graph.png")

trainName = 'train.txt'
validName = 'valid.txt'
testName  = 'test.txt'

# Seed
seed = torch.Generator().seed()
print("Current seed: " + str(seed))

# Computing device
cuda = opt.use_gpu and torch.cuda.is_available()
device = 'cpu'
if cuda: device = 'cuda:0'

#--- Search space settings
# Learning rate
tune_lr_min = 1e-4 #default: 1e-4
tune_lr_max = 1e-1 #default: 1e-1

tune_batch_sizes = [16, 32, 64, 128, 256] #default: [16, 32, 64, 128, 256]
tune_latent_dims = [32, 64, 128, 256, 512] #default: [32, 64, 128, 256, 512]

tune_n_critics = [1, 2, 3] #default: [1, 2, 3]

tune_f_loss_min_min = 1e-6 #default: 1e-6
tune_f_loss_min_max = 1e-1 #default: 1e-1




# --- Dataset loading & formatting ---
# Downloads nations dataset
nationsDatasetDir = path_join(dataDir, 'nations')
if not os.path.exists(nationsDatasetDir) and opt.disable_download == False:
	os.makedirs(nationsDatasetDir)

	url = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Nations/test.txt'
	r = requests.get(url, allow_redirects=True)
	open(path_join(nationsDatasetDir, 'test.txt'), 'wb').write(r.content)

	url = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Nations/train.txt'
	r = requests.get(url, allow_redirects=True)
	open(path_join(nationsDatasetDir, 'train.txt'), 'wb').write(r.content)

	url = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Nations/valid.txt'
	r = requests.get(url, allow_redirects=True)
	open(path_join(nationsDatasetDir, 'valid.txt'), 'wb').write(r.content)

trainFile = open(path_join(inDataDir, trainName), 'r')
validFile = open(path_join(inDataDir, validName), 'r')
testFile  = open(path_join(inDataDir, testName), 'r')

trainData = []
validData = []
testData  = []

dataToLoad = [(trainFile, trainData), (validFile, validData), (testFile, testData)]

#only used to: 
# 1. get the number of unique entities and relations, for one-hot encoding 
# 2. get their individual indices (key: name, value: index)
entities = dict()
entityID = 0
relations  = dict()
relationID = 0

#potentially load generated data
genDir = path_join(dataDir, "_gen")
if not os.path.exists(genDir):
	os.makedirs(genDir)
genData = []
if opt.mode == "test":
	genFile = open(path_join(genDir, "triples.csv"), 'r')
	dataToLoad.append((genFile, genData))
	genReader = pd.read_csv(path_join(genDir, "triples.csv"), sep='\t', names=['head','relation','tail'])
	SDGenData = genReader
	testReader = pd.read_csv(path_join(inDataDir, testName), sep='\t', names=['head','relation','tail'])
	SDTestData = testReader
	validReader = pd.read_csv(path_join(inDataDir, validName), sep='\t', names=['head','relation','tail'])
	SDValidData = validReader
	trainReader = pd.read_csv(path_join(inDataDir, trainName), sep='\t', names=['head','relation','tail'])
	SDTrainData = trainReader
	SDFullDBData = SDTestData.append(SDValidData.append(SDTrainData, ignore_index=True), ignore_index=True)
	metadataDir = path_join(workDir.parent.resolve(), "metadata")
	with open(path_join(metadataDir, 'metadatafile.json')) as f:
		my_metadata_dict = json.load(f)

if opt.mode == "dataTest":
	genReader = pd.read_csv(path_join(genDir, "triples.csv"), sep='\t', names=['head','relation','tail'])
	SDGenData = genReader
	testReader = pd.read_csv(path_join(inDataDir, testName), sep='\t', names=['head','relation','tail'])
	SDTestData = testReader
	validReader = pd.read_csv(path_join(inDataDir, validName), sep='\t', names=['head','relation','tail'])
	SDValidData = validReader
	trainReader = pd.read_csv(path_join(inDataDir, trainName), sep='\t', names=['head','relation','tail'])
	SDTrainData = trainReader
	SDFullDBData = SDTestData.append(SDValidData.append(SDTrainData, ignore_index=True), ignore_index=True)
	metadataDir = path_join(workDir.parent.resolve(), "metadata")
	with open(path_join(metadataDir, 'metadatafile.json')) as f:
		my_metadata_dict = json.load(f)

# Each part of the dataset is loaded the same way
for (file, data) in dataToLoad:
	# Load the specific file's contents, line by line
	for line in tqdm(file, desc="load"):
		#load tab-separated triple
		split = line.split('\t')
		h = split[0]
		r = split[1]
		t = split[2][:-1] #remove \n
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

print("")
print("Entities loaded:  " + str(len(entities)))
print("Relations loaded: " + str(len(relations)))

entitiesN = len(entities)
relationsN = len(relations)

trainFile.close()
validFile.close()
testFile.close()

# make dataset encoders
trainDataEncoder = Encoder(trainData, entities, entitiesN, relations, relationsN)
validDataEncoder = Encoder(validData, entities, entitiesN, relations, relationsN)
testDataEncoder  = Encoder(testData,  entities, entitiesN, relations, relationsN)





# --- Training ---
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(config, gen, disc, Gen, Disc, tune_done=False):
	# make data loaders
	validDataloader = torch.utils.data.DataLoader(validDataEncoder, batch_size=config["batch_size"], shuffle=True)
	testDataloader  = torch.utils.data.DataLoader(testDataEncoder,  batch_size=config["batch_size"], shuffle=True)

	#For tuning, only use a subset of the data
	tune_subset_size = opt.tune_subset_size
	if (opt.mode != "tune") or tune_done:
		tune_subset_size = 1.0
	subset_total_count = floor(len(trainData)*tune_subset_size)
	trainSubset, _ = torch.utils.data.random_split(trainDataEncoder, [subset_total_count, len(trainData)-subset_total_count])
	trainDataloader = torch.utils.data.DataLoader(trainSubset, batch_size=config["batch_size"], shuffle=True)
	print("Data subset size: " + str(len(trainDataloader.dataset)))

	
	real_epochs = opt.n_epochs
	if opt.mode == "tune":
		real_epochs = opt.tune_max_epochs

	trainStart = datetime.now()
	# setup
	gen	=	 Gen(config["latent_dim"], entitiesN, relationsN)
	disc =	Disc(config["latent_dim"], entitiesN, relationsN)

	loss_func = torch.nn.BCELoss()
	optim_gen =  torch.optim.Adam(gen.parameters(),		lr=config["lr"], betas=(opt.beta1, 0.999))
	optim_disc = torch.optim.Adam(disc.parameters(),	lr=config["lr"], betas=(opt.beta1, 0.999)) 

	# Checkpoint restoring
	if opt.mode == "tune" and opt.load_checkpoint:
		loaded_checkpoint = session.get_checkpoint()
		if loaded_checkpoint:
			disc.to('cpu')
			gen.to('cpu')
			with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
				D_state, G_state, D_optimizer_state, G_optimizer_state = torch.load(path_join(loaded_checkpoint_dir, "checkpoint.pt"), map_location=torch.device('cpu'))
				disc.load_state_dict(D_state)
				optim_disc.load_state_dict(D_optimizer_state)
				gen.load_state_dict(G_state)
				optim_gen.load_state_dict(G_optimizer_state)
	
				
	# Find out computing device again, in case it's a worker thread
	cuda = opt.use_gpu and torch.cuda.is_available()
	device = 'cpu'
	if cuda: device = 'cuda:0'
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	disc.to(device)
	gen.to(device)


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
	columns = opt.tqdm_columns

	# run training loop 
	if real_epochs > 0:
		print("Starting training Loop...")
	
	D_trains_since_G_train = 0
	epochs = range(epochsDone, real_epochs)
	if opt.mode != "tune" or tune_done:
		epochs = tqdm(epochs, position=0, leave=False, ncols=columns)
		print("epochs: " + str(epochs))
	for epoch in epochs:
		# run an epoch 
		print("")
		dataLoader = enumerate(trainDataloader)
		if opt.mode != "tune" or tune_done:
			dataLoader = tqdm(dataLoader, position=0, leave=True, total=iters_per_epoch, ncols=columns)
		for i, batch in dataLoader:
			#run a batch

			# train discriminator
			disc_losses = (real_losses, fake_losses, discriminator_losses)
			real_batch_size = train_discriminator(opt, Tensor, batch, fake_data, device, disc, gen, optim_disc, loss_func, disc_losses)
			D_trains_since_G_train += 1

			# only train generator every n_critic iterations or if the discriminator is overperforming
			D_overperforming = False
			if epoch >= 1 or i >= 1:
				D_overperforming = fake_losses[-1] < config["f_loss_min"]
			if(D_trains_since_G_train >= config["n_critic"] or D_overperforming or i == 0):
				train_generator(fake_data, device, disc, optim_gen, loss_func, real_batch_size, generator_losses)
				D_trains_since_G_train = 0
			else:
				generator_losses.append(generator_losses[-1])

			fake_data = []

			if opt.mode != "tune":
				desc = " - losses r/f/D/G:  "
				desc += "{:.3f}".format(real_losses[-1])
				desc += " / " + "{:.4f}".format(fake_losses[-1])
				desc += " / " + "{:.3f}".format(discriminator_losses[-1])
				desc += " / " + "{:.1f}".format(generator_losses[-1])
				print(desc, end='\r')

			# save graph every sample_interval iteration
			if (i % opt.sample_interval == 0) and opt.mode != "tune":
				saveGraph(graphDirAndName, generator_losses, discriminator_losses)

		# save graph after each epoch
		if opt.mode != "tune":
			saveGraph(graphDirAndName, generator_losses, discriminator_losses)


		# Here we save a checkpoint. It is automatically registered with
		# Ray Tune and can be accessed through `session.get_checkpoint()`
		# API in future iterations.
		if opt.mode == "tune" and opt.save_checkpoints:
			os.makedirs(path_join(genDir, "my_model"), exist_ok=True)
			torch.save(
				(disc.to('cpu').state_dict(), gen.to('cpu').state_dict(), optim_disc.state_dict(), optim_gen.state_dict()),
				path_join(path_join(genDir, "my_model"), "checkpoint.pt")
			)
			disc.to(device)
			gen.to(device)
			checkpoint = Checkpoint.from_directory(path_join(genDir, "my_model"))

			# Calculate SDS score 
			if opt.mode == "tune":
				synthData = gen_synth(gen, opt.tune_n_valid_triples, printing=False)
				(score, _) = SDS(validData, synthData, printing=False)
				session.report({"score": (score)}, checkpoint=checkpoint)
	

	trainEnd = datetime.now()
	if real_epochs > 0 and (opt.mode != "tune" or tune_done):
		trainTime = (trainEnd - trainStart).total_seconds()
		print("Training time: " + "{:.0f}".format(trainTime) + " seconds")
		tpsTrain = (len(trainData)*real_epochs)/trainTime
		print("Average triples/s:" + "{:.0f}".format(tpsTrain) + "\n")





# --- Generating synthetic data ---
#flip key/value for dictionaries for fast decoding
entitiesRev = dict()
relationsRev = dict()
for key in entities.keys():
	value = entities[key]
	entitiesRev[value] = key
for key in relations.keys():
	value = relations[key]
	relationsRev[value] = key

def gen_synth(gen, num_triples = opt.out_n_triples, printing=True):
	# When raytune is used, the actual latent dim may differ from option
	real_latent_dim = 0
	if isinstance(gen.model[0], torch.nn.ConvTranspose1d):
		real_latent_dim = gen.model[0].in_channels
	else:
		real_latent_dim = gen.model[0].in_features

	cuda = opt.use_gpu and torch.cuda.is_available()
	device = 'cpu'
	if cuda: device = 'cuda:0'
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	gen.to(device)

	syntheticTriples = []
	genStart = datetime.now()

	columns = opt.tqdm_columns
	iters = range(ceil(num_triples/opt.batch_size))
	if printing:
		iters = tqdm(iters, ncols=columns, desc="gen")
	for i in iters:
		z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, real_latent_dim,))))

		start = datetime.now()

		triplesEnc = gen(z)

		mid = datetime.now()

		triples = []
		for tE in triplesEnc:
			triple = decode(tE, entitiesRev, entitiesN, relationsRev, relationsN)
			triples.append(triple)

		end = datetime.now()
		time1 = (mid - start).total_seconds()
		time2 = (end - mid).total_seconds()
		if printing:
			print(" - times: " + "{:.2f}".format(time1*1000) + "ms gen / " + "{:.2f}".format(time2*1000) + "ms decode", end='\r')

		for t in triples:
			syntheticTriples.append(t)
	
	genEnd = datetime.now()
	if printing:
		genTime = (genEnd - genStart).total_seconds()
		print("\nGeneration time: " + "{:.0f}".format(genTime) + " seconds", end="")
		tpsGen = (len(syntheticTriples))/genTime
		print("Average triples/s:" + "{:.0f}".format(tpsGen) + "\n")

	return syntheticTriples




# --- Data formatting & saving ---
def save_triples(triples):
	# make/overwrite generated files 
	nodesFile = open(path_join(genDir, "nodes.csv"), "w")
	edgesFile = open(path_join(genDir, "edges.csv"), "w")
	triplesFile = open(path_join(genDir, "triples.csv"), "w")
	triples_noRelFile = open(path_join(genDir, "triples_noRel.csv"), "w")

	# format & save data 
	#format data as nodes and edges
	nodes = entitiesRev
	edges = triples

	#save
	nodesFile.write("Id,Label,timeset,modularity_class\n")
	edgesFile.write("Source,Target,Type,Id,Label,timeset,Weight\n")
	for i in tqdm(range(len(nodes)), desc="save"):
		nodesFile.write(str(i) + "," + nodes[i] + ",,1\n")

	nextEdgeID = 0
	for triple in tqdm(edges, desc="save"):
		(h, r, t) = (triple.h, triple.r, triple.t)
		hID, tID = entities[h], entities[t]
		edgesFile.write(str(hID) + "," + str(tID) + ",Directed," + str(nextEdgeID) + "," + r + ",,1\n")
		nextEdgeID += 1
		triplesFile.write(h + "\t" + r + "\t" + t + "\n")
		triples_noRelFile.write(h + "\t" + t + "\n")

	nodesFile.close()
	edgesFile.close()
	triplesFile.close()





# --- model testing ---
def test_best_model(result):
	config = result.config
	print("\nRerunning best model")
	print(config)
	
	# define generator and discriminator
	generator	=		Generator(opt.latent_dim, entitiesN, relationsN)
	discriminator = Discriminator(opt.latent_dim, entitiesN, relationsN)
	generator.to(device)
	discriminator.to(device)

	#rerun best model
	opt.load_checkpoint = False
	opt.save_checkpoints = False
	train(config, generator, discriminator, Generator, Discriminator, tune_done=True)

	#generate new triples and save them
	synth_triples = gen_synth(generator)
	save_triples(synth_triples)

	#calculate metrics
	(SDSscore, _) = SDS(synth_triples, testData)
	print("SDS score: " + str(SDSscore))





# --- hyperparameter tuning ---
def main(config, num_samples=10, max_num_epochs=10, cpus_per_trial=1, gpus_per_trial=2):
	scheduler = ASHAScheduler(
		max_t=max_num_epochs,
		grace_period=1,
		reduction_factor=2
	)
	
	# define generator and discriminator
	generator	=		Generator(opt.latent_dim, entitiesN, relationsN)
	discriminator = Discriminator(opt.latent_dim, entitiesN, relationsN)
	generator.to(device)
	discriminator.to(device)

	gen = ray.put(generator)
	disc = ray.put(discriminator)
	genClass = ray.put(Generator)
	discClass = ray.put(Discriminator)

	tuner = tune.Tuner(
		tune.with_resources(
			tune.with_parameters(train, gen=gen, disc=disc, Gen=genClass, Disc=discClass),
			resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
		),
		tune_config=tune.TuneConfig(
			metric="score",
			mode="min",
			scheduler=scheduler,
			num_samples=num_samples,
			max_concurrent_trials=opt.max_concurrent_samples,
		),
		param_space=config,
	)
	results = tuner.fit()

	best_result = results.get_best_result("score", "min")
	
	print("Best trial config: {}".format(best_result.config))
	print("Best trial final validation loss: {}".format(
		best_result.metrics["score"]))
	
	#rerun best model
	test_best_model(best_result)

#potentially run raytune, otherwise just train once
syntheticTriples = []
if opt.mode == "tune":
	config = {
		#"l1":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		#"l2":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		"lr":			tune.loguniform(tune_lr_min, tune_lr_max),
		"batch_size":	tune.choice(tune_batch_sizes),
		"latent_dim":	tune.choice(tune_latent_dims),
		"n_critic":		tune.choice(tune_n_critics),
		"f_loss_min":	tune.loguniform(tune_f_loss_min_min, tune_f_loss_min_max),
	}
	main(config, num_samples=opt.tune_samples, max_num_epochs=opt.tune_max_epochs, cpus_per_trial=opt.tune_cpus, gpus_per_trial=opt.tune_gpus)
elif opt.mode == "run":
	config = {
		#"l1":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		#"l2":			tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
		"lr":			opt.lr,
		"batch_size":	opt.batch_size,
		"latent_dim":	opt.latent_dim,
		"n_critic":		opt.n_critic,
		"f_loss_min":	opt.f_loss_min,
	}	
	# define generator and discriminator
	generator	=		Generator(opt.latent_dim, entitiesN, relationsN)
	discriminator = Discriminator(opt.latent_dim, entitiesN, relationsN)
	generator.to(device)
	discriminator.to(device)

	train(config, generator, discriminator, Generator, Discriminator)
	
	# generate final synthetic data for run mode and save it
	syntheticTriples = gen_synth(generator)
	save_triples(syntheticTriples)





# --- Testing ---
if opt.mode != "tune":
	print("\nTesting:")
	(score, results) = (0, [])
	if opt.mode == "run":
		#test on newly generated data
		(score, results) = SDS(testData, syntheticTriples)
	elif opt.mode == "test":
		#test on generated data from last run
		# save_triples(genData) - used to generate relation-less versions of datasets for Gephi viewing
		if opt.use_sdmetrics:
			print("Calculating CategoricalCAP...")
			print("CategoricalCAP: " + str((CategoricalCAPHead(SDFullDBData, SDGenData)+CategoricalCAPTail(SDFullDBData, SDGenData))/2))
			print("Calculating NewRowSynthesis...")
			print("NewRowSynthesis:" + str(NewRowSynthesisTest(SDFullDBData, SDGenData, my_metadata_dict)))
			print("Calculating QualityReport...")
			ProduceQualityReport(SDTestData, SDGenData, my_metadata_dict)
		(score, results) = SDS(testData, genData)
	elif opt.mode == "dataTest":
		#test difference between test and validation data
		if opt.use_sdmetrics:
			print("Calculating CategoricalCAP...")
			print("CategoricalCAP: " + str((CategoricalCAPHead(SDFullDBData, SDValidData)+CategoricalCAPTail(SDFullDBData, SDValidData))/2))
			print("Calculating NewRowSynthesis...")
			print("NewRowSynthesis:" + str(NewRowSynthesisTest(SDFullDBData, SDValidData, my_metadata_dict)))
			ProduceQualityReport(SDTestData, SDValidData, my_metadata_dict)
		(score, results) = SDS(validData, testData)

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