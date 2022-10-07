from .Triple import *
import torch
import torch.nn as nn
import datetime 
from datetime import datetime

# Allows the dataloaders to be encoded without massive space requirements
class Encoder(object):
	def __init__(self, data, entities, entN, relations, relN):
		self.data = data
		self.entities = entities
		self.entN = entN
		self.relations = relations
		self.relN = relN

	def __getitem__(self, tripleID:int):
		triple = self.data[tripleID]
		tripleEnc = encode(triple, self.entities, self.entN, self.relations, self.relN)
		return tripleEnc

	def __len__(self):
		return len(self.data)


# takes a triple (h,r,t) and encodes it with one-hot encoding
def encode(triple:Triple, entities, entN, relations, relN):
	hID = entities[triple.h]
	rID = relations[triple.r]
	tID = entities[triple.t]

	hEnc = nn.functional.one_hot(torch.arange(1) + hID, num_classes = entN)
	rEnc = nn.functional.one_hot(torch.arange(1) + rID, num_classes = relN)
	tEnc = nn.functional.one_hot(torch.arange(1) + tID, num_classes = entN)

	enc = torch.cat((hEnc, rEnc, tEnc), 1)
	return enc


# takes a one-hot encoded triple and converts it to (h,r,t)
# WARNING: VERY slow (O(n) but slower than encoding due to float comparisons), use as sparsely as possible 
def decode(triple:list, entitiesRev, entN, relationsRev, relN):
	start = datetime.now()

	# Generally, relies on the encoding being ordered (hEnc, rEnc, tEnc)

	#split vector in its three parts
	(hEnc, rEnc, tEnc) = torch.tensor_split(triple.cpu(), (entN, entN+relN), dim=0)

	#find index with highest value for each
	hID = torch.argmax(hEnc, dim=0).item()
	rID = torch.argmax(rEnc, dim=0).item()
	tID = torch.argmax(tEnc, dim=0).item()

	mid = datetime.now()

	# Find the name of the entities and relation based on their indices in the dictionary
	h = entitiesRev[hID]
	r = relationsRev[rID]
	t = entitiesRev[tID]

	end = datetime.now()
	loopTime = (mid - start).total_seconds()
	keyTime = (end - mid).total_seconds()
	print(" - times: " + "{:.2f}".format(loopTime) + "s loop / " + "{:.2f}".format(keyTime) + "s key", end='\r')

	dec = Triple(h, r, t)
	return dec