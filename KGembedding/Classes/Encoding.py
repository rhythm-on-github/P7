from .Triple import *
import torch
import torch.nn as nn
import datetime 
from datetime import datetime
import numpy as np

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
def decode(triple:list, entities, entN, relations, relN):
	start = datetime.now()

	# Generally, relies on the encoding being ordered (hEnc, rEnc, tEnc)
	# Find the highest-value ID for the head, relation, and tail, individually but in the same way
	hID, rID, tID = 0, entN, entN+relN
	hVal, rVal, tVal = triple[0], triple[entN], triple[entN+relN]
	for (ID, Val, N) in [(hID, hVal, entN), (rID, rVal, relN), (tID, tVal, entN)]:
		# range is for within that relation's one-hot encoding

		#TODO: split vector in its three parts and try to see if argmax is faster
		# np.argmax(triple, axis=0, 
		for i in range(ID+1, ID+N):
			iVal = triple[i]
			if iVal > Val:
				ID = i
				Val = iVal

	mid = datetime.now()

	# Find the name of the encodings and relation based on their indices in the dictionary
	h = {key for key in entities if entities[key] == hID}
	r = {key for key in relations if relations[key] == rID}
	t = {key for key in entities if entities[key] == tID}

	end = datetime.now()
	loopTime = (mid - start).total_seconds()
	keyTime = (end - mid).total_seconds()
	print(" - times: " + "{:.2f}".format(loopTime) + "s loop / " + "{:.2f}".format(keyTime) + "s key", end='\r')

	dec = Triple(h, r, t)
	return dec