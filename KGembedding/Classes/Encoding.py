from .Triple import *
import torch
import torch.nn as nn

# Allows the dataloaders to be encoded without massive space requirements
class Encoder(object):
	def __init(self, data, entities, entN, relations, relN):
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

	enc = torch.cat((hEnc, rEnc, tEnc))
	return enc


# takes a one-hot encoded triple and converts it to (h,r,t)
# WARNING: probably pretty damn slow (O(n) but slower than encoding due to float comparisons), use as sparsely as possible 
def decode(triple:list, entities, entN, relations, relN):
	# Generally, relies on the encoding being ordered (hEnc, rEnc, tEnc)
	# Find the highest-value ID for the head, relation, and tail, individually but in the same way
	hID, rID, tID = 0, entN, entN+relN
	hVal, rVal, tVal = triple[0], triple[entN], triple[entN+relN]
	for (ID, Val, N) in [(hID, hVal, entN), (rID, rVal, relN), (tID, tVal, entN)]:
		# range is for within that relation's one-hot encoding
		for i in range(ID+1, ID+N):
			Val = triple[i]
			if Val > Val:
				ID = i
				Val = iVal
	# Find the name of the encodings and relation based on their indices in the dictionary
	h = {key for key in entities if entities[key] == hID}
	r = {key for key in relations if relations[key] == rID}
	t = {key for key in entities if entities[key] == tID}

	dec = Triple(h, r, t)
	return dec