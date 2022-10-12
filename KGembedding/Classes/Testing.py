import torch
from tqdm import tqdm

from .Triple import *

"""
 Idea: Use statistical analysis, e.g. P((x,isA,y) | (x,isA,z))
Types of things to analyze:
	1: P((x,y,a) | (x,y,b))
	2: P((x,a,_) | (x,b,_))
	3: P((_,a,z) | (_,b,z))
	4: P((a,y,z) | (b,y,z))
	5: P((x,a,_) | (x,b,z))
(for each node / some subset of nodes (x,y,z), where a and b are placeholders for any existing node/edge)
Hypothesis 1: number 2-3 are most interesting and most feasibly implementable 

For now, testing is done with FB15K237 (test as train), simpGAN, 1/0 epochs, 5000 output triples
Initially the score was supposed to be the sum, but preliminary tests showed average to be more meaningful

---------------------------
SDS v0.1: only test P((x,a,_) | (x,b,_))
(lower = better)
1 epoch avgs:
0.33
0.35
0.31

0 epochs avgs:
0.1
0.13
0.08

Judgement: Since the graph for 1 epoch looks more inline with the dataset,
this version doesn't judge well.
Hypothesis 2: relations such as P((x,a,_) | (x,b,_)) can only be accurately
judged assuming the data is already of higher quality than we have. Therefore,
a simpler analysis should be employed before this.

-------------------------
SDS v0.2: test P((_,r,_)) and P((x,a,_) | (x,b,_))
(lower = better, but the former of the two is prioritised)
1 epoch avgs:
(0.33, 0.27)
(0.32, 0.39)
(0.29, 0.26)
(0.33, 0.18)
(0.35, 0.24)

0 epochs avgs:
(0.36, 0.15)
(0.36, 0.09)
(0.34, 0.14)
(0.35, 0.14)
(0.36, 0.15)

This seems to confirm hypothesis 2, although the data does seem somewhat sporadic.
Therefore, the number of output triples is increased to 10k goinng onwards (from 5k) and this test is repeated.
1 epoch avgs:
(0.32, 0.31)
(0.34, 0.43)
(0.34, 0.41)
(0.33, 0.39)
(0.31, 0.31)

0 epochs avgs:
(0.42, 0.20)
(0.41, 0.17)
(0.40, 0.11)
(0.45, 0.16)
(0.43, 0.19)

This more definitively confirms hypothesis 2, as there is a consistent disparity in the expected direction.
Running a test with 5 epochs yielded a score of (0.45, 0.44), which is consistent with Gephi showing clear overfitting.
Furthermore, a test with 1 poch on FB15K237 actual train data gave (0.55, NaN), consistent with Gephi showing extreme overfitting.
Judgement: SDS v0.2 is usable, and the current NN/hyperparameters lead to extreme overfitting on larger datasets. 
Hypothesis 3: the batch size is too large (128), leading to overfitting
Test: Change the batch size to 64/32/16
From here, the training dataset is also changed to FB15K237 actual train data, and it is only tested for 1 epoch.
Results:
batch size 64: (0.55, NaN) but 10 instead of 7 nodes used
batch size 32: (0.54, NaN) but 3 instead of 7 nodes used
batch size 16: (0.52, NaN) but 2 instead of 7 nodes used

Since the SDS has consistently improved as batch size decreases, we will also attempt lower batch sizes. The number of nodes used has worsened, but that may be a fluke.
batch size 8: (0.52, NaN) but 3 instead of 7 nodes used
batch size 4: (0.55, NaN) but 2 instead of 7 nodes used

With the additional results, it seems that the lower nodes used is not a fluke. With this in mind, the ideal batch size (given the other hyperparameters) seems to be 64.
However, a hyperparameter tuner could also be used, so that will be tested as soon as it can be implemented. 
"""

def SDS(A: [Triple], B: [Triple]):
	""" Statistical Disagreement Score
	Takes two KGs  (intended for a real and a generated)
    Gives back a score of the sum of difference between statistical predictions for A and B
	Assumes that every node has a unique name (maybe has to, given input is just a list of triples)
	"""
	print("Calculating SDS...")
	results = []

	# --- calculate P((_,r,_)) ---
	# count up how many times each relation appears in A and B, and which relations there are
	Acount = dict()
	Bcount = dict()
	relations = set()
	for (count, data) in [(Acount, A), (Bcount, B)]:
		for triple in data:
			#counting
			r = triple.r
			if r not in count:
				count[r] = 1
			else:
				count[r] += 1
			#relations - since it is a set, will not contain copies
			relations.add(r)

	# calculate the SDS for P((_,r,_))
	sum = 0
	n = len(relations)
	for r in relations:
		if r in Acount.keys():
			if r in Bcount.keys():
				#calculate difference
				sum += abs(Acount[r]-Bcount[r]) #/n
			else:
				sum += Acount[r] #/n
		else:
			if r in Bcount.keys():
				sum += Bcount[r] #/n
	sum = sum/n #do division last as optimisation, since all the amounts added to sum need to be divided by n
	results.append(("P((_,r,_))", n, sum))



	# --- calculate P((x,a,_) | (x,b,_)) ---
	#make dictionaries of nodes' outgoing edges
	Ax = dict()
	Bx = dict()
	for (dictionary, X) in [(Ax, A), (Bx, B)]:
		for t in X:
			if t.h not in dictionary.keys():
				dictionary[t.h] = [t.r]
			else: 
				if t.r not in dictionary[t.h]:
					dictionary[t.h].append(t.r)

	#calculate triples (x,y,_) present in both A and B
	C = []
	for h in tqdm(Ax.keys(), desc="C"):
		for r in Ax[h]:
			if h in Bx.keys():
				if r in Bx[h] and (h, r, "") not in C:
					C.append((h, r, ""))

	#calculate unique x and y values in C
	xC = []
	yC = []
	for (x,y,z) in C:
		if x not in xC:
			xC.append(x)
		if y not in yC:
			yC.append(y)

	#calculate the SDS for P((x,a,_) | (x,b,_)) (ca. O(n^2) time in testing)
	sum = 0;
	n = 0;
	for a in tqdm(yC, desc="sum"):
		for b in yC:
			#count up number of nodes in A with name x that have (both a and b) and (b)
			Axab = 0;
			Axb = 0;
			for x in Ax.keys():
				if b in Ax[x]:
					Axb += 1
					if a in Ax[x]:
						Axab += 1
			#count up number of nodes in B with name x that have (both a and b) and (b)
			Bxab = 0;
			Bxb = 0;
			for x in Bx.keys():
				if b in Bx[x]:
					Bxb += 1
					if a in Bx[x]:
						Bxab += 1
			#calculate difference and add to sum
			sum += abs((Axab/Axb)-(Bxab/Bxb))
			n += 1;

	results.append(("P((x,a,_) | (x,b,_))", n, sum))
	return results