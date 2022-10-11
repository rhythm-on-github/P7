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
Hypothesis: number 2-3 are most interesting and most feasibly implementable 

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
New hypothesis: relations such as P((x,a,_) | (x,b,_)) can only be accurately
judged assuming the data is already of higher quality than we have. Therefore,
a simpler analysis should be employed before this.

-------------------------
SDS v0.2: test P((_,r,_)) and P((x,a,_) | (x,b,_))
(lower = better, but the former of the two is prioritised)
"""

def SDS(A: [Triple], B: [Triple]):
	""" Statistical Disagreement Score
	Takes two (intended for a real and a generated) KGs
    Gives back a score of the sum of difference between statistical predictions for A and B
	Assumes that every node has a unique name (maybe has to, given input is just a lsit of triples)
	"""
	print("Calculating SDS...")

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

	#calculate SDS (ca. O(n^2))
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

	return (n, sum)