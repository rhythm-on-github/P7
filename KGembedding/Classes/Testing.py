import torch

# Idea: Use statistical analysis, e.g. P((x,isA,y) | (x,isA,z))
#Types of things to analyze:
	# P((x,y,a) | (x,y,b)) - chance of a given node x having edge y to two different nodes
	# P((x,a,_) | (x,b,_))
	# P((_,a,z) | (_,b,z))
	# P((a,y,z) | (b,y,z))
#(for each node / some subset of nodes (x,y,z), where a and b are placeholders for any existing node/edge)