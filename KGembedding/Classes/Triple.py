class Triple(object):
    """A triple head, relation, tail (h,r,t) from a Knowledge Graph"""
    def __init__(self, h:str, r:str, t:str):
        self.h = h
        self.r = r
        self.t = t


