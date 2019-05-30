import re
import numpy as np
from TextHandler import Corpus

c=Corpus()
s="abcdefghijklmnopqrstuvwxyz"

class sent2phone:
    def __init__(self):
        pass

    def addNoise(self, word):
        if np.random.randn() > 0:
            i = np.random.randint(0,len(word))
            char = s[np.random.randint(0,26)]
            word = word[:i] + char + word[i:]
        return word

class phone2sent:
    def __init__(self):
        pass
