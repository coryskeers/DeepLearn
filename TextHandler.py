import re
import numpy as np

class Corpus:

    class Word:
        def __init__(self, name, phonemes=[]):
            self.name=name
            self.phonemes=phonemes

#############################
#### Constructor Methods ####
#############################
    
    def __init__(self, CMU_PHONEMES=True, ADD_TO_CORPUS=True):
        self.punct = re.compile(r'''([?!.",;:]+)''')
        self.sentenceQueue = []
        self.phonemeQueue = []
        self.fullCorpus = {}
        self.indexer = {}
        self.corpLength = 0
        if CMU_PHONEMES:
            self.buildPhonemes()
        self.CMU_PHONEMES = CMU_PHONEMES
        self.ADD_TO_CORPUS = ADD_TO_CORPUS
        self.sentSplitter = re.compile(r' [.!?]+ ')
        

    def buildPhonemes(self):
        print("Building phoneme list")
        with open("cmu dict.txt") as infile:
            phonemeList = [line.strip().split('  ',1) for line in infile if line[0] != ';']
        print("Phoneme list complete")
        phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'D', 'G', 'K', 'P', 'T', 'W', 'Y', 'M', 'N', 'NG', 'L', 'R', 'DH', 'F', 'S', 'SH', 'TH', 'V', 'Z', 'ZH', 'HH', 'CH', 'JH']
        syllables = ['', '0', '1', '2']
        self.phonemes = [p + s for p in phonemes for s in syllables]
        self.phonemes.append("|")
        i = 0
        for entry in phonemeList:
            word = entry[0].lower()
            self.fullCorpus[word] = self.Word(word, entry[1])
            self.indexer[word] = i
            i += 1
            self.corpLength += 1
        print("Phoneme dictionary complete")

    def wordNoise(self, words):
        s="abcdefghijklmnopqrstuvwxyz"
        for word in words:
            if np.random.randn() > 0:
                i = np.random.randint(0,len(word))
                char = s[np.random.randint(0,26)]
                word = word[:i] + char + word[i:]
        return words

    def phonemeNoise(self, phones):
        if np.random.randn() > 0.5:
            i = np.random.randint(0,len(phones))
            phone = self.phonemes[np.random.randint(0,len(self.phonemes))]
            phones[i] = phone
        return phones
    
##########################
#### Standard Methods ####
##########################

    def addSentences(self, string):
        l = self.sentSplitter.split(self.punct.sub(' \\1 ', string.lower()))
        for i in range(len(l) - 1): #extra, empty sentence post final period.
            sent = l[i].split()
            phones = []
            if self.ADD_TO_CORPUS:
                for word in sent:
                    if word not in self.fullCorpus:
                        self.fullCorpus[word] = self.Word(word, "")
                    else:
                        phones.append(self.fullCorpus[word].phonemes)
                        phones.append(" | ")
            self.sentenceQueue.append(sent)
            if len(phones) > 0:
                self.phonemeQueue.append(' '.join(phones))
        print("Lines added.")
        print(len(self.sentenceQueue), "sentences in queue.")
        print(len(self.phonemeQueue), "phoneme strings in queue.")

    def phonemeMatrix(self, ADD_NOISE=True):
        if len(self.phonemeQueue) == 0:
            return -1
        sent = self.phonemeQueue.pop(0).split()
        if ADD_NOISE:
            sent = self.phonemeNoise(sent)
        matrix = np.zeros((len(sent), len(self.phonemes)))
        for i in range(len(sent)):
            matrix[i][self.phonemes.index(sent[i])] = 1
        return matrix

    def sentenceMatrix(self, ADD_NOISE=True):
        if len(self.sentenceQueue) == 0:
            return -1
        sent = self.sentenceQueue.pop(0)
        if ADD_NOISE:
            sent = self.wordNoise(sent)
        matrix = np.zeros((len(sent), self.corpLength))
        for i in range(len(sent)):
            matrix[i][self.indexer[sent[i]]] = 1
        return matrix
