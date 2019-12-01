import itertools
import copy
import flair 
import numpy as np
import pandas as pd
import random
import pickle
import os

from sklearn.utils import shuffle

import token_generator

dem_data = pd.read_csv("../data/raw/WEATdata.csv")

class Word:
    def __init__(self, wordArray):

        word = wordArray[0]

        if word == '""""':
            word = '"'
        elif word == '``':
            word = '`'
        
        nerTag = wordArray[3]

        #hack - only train data has B-LOC
        if (nerTag == "B-LOC"):
            nerTag = "I-LOC"

        self.txt = word
        self.origTxt = word
        self.pos = wordArray[1]
        self.chunk_tag = wordArray[2]
        self.named_entity_tag = nerTag
        
    def isPerson(self):
        return self.named_entity_tag=='I-PER'

class Sentence:
    def __init__(self, sentArray):
        self.words = []
        for index, wordArray in enumerate(sentArray):   
            new_word = Word(wordArray)
            if (new_word.isPerson()):
                if(len(self.words)==0 or not self.words[-1].isPerson()):
                    self.words.append(new_word)
            else:
                 self.words.append(new_word)

    def getText(self):
        return_str = ""
        for index, word in enumerate(self.words):
            if word.pos in ['POS','"','.',','] or index==0:
                return_str += word.txt
            else:
                return_str += ' ' + word.txt
                
        return return_str

    def getNameIndices(self):
        return [index for index, word in enumerate(self.words) if word.isPerson()]
    
    def mask(self, index, maskName):
        self.words[index].txt = maskName
    
    def setSentiment(self, label):
        self.sentiment = label[0].to_dict()

def calculateSentiments(sents):
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    flair_sentences = [flair.data.Sentence(sent.getText()) for sent in sents]
    flair_sentiment.predict(flair_sentences)
    for i, sent in enumerate(sents):
        sent.setSentiment(flair_sentences[i].labels)

def augmentSentencesWithDemData(sents):

    augmented_sents = []

    for sent in sents:
        for name_index, word in enumerate(sent.words):
            if word.txt == "[NAME]":
                for index, row in dem_data.iterrows():
                    word.txt = row["Person"]
                    sent.new_name = row["Person"]
                    sent.race = row["Race"].upper()
                    sent.gender = row["Gender"].upper()
                    sent.name_index = name_index
                    sent_copy = copy.deepcopy(sent)
                    augmented_sents.append(sent_copy)

    return augmented_sents

def getNamedSentences(filename, sample, force_regenerate):

    save_filename = filename.replace('raw/CoNLL-2003','processed') + ".augmented_sents.pkl"

    if (not force_regenerate and os.path.exists(save_filename)):
        with open(save_filename, 'rb') as savefile:
            augmented_sents = pickle.load(savefile)

    else:

        alist = [line.rstrip() for line in open(filename)][1:]
        sents = [list(g) for k,g in itertools.groupby(alist, key=lambda x: x != '') if k]

        data = [[word.split() for word in sent] for sent in sents]
        named_data = []

        for sent in data:
            for word in sent:
                if word[-1]=='I-PER':
                    named_data.append(sent)
                    break
        
        named_sentence_generator = (Sentence(sentArray) for sentArray in named_data)

        masked_sents = []

        for sentence in named_sentence_generator:
            if len(sentence.words)>1:
                nameIndices = sentence.getNameIndices()
                for index in nameIndices:
                    sent_copy = copy.deepcopy(sentence)
                    sent_copy.mask(index, "[NAME]")
                    masked_sents.append(sent_copy)

        calculateSentiments(masked_sents)

        augmented_sents = augmentSentencesWithDemData(masked_sents)

        with open(save_filename, 'wb') as savefile:
            pickle.dump(augmented_sents, savefile)

    if sample:
        augmented_sents = random.sample(augmented_sents, sample)

    random.shuffle(augmented_sents)

    return augmented_sents

def generateSents(sample, force_regenerate):

    train_data_path="../data/raw/CoNLL-2003/eng.train"
    val_data_path="../data/raw/CoNLL-2003/eng.testa"
    test_data_path="../data/raw/CoNLL-2003/eng.testb"

    return [getNamedSentences(filepath, sample, force_regenerate) for filepath in [
        train_data_path,val_data_path,test_data_path
    ]]

def GetData(max_length, sample=None, force_regenerate=None):
    train_sents, val_sents, test_sents = generateSents(sample, force_regenerate)

    train_data_start = 0
    train_data_end = val_data_start = len(train_sents)
    val_data_end = test_data_start = val_data_start + len(val_sents)
    test_data_end = test_data_start + len(test_sents)

    all_sents = train_sents + val_sents + test_sents

    sentenceIDs, \
    masks, \
    sequenceIDs, \
    nerLabels, \
    raceLabels, \
    genderLabels, \
    nameMasks = token_generator.getTokensAndLabelsFromSents(all_sents, max_length)

    sentenceIDs_train = np.array(sentenceIDs[train_data_start:train_data_end])
    masks_train = np.array(masks[train_data_start:train_data_end])
    sequenceIDs_train = np.array(sequenceIDs[train_data_start:train_data_end])
    inputs_train = [sentenceIDs_train, masks_train, sequenceIDs_train]
    nerLabels_train = nerLabels[train_data_start:train_data_end]
    genderLabels_train = genderLabels[train_data_start:train_data_end]
    raceLabels_train = raceLabels[train_data_start:train_data_end]
    nameMasks_train = nameMasks[train_data_start:train_data_end]

    sentenceIDs_val = np.array(sentenceIDs[val_data_start:val_data_end])
    masks_val = np.array(masks[val_data_start:val_data_end])
    sequenceIDs_val = np.array(sequenceIDs[val_data_start:val_data_end])
    inputs_val = [sentenceIDs_val, masks_val, sequenceIDs_val]
    nerLabels_val = nerLabels[val_data_start:val_data_end]
    genderLabels_val = genderLabels[val_data_start:val_data_end]
    raceLabels_val = raceLabels[val_data_start:val_data_end]
    nameMasks_val = nameMasks[val_data_start:val_data_end]

    sentenceIDs_test = np.array(sentenceIDs[test_data_start:test_data_end])
    masks_test = np.array(masks[test_data_start:test_data_end])
    sequenceIDs_test = np.array(sequenceIDs[test_data_start:test_data_end])
    inputs_test = [sentenceIDs_test, masks_test, sequenceIDs_test]
    nerLabels_test = nerLabels[test_data_start:test_data_end]
    genderLabels_test = genderLabels[test_data_start:test_data_end]
    raceLabels_test = raceLabels[test_data_start:test_data_end]
    nameMasks_test = nameMasks[test_data_start:test_data_end]

    full_train_data = {
        "inputs": inputs_train,
        "nerLabels": nerLabels_train,
        "genderLabels": genderLabels_train,
        "raceLabels": raceLabels_train,
        "nameMasks": nameMasks_train,
        "sentences": train_sents
    }

    full_val_data = {
        "inputs": inputs_val,
        "nerLabels": nerLabels_val,
        "genderLabels": genderLabels_val,
        "raceLabels": raceLabels_val,
        "nameMasks": nameMasks_val,
        "sentences": val_sents
    }

    full_test_data = {
        "inputs": inputs_test,
        "nerLabels": nerLabels_test,
        "genderLabels": genderLabels_test,
        "raceLabels": raceLabels_test,
        "nameMasks": nameMasks_test,
        "sentences": test_sents
    }

    return full_train_data, full_val_data, full_test_data