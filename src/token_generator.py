import itertools
from bert import tokenization
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module("https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1")
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
    return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

def addWord(word, pos, ner):
    """
    Convert a word into a word token and add supplied NER and POS labels. Note that the word can be  
    tokenized to two or more tokens. Correspondingly, we add - for now - custom 'X' tokens to the labels in order to 
    maintain the 1:1 mappings between word tokens and labels.
    
    arguments: word, pos label, ner label
    returns: dictionary with tokens and labels
    """
    # the dataset contains various '"""' combinations which we choose to truncate to '"', etc. 
    if word == '""""':
        word = '"'
    elif word == '``':
        word = '`'
        
    tokens = tokenizer.tokenize(word)
    tokenLength = len(tokens)      # find number of tokens corresponfing to word to later add 'X' tokens to labels
    
    addDict = dict()
    
    addDict['wordToken'] = tokens
    addDict['posToken'] = [pos] + ['[posX]'] * (tokenLength - 1)
    addDict['nerToken'] = [ner] + ['[nerX]'] * (tokenLength - 1)
    addDict['tokenLength'] = tokenLength
    
    return addDict

class TokenizedSentence:

    def __init__(self, sentence, max_length):

        sentenceTokens = ['[CLS]']
        posTokens = ['[posCLS]']
        nerTokens = ['[nerCLS]']

        raceTokens = ['[raceCLS]']
        genderTokens = ['[genderCLS]']

        nameMask = [0]

        for index, word in enumerate(sentence.words):

            addDict = addWord(word.txt, word.pos, word.named_entity_tag)
            sentenceTokens += addDict['wordToken']
            posTokens += addDict['posToken']
            nerTokens += addDict['nerToken']

            tokenLength = addDict['tokenLength']
            
            raceTokens += [sentence.race]*tokenLength
            genderTokens += [sentence.gender]*tokenLength

            if index == sentence.name_index:                
                nameMask += [1]*tokenLength

            else:
                nameMask += [0]*tokenLength

        sentenceLength = min(max_length -1, len(sentenceTokens))
            
        # Create space for at least a final '[SEP]' token
        if sentenceLength >= max_length - 1: 
            sentenceTokens = sentenceTokens[:max_length - 2]
            posTokens = posTokens[:max_length - 2]
            nerTokens = nerTokens[:max_length - 2]
            raceTokens = raceTokens[:max_length - 2]
            genderTokens = genderTokens[:max_length - 2]
            nameMask = nameMask[:max_length - 2]

        sentenceTokens += ['[SEP]'] + ['[PAD]'] * (max_length -1 - len(sentenceTokens))
        posTokens += ['[posSEP]'] + ['[posPAD]'] * (max_length - 1 - len(posTokens) )
        nerTokens += ['[nerSEP]'] + ['[nerPAD]'] * (max_length - 1 - len(nerTokens) )
        raceTokens += ['[raceSEP]'] + ['[racePAD]'] * (max_length - 1 - len(raceTokens) )
        genderTokens += ['[genderSEP]'] + ['[genderPAD]'] * (max_length - 1 - len(genderTokens) )
        nameMask += [0] + [0] * (max_length - 1 - len(nameMask) )

        self.sentenceIDs = tokenizer.convert_tokens_to_ids(sentenceTokens)
        self.masks = [1] * (sentenceLength + 1) + [0] * (max_length -1 - sentenceLength )
        self.sequenceIDs = [0] * (max_length)

        self.nerTokens = nerTokens
        self.posTokens = posTokens
        
        self.raceTokens = raceTokens
        self.genderTokens = genderTokens

        self.nameMask = nameMask

def getLabels(tokens):

    classes = pd.DataFrame(np.array(tokens).reshape(-1))
    classes.columns = ['tag']
    classes.tag = pd.Categorical(classes.tag)
    classes['cat'] = classes.tag.cat.codes
    classes['sym'] = classes.tag.cat.codes
    labels = np.array(classes.cat).reshape(len(tokens), -1) 

    distribution = (classes.groupby(['tag', 'cat']).agg({'sym':'count'}).reset_index()
                   .rename(columns={'sym':'occurences'}))


    print(distribution)
    print()

    return labels

def getTokensAndLabelsFromSents(sents, max_length):

    sentenceIDs = []
    masks = []
    sequenceIDs = []

    nerTokens = []
    posTokens = []

    raceTokens = []
    genderTokens = []

    nameMasks = []

    for sent in tqdm(sents):
        tokenized_sent = TokenizedSentence(sent, max_length)
        
        sentenceIDs.append(tokenized_sent.sentenceIDs)
        masks.append(tokenized_sent.masks)
        sequenceIDs.append(tokenized_sent.sequenceIDs)

        nerTokens.append(tokenized_sent.nerTokens)
        posTokens.append(tokenized_sent.posTokens)

        raceTokens.append(tokenized_sent.raceTokens)
        genderTokens.append(tokenized_sent.genderTokens)
        nameMasks.append(tokenized_sent.nameMask)

    nerLabels = getLabels(nerTokens)
    #posLabels = getLabels(posTokens)

    raceLabels = getLabels(raceTokens)
    genderLabels = getLabels(genderTokens)

    return sentenceIDs, masks, sequenceIDs, nerLabels, raceLabels, genderLabels, nameMasks




