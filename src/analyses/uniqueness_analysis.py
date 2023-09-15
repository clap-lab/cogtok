import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os
from ast import literal_eval
from statistics import mean, stdev
import seaborn as sns
from tokenization_util import train_tokenizer, Tokenizer, tokenize

lexicon_file = "../../data/uniqueness/lexicon_fasttext_wiki_en.txt"


tokens = []
min_length = 4
with open(lexicon_file) as file:
    for line in file:
        token = line.rstrip()
        # Could use some more filtering
        if len(token)>=min_length and not str.isnumeric(token):
            tokens.append(token.lower())


models = []

tokensplits = {}
modelpath = "../../results/trained_models/en/models/"
resultpath = "../../results/"

models  = ["BPE_50000", "WPC_50000", "UNI_50000"]
tokenizers = [Tokenizer.from_file(modelpath + model+ "/trained.json") for model in models]

# transform into set to remove duplicates
for token in set(tokens):

    subtokens = [tokenize(token, tokenizer).tokens for tokenizer in tokenizers]
    tokensplits[token] = subtokens
    print(token, subtokens)

splitsframe = pd.DataFrame(tokensplits.items(), columns=["Token", "Subtokens [BPE, WPC, UNI]"],index=None)
print(splitsframe)
splitsframe.to_csv(resultpath +"instancelevel/english_splits_fasttext.csv")

# For each model: make a counter
# count for each subtoken how often it is used for splitting a token (how to deal with subtokens that occur twice for a token? for example [m, elka, m]: do we count 1 or 2 for subtoken "m')

# for token, subtokens
    # for subtoken in subtokens
    # look up number of other words it appears in
    # calculate score (e.g. minimum?)