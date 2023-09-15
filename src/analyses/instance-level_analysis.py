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
# I started some analyses here how the output of two trained_models differs.
# I did not yet use this systematically

# output1file = "../../results/en/trained_models/bert-base-uncased/output.csv"
# output2file = "../../results/en/trained_models/bert-base-multilingual-uncased/output.csv"
#
# output1 = pd.read_csv(output1file, delimiter=",")
# output2 = pd.read_csv(output2file, delimiter=",")
#
#
# # Statistics
# for tokenized in [output1, output2]:
#     nonwords = tokenized[tokenized["Lexicality"] == "N"]
#     words = tokenized[tokenized["Lexicality"] == "W"]
#
#     print("Words:")
#     print(words.describe())
#
#     print("\nNon-Words:")
#     print(nonwords.describe())
#     print("\n-------------------\n")
#
# splits1 = dict(zip(output1['Stimulus'], output1['Subtokens']))
# splits2 = dict(zip(output2['Stimulus'], output2['Subtokens']))
#
# samesplits = 0
# samenumsplits = 0
# for key, value1 in splits1.items():
#     value2 = splits2[key]
#
#     if len(value1) == len(value2):
#         samenumsplits += 1
#         if value1 == value2:
#             samesplits += 1
#         else:
#             print(value1, value2)
#
# # TODO: distinguish between words and nonwords here!
# print("Same splits: {:.2f}".format(samesplits/len(splits1)))
# print("Same number of splits: {:.2f}".format(samenumsplits/len(splits1)))


# How does number of splits change with respect to vocab size

langs = ["en", "nl", "fr", "es"]

resultpath = "../../results/"
stats = []

for lang in langs:

    evaldata = pd.read_csv(resultpath + "results_overview/" + lang + ".csv")
    evaldata = evaldata.dropna()
    words = evaldata[evaldata["lexicality"] == "W"]
    nonwords = evaldata[evaldata["lexicality"] == "N"]

    datasets = [words, nonwords]


    for category in ["Words", "Non-Words"]:

        if category == "Words":
            dataset = datasets[0]
        else:
            dataset = datasets[1]


        models = []
        for col in dataset:

            if col.startswith("Model_WPC"):
                try:
                    _, modelname, vocabsize = col.split("_")
                    models.append(col)
                except ValueError:
                    print("Excluding pretrained model: " + col)
        tokensplits = defaultdict(list)

        splitvariance = []
        for token in dataset["spelling"]:

            # todo sort this by model and vocab size
            subtokens = [list(dataset.loc[dataset.spelling == token, model].apply(literal_eval))[0] for model in models ]
            lengths = [len(splits) for splits in subtokens]
            stdev = np.std(lengths)
            splitvariance.append([token, stdev, subtokens])




        splitsframe = pd.DataFrame(splitvariance, columns =["Stimulus",  "Stdev", "Subtokens"] )
        print(lang, category)
        splitsframe.nlargest(100, "Stdev").to_csv(resultpath +"instancelevel/"+ lang + "_" + category + "_highest_splitvariance.csv")
        print()
