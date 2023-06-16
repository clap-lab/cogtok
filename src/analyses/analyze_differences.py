import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os

# I started some analyses here how the output of two trained_models differs.
# I did not yet use this systematically

output1file = "../../results/en/trained_models/bert-base-uncased/output.csv"
output2file = "../../results/en/trained_models/bert-base-multilingual-uncased/output.csv"

output1 = pd.read_csv(output1file, delimiter=",")
output2 = pd.read_csv(output2file, delimiter=",")


# Statistics
for tokenized in [output1, output2]:
    nonwords = tokenized[tokenized["Lexicality"] == "N"]
    words = tokenized[tokenized["Lexicality"] == "W"]

    print("Words:")
    print(words.describe())

    print("\nNon-Words:")
    print(nonwords.describe())
    print("\n-------------------\n")

splits1 = dict(zip(output1['Stimulus'], output1['Subtokens']))
splits2 = dict(zip(output2['Stimulus'], output2['Subtokens']))

samesplits = 0
samenumsplits = 0
for key, value1 in splits1.items():
    value2 = splits2[key]

    if len(value1) == len(value2):
        samenumsplits += 1
        if value1 == value2:
            samesplits += 1
        else:
            print(value1, value2)

# TODO: distinguish between words and nonwords here!
print("Same splits: {:.2f}".format(samesplits/len(splits1)))
print("Same number of splits: {:.2f}".format(samenumsplits/len(splits1)))
