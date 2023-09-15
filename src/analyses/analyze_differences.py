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
    print(lang)
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

        tokens = list(dataset["spelling"])
        tokensplits = defaultdict(list)
        print(category)
        for model in dataset:
            print(model)
            if model.startswith("Model"):
                # splits in model output
                try:
                    _, modelname, vocabsize = model.split("_")

                    subtokens = list(dataset[model].apply(literal_eval))
                    num_splits = [len(x) - 1 for x in subtokens]
                    num_subtokens = [len(x) for x in subtokens]
                    max = len(dataset[model])
                    chunkability = [1 - (len(subtokens[i]) / len(str(tokens[i]))) for i in range(max)]

                    stats.append([lang, category, modelname, vocabsize, mean(num_splits), stdev(num_splits), mean(chunkability), stdev(chunkability)])


                except ValueError:
                    print("\nExcluding pre-trained model: " + model)
                    print()


        print(tokensplits)
        variances = [np.var([len(splits) for splits in subtokens]) for token, subtokens in tokensplits.items()]
        splitsframe = pd.DataFrame(zip(tokens, subtokens, variances), columns =["Stimulus", "Subtokens",  "Variance"])
        print(lang, category)
        print(splitsframe.nlargest(50, "Variance"))
        print(splitsframe.nsmallest(50, "Variance"))
        print()
results = pd.DataFrame(stats,
        columns =["Language", "Category", "Model", "Vocab_Size", "NumSplits_Mean", "NumSplits_Stdev", "Chunkability_Mean",
         "Chunkability_Stdev"])
results = results.astype( dtype={'Language' : str,
                 'Category': str,
                 'Model': str,
                 'Vocab_Size': int,
                 'NumSplits_Mean': float,
                 'NumSplits_Stdev': float,
                                 'Chunkability_Mean': float,
                                 'Chunkability_Stdev': float})
print(results.dtypes)

myplot = sns.lmplot(
    data=results, x="Vocab_Size", y="NumSplits_Mean",col="Language", hue="Model" ,row="Category")
#plt.title('Average Number of Splits')
plt.ylim(0, 4)
plt.show()
myplot.savefig(resultpath + "plots/avg_splits_by_vocabsize")


myplot2 = sns.lmplot(
    data=results, x="Vocab_Size", y="Chunkability_Mean",col="Language", hue="Model" ,row="Category")
# plt.title('Average Chunkability')
plt.ylim(0, 1)
plt.show()
myplot2.savefig(resultpath + "plots/chunkability_by_vocabsize")

