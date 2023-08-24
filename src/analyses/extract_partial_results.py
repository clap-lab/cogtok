
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Data
# results = pd.DataFrame()
# for language in ["en", "nl", "fr", "es"]:
#     language_results = pd.read_csv("../../results/results_overview/" + language + "_correlations.csv")
#     language_results["Language"] = language
#     results.append(language_results)
#
#
#     partial_data = language_results[language_results["Model"]=="BPE"]
#     partial_data = partial_data[partial_data["Vocab_Size"]=="50000"]
#     print(language)
#     print(partial_data.corr)
# words = results[results["Category"]=="Words"]
# nonwords = results[results["Category"]=="Non-Words"]


# Number of splits
results = pd.DataFrame()
for language in ["en", "nl", "fr", "es"]:
    language_results = pd.read_csv("../../results/results_overview/" + language + "_correlations.csv")
    language_results["Language"] = language
    results.append(language_results)
    partial_data = language_results.drop([])
    partial_data = partial_data[partial_data["Vocab_Size"]=="50000"]
    print(language)
    print(partial_data.corr)
words = results[results["Category"]=="Words"]
nonwords = results[results["Category"]=="Non-Words"]

# Calculate fourlingual results
# lang = "es"
# models = ["WPC_50000", "WPC_70000"]
# evalpath = "../../data/eval/"
# evaldata = pd.read_csv(open(evalpath + lang + ".txt", "r"), delimiter="\t")
# evaldata = evaldata.dropna()
# words = evaldata[evaldata["lexicality"] == "W"]
# nonwords = evaldata[evaldata["lexicality"] == "N"]
#
# outputs = {key: [] for key in evaldata["spelling"]}
# datasets = [words, nonwords]
# all_results = {}
# for category in ["words", "nonwords"]:
#
#     if category == "words":
#         dataset = datasets[0]
#     else:
#         dataset = datasets[1]
#
#     for model in models:
#         modeloutput = pd.read_csv("../../results/trained_models/fourlingual/models/" + model +"/" + lang + "_output.csv",delimiter=",", index_col=False)
#
#         segmentations = dict(zip(modeloutput["Stimulus"], modeloutput["Subtokens"]))
#
#         tokens = list(dataset["spelling"])
#         rts = list(dataset["rt"])
#         accs = list(dataset["accuracy"])
#         splits = []
#         # splits in model output
#         for t in tokens:
#             splits.append(segmentations[t])
#
#         num_splits = [len(x) - 1 for x in splits]
#         max = len(splits)
#         wordiness = [1 - (len(splits[i]) / len(str(tokens[i]))) for i in range(max)]
#
#         # correlation
#
#         corr3, p3 = pearsonr(wordiness, rts)
#         corr4, p4 = pearsonr(wordiness, accs)
#
#         results = [model, "{:.2f}".format(corr3), "{:.2f}".format(corr4)]
#         print(category)
#         print(results)
#         print(p3,p4)
#         print()