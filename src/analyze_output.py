import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config["vocab_size"])
    outdir = config["outdir"]

    languages = config["language"]
    algorithms = config["tokenizers"]
    vocab_sizes = config["vocab_size"]

# Calculate correlations and plot
for language in languages:
    results = {}
    measure1 = "Reading_Time"
    measure2 = "Accuracy"
    feature1 = "Num_Splits"
    feature2 = "Wordiness"
    length_correlations = {}


    for vocab_size in vocab_sizes:
        for alg in algorithms:
            setting = alg + "_" + str(vocab_size)
            modeldir = outdir + language + "/models/" + setting + "/"
            outputdata = pd.read_csv(modeldir + "output.csv", delimiter=",", index_col=0)

            nonwords = outputdata[outputdata["Lexicality"] == "N"]
            words = outputdata[outputdata["Lexicality"] == "W"]

            # Correlation with length is the same for all algorithms and vocab sizes
            if len(length_correlations) == 0:
                length_correlations["lc_w1"] = words[measure1].corr(words["Length"])
                length_correlations["lc_w2"]  = words[measure2].corr(words["Length"])
                length_correlations["lc_nw1"]  = nonwords[measure1].corr(nonwords["Length"])
                length_correlations["lc_nw2"]  = nonwords[measure2].corr(nonwords["Length"])

            correlations = {}
            correlations["c_w11"] = words[measure1].corr(words[feature1])
            correlations["c_w12"] = words[measure1].corr(words[feature2])
            correlations["c_w21"] = words[measure2].corr(words[feature1])
            correlations["c_w22"] = words[measure2].corr(words[feature2])

            correlations["c_nw11"] = nonwords[measure1].corr(nonwords[feature1])
            correlations["c_nw12"] = nonwords[measure1].corr(nonwords[feature2])
            correlations["c_nw21"] = nonwords[measure2].corr(nonwords[feature1])
            correlations["c_nw22"] = nonwords[measure2].corr(nonwords[feature2])

            results[setting] = correlations
            print(setting)
            print(correlations)
            print()

    # TODO: adding the morfessor correlations at the end

    outputdata = pd.read_csv("../morfessor/models/" + language + "/output.csv", delimiter=",", index_col=0)
    nonwords = outputdata[outputdata["Lexicality"] == "N"]
    words = outputdata[outputdata["Lexicality"] == "W"]
    correlations = {}
    correlations["c_w11"] = words[measure1].corr(words[feature1])
    correlations["c_w12"] = words[measure1].corr(words[feature2])
    correlations["c_w21"] = words[measure2].corr(words[feature1])
    correlations["c_w22"] = words[measure2].corr(words[feature2])

    correlations["c_nw11"] = nonwords[measure1].corr(nonwords[feature1])
    correlations["c_nw12"] = nonwords[measure1].corr(nonwords[feature2])
    correlations["c_nw21"] = nonwords[measure2].corr(nonwords[feature1])
    correlations["c_nw22"] = nonwords[measure2].corr(nonwords[feature2])
    results["morfessor_unsup"] = correlations

    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    fig.suptitle(language)
    custom_ylim = (-0.6, 0.6)
    plt.setp(axs, ylim=custom_ylim)

    mycolors = ['navy', 'cornflowerblue', 'mediumpurple'] * len(vocab_sizes)
    mycolors.append("orange")

    # Results for words
    results_w11 = [results[setting]["c_w11"] for setting in results.keys()]
    results_w12 = [results[setting]["c_w12"] for setting in results.keys()]
    results_w21 = [results[setting]["c_w21"] for setting in results.keys()]
    results_w22 = [results[setting]["c_w22"] for setting in results.keys()]

    # Results for non-words
    results_nw11 = [results[setting]["c_nw11"] for setting in results.keys()]
    results_nw12 = [results[setting]["c_nw12"] for setting in results.keys()]
    results_nw21 = [results[setting]["c_nw21"] for setting in results.keys()]
    results_nw22 = [results[setting]["c_nw22"] for setting in results.keys()]

    # Plotting
    axes = [axs[0,0], axs[1,0], axs[0,1], axs[1,1]]
    data = [[results_w11, results_w12], [results_w21, results_w22], [results_nw11, results_nw12], [results_nw21, results_nw22]]
    lines = [length_correlations["lc_w1"], length_correlations["lc_w2"], length_correlations["lc_nw1"], length_correlations["lc_nw2"]]
    titles = ["Correlation with Reading Time for Words", "Correlation with Accuracy for Words", "Correlation with Reading Time for Non-Words", "Correlation with Accuracy for Non-Words"]
    # I need this as the bar labels
    sizes = [[ "",str(int(size/1000))+"k", ""] for size in vocab_sizes]
    sizes = [item for sublist in sizes for item in sublist]

    for i, ax in enumerate(axes):
        df = pd.DataFrame(data[i], columns=results.keys(), index=[feature1, feature2])
        df.reset_index().plot(x='index', kind='bar', stacked=False, title=titles[i],
                                ax=ax, color=mycolors, rot = 0,  legend=False, xlabel="", ylabel="Pearson Correlation", width=1)
        # add sizes as labels
        for container, label in zip(ax.containers, sizes):
            ax.bar_label(container, [label,label], fontsize=7, padding=15)

        if i == 0:
            ax.legend([a for a in algorithms], prop={'size': 10}, loc="upper right")

        ax.hlines(y=lines[i], xmin=-2, xmax=2, color='grey', linestyle='--', label = "Length")
        ax.hlines(y=0, xmin=-2, xmax=2, color='black')





    plt.savefig("../results/overview_" + language + ".png")
