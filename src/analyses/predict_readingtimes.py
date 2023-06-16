import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import seaborn as sns

# I did not finalize these analyses because there is not enough variance in the number of splits
# It would only work if we would predict binary classes (e.g. high vs low reading time and accuracy)
matplotlib.use("MacOSX")
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config["vocab_size"])
    outdir = config["outdir"]

    languages = config["language"]
    algorithms = ["BPE"]
    vocab_sizes = [30000, 50000]

# Calculate correlations and plot
for language in languages:
    print(language)
    results = {}
    measure1 = "Reading_Time"
    measure2 = "Accuracy"
    feature1 = "Num_Splits"
    feature2 = "Chunkability"
    length_correlations = {}


    for vocab_size in vocab_sizes:
        print(vocab_size)
        for alg in algorithms:
            print(alg)
            setting = alg + "_" + str(vocab_size)
            modeldir = outdir + language + "/trained_models/" + setting + "/"
            outputdata = pd.read_csv(modeldir + "output.csv", delimiter=",")
            outputdata = outputdata.dropna()
            test = outputdata.sample(n=1000)
            nonwords = outputdata[outputdata["Lexicality"] == "N"]
            words = outputdata[outputdata["Lexicality"] == "W"]

            sns.lmplot(
                data=test, x=measure1, y=feature2,
                hue="Lexicality")
            plt.show()


            split = int(0.8*len(words))

            # Words
            train_words = nonwords[0:split]
            test_words = nonwords[split:]

            train_length_feature = [[len(x)] for x in train_words["Stimulus"]]
            #train_features = [[x] for x in train_words[feature1]]
            train_features = list(zip(train_words[feature2], train_words[feature2]))
            #print(train_features)
            test_features = [[x] for x in test_words[feature1]]
            test_length_feature = [[len(x)] for x in test_words["Stimulus"]]
            test_features = list(zip(test_words[feature2], test_words[feature2]))
            train_labels = train_words[measure1].to_list()
            test_labels = test_words[measure1].to_list()

            model = linear_model.LinearRegression()
            model.fit(train_features, train_labels)
            training_predictions = model.predict(train_features)
            print("Fit within training data: ")
            print("Mean squared error: %.2f" % mean_squared_error(training_predictions, train_labels))
            print("Explained Variance: %.2f" % explained_variance_score(training_predictions, train_labels))
            print("R2: %.2f" % r2_score(training_predictions, train_labels))

            predictions = model.predict(test_features)
            print("Mean squared error: %.2f" % mean_squared_error(predictions, test_labels))
            print("Explained Variance: %.2f" % explained_variance_score(predictions, test_labels))
            print("R2: %.2f" % r2_score(predictions, test_labels))
            plt.scatter(predictions, test_labels)
            plt.title(language + ": " + setting)
            plt.savefig(modeldir + "predictions.png")

            print("Just Length: ")
            model = linear_model.LinearRegression()
            model.fit(train_length_feature, train_labels)
            predictions = model.predict(test_length_feature)
            print("Mean squared error: %.2f" % mean_squared_error(predictions, test_labels))
            print("Explained Variance: %.2f" % explained_variance_score(predictions, test_labels))
            print("R2: %.2f" % r2_score(predictions, test_labels))
            print("\n\n")

