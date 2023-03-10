import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config["vocab_size"])
    outdir = config["outdir"]

    languages = config["language"]
    algorithms = ["WPC"]
    vocab_sizes = [30000]

# Calculate correlations and plot
for language in languages:
    print(language)
    results = {}
    measure1 = "Reading_Time"
    measure2 = "Accuracy"
    feature1 = "Num_Splits"
    feature2 = "Wordiness"
    length_correlations = {}


    for vocab_size in vocab_sizes:
        print(vocab_size)
        for alg in algorithms:
            print(alg)
            setting = alg + "_" + str(vocab_size)
            modeldir = outdir + language + "/models/" + setting + "/"
            outputdata = pd.read_csv(modeldir + "output.csv", delimiter=",", index_col=0)
            outputdata = outputdata.dropna()
            nonwords = outputdata[outputdata["Lexicality"] == "N"]
            words = outputdata[outputdata["Lexicality"] == "W"]
            split = int(0.8*len(words))
            train = nonwords[0:split]
            test = nonwords[split:]

            train_features = list(zip(train[feature1], train[feature2]))
            #print(train_features)
            test_features = list(zip(test[feature1], test[feature2]))
            train_labels = train[measure1].to_list()
            test_labels = test[measure1].to_list()

            model = linear_model.LinearRegression()
            model.fit(train_features, train_labels)
            predictions = model.predict(test_features)
            print("Mean squared error: %.2f" % mean_squared_error(predictions, test_labels))
            plt.scatter(predictions, test_labels)
            plt.title(language)
            plt.show()

